########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from torch.utils.cpp_extension import load
import math
import numpy as np
import os
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)


########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = 2080          # increase this if your ctx_len > 1024
B_GROUP_FORWARD = 4   # set to 8 for best performance
B_GROUP_BACKWARD = 2  # set to 2 for best performance

timex_cuda = load(name="timex", sources=["./MIC-RWKV/cuda/wkv_op.cpp", "./MIC-RWKV/cuda/wkv_cuda.cu"],
                  verbose=True, extra_cuda_cflags=['--use_fast_math', '--extra-device-vectorization', f'-DTmax={T_MAX}', f'-DBF={B_GROUP_FORWARD}', f'-DBB={B_GROUP_BACKWARD}'])



class TimeX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C

        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        timex_cuda.forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, C, T), device='cuda').contiguous()
        gv = torch.zeros((B, C, T), device='cuda').contiguous()
        timex_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        gw = gw.sum(dim=0, keepdim=True)
        gu = gu.sum(dim=0, keepdim=True)
        return (None, None, None, gw, gu, gk, gv)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


RWKV_K_CLAMP = 60  # e^60 = 1e26
RWKV_K_EPS = 1e-16
RWKV_HEAD_QK_DIM = 8


def RWKV_Init(module, config):  # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters():  # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0
            scale = 1.0 

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == config.vocab_size and shape[1] == config.n_embd:
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == config.vocab_size and shape[1] == config.n_embd:
                    scale = 0.5

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            gain *= scale
            if scale == -999:
                nn.init.eye_(m.weight)
            elif gain == 0:
                # zero init is great for some RWKV matrices
                nn.init.zeros_(m.weight)
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=-scale)


class RWKV_TimeMix(nn.Module):
    def __init__(self, config, ctx_len, n_embd, attn_sz, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.n_embd = n_embd

        attn_sz = n_embd

        ############# fancy init of time_w curves ###################################
        f1_begin = 3.0
        f1_end = 1.2
        f2_begin = 0.65
        f2_end = 0.4
        with torch.no_grad():  # initial time_w curves for better convergence
            decay_speed = torch.ones(attn_sz, 1)
            first_sa_layer_id = 1
            for h in range(attn_sz):
                f1 = f1_begin + (layer_id-first_sa_layer_id) / \
                    (config.n_layer-1-first_sa_layer_id) * (f1_end - f1_begin)
                f2 = f2_begin + (layer_id-first_sa_layer_id) / \
                    (config.n_layer-1-first_sa_layer_id) * (f2_end - f2_begin)
                if layer_id == first_sa_layer_id:
                    f1 += 0.5
                if layer_id == config.n_layer-2:
                    f2 = 0.4
                if layer_id == config.n_layer-1:
                    f2 = 0.37
                decay_speed[h][0] = math.pow(f2, h / (attn_sz-1) * 7) * f1
        self.time_decay = nn.Parameter(torch.log(decay_speed)) # will use exp(self.time_decay) to ensure time_decay > 0
        self.time_curve = torch.tensor(
            [-(self.ctx_len - 2 - i) for i in range(self.ctx_len-1)]).unsqueeze(0)
        self.time_curve = self.time_curve.to('cuda')
        self.time_first = nn.Parameter(torch.ones(attn_sz, 1) * math.log(0.3))
        #############################################################################

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():  # init to "shift half of the channels"
            ww = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd // 2):
                ww[0, 0, i] = 0
        self.time_mix = nn.Parameter(ww)

        self.key = nn.Linear(self.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(self.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(self.n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, self.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.combined_mix = nn.Parameter(torch.full((1, 1, self.n_embd), 0.01))


    def forward(self, x):
        B, T, C = x.size()
        x_t_minus_1 = torch.cat([torch.zeros(B, 1, C, device='cuda'), x[:, :-1, :]], dim=1)
        x_t_plus_1 = torch.cat([x[:, 1:, :], torch.zeros(B, 1, C, device='cuda')], dim=1)
        x_t_minus_1_half = x_t_minus_1[:, :, :C//2]
        x_t_plus_1_half = x_t_plus_1[:, :, C//2:]

        x_combined = torch.cat([x_t_minus_1_half, x_t_plus_1_half], dim=-1)

        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix) + x_combined * self.combined_mix

        k = self.key(x).transpose(-1, -2)
        v = self.value(x).transpose(-1, -2)
        r = self.receptance(x)

        k = torch.clamp(k, max=RWKV_K_CLAMP)
        k = torch.exp(k)
        wkv = TimeX.apply(B, T, C, self.time_decay.transpose(-1, -2) / T, self.time_first.transpose(-1, -2) / T, k, v)
        rwkv = torch.sigmoid(r) * wkv
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, layer_id):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # init to "shift half of the channels"
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd // 2):
                x[0, 0, i] = 0
        self.time_mix = nn.Parameter(x)

        hidden_sz = 4 * self.n_embd
        self.key = nn.Linear(self.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, self.n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

        self.combined_mix = nn.Parameter(torch.full((1, 1, self.n_embd), 0.01))
    def forward(self, x):
        B, T, C = x.size()
        x_t_minus_1 = torch.cat([torch.zeros(B, 1, C, device='cuda'), x[:, :-1, :]], dim=1)
        x_t_plus_1 = torch.cat([x[:, 1:, :], torch.zeros(B, 1, C, device='cuda')], dim=1)
        x_t_minus_1_half = x_t_minus_1[:, :, :C//2]
        x_t_plus_1_half = x_t_plus_1[:, :, C//2:]
        x_combined = torch.cat([x_t_minus_1_half, x_t_plus_1_half], dim=-1)

        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix) + x_combined * self.combined_mix

        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(x)) * kv
        return rkv


class Block(nn.Module):
    def __init__(self, config, layer_id, mode):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if mode == 'intra':
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.ln2 = nn.LayerNorm(config.n_embd)
            self.att = RWKV_TimeMix(config, config.ctx_len, config.n_embd, config.n_embd, layer_id)
            self.ffn = RWKV_ChannelMix(config.n_embd, layer_id)
        elif mode == 'inter':
            self.ln1 = nn.LayerNorm(config.cross_hid)
            self.ln2 = nn.LayerNorm(config.cross_hid)
            self.att = RWKV_TimeMix(config, config.n_amr, config.cross_hid, config.cross_hid, layer_id)
            self.ffn = RWKV_ChannelMix(config.cross_hid, layer_id)
        else:
            raise ValueError(f"mode {mode} not supported")


    def forward(self, x):
        x = self.ln1(x)
        x = x + self.att(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x


class GlobalAttention(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.num_heads = config.num_heads
        self.q_ln = nn.Linear(dim, 256)
        self.k = 64
        self.mk = nn.Linear(256 // self.num_heads, self.k)
        self.mv = nn.Linear(self.k, dim // self.num_heads)

    def forward(self, x):
        B, n_amr, T = x.shape
        q = self.q_ln(x)
        q = q.view(B, n_amr, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = self.mk(q)
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        x = self.mv(attn).permute(0, 2, 1, 3).reshape(B, n_amr, -1)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        euclidean_distance = F.pairwise_distance(x1, x2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, preds, target):
        one_hot = F.one_hot(target, self.num_classes).float()
        smoothed_labels = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        log_probs = F.log_softmax(preds, dim=-1)
        loss = -torch.sum(log_probs * smoothed_labels, dim=-1).mean()
        return loss
    

class MIC_RWKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config
        val2cls = config.val2cls[config.ANTIBIOTIC]
        num_cls = len(val2cls)
        self.contra_loss = ContrastiveLoss()

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i, 'intra')
                                    for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

        self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM)
        self.head_q.scale_init = 0
        self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM)
        self.head_k.scale_init = 0.1
        self.register_buffer("copy_mask", torch.tril(
            torch.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len


        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))
        
        self.ln_flatten = nn.Sequential(nn.Linear(config.ctx_len, 128), nn.Dropout(0.5), nn.Linear(128, num_cls))
        self.ln_mapping = nn.Sequential(nn.Linear(config.ctx_len, config.cross_hid * 2), nn.ReLU(), nn.Dropout(0.5), nn.Linear(config.cross_hid * 2, config.cross_hid))
        self.blocks_inter = nn.Sequential(*[Block(config, i, 'inter')
                                    for i in range(6)])
        self.ln_out_inter = nn.LayerNorm(config.cross_hid)

        self.head_q_inter = nn.Linear(config.cross_hid, RWKV_HEAD_QK_DIM)
        self.head_k_inter = nn.Linear(config.cross_hid, RWKV_HEAD_QK_DIM)

        self.global_attn = GlobalAttention(config, config.ctx_len)
        self.criterion = LabelSmoothingLoss(num_cls, smoothing=0.1)

        RWKV_Init(self, config)


    def forward(self, idx, targets, mask=None, num_seq_mask=None):
        self.step += 1
        B, n_amr, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        # intra
        intra_amr_atten = torch.zeros(B, n_amr, T).to(idx.device)
        for i in range(B):
            x = self.emb(idx[i])
            x = self.blocks(x)
            x = self.ln_out(x)
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            mask_tmp = mask[i] * mask[i].transpose(-2, -1)
            c = c.masked_fill(mask_tmp == 0, 0)
            x = c @ x
            x = torch.max(x, dim=2)[0]
            intra_amr_atten[i] = x

        # inter
        x_inter = self.blocks_inter(self.ln_mapping(intra_amr_atten))
        x_inter = self.ln_out_inter(x_inter)

        q_inter = self.head_q_inter(x_inter)[:, :n_amr, :]
        k_inter = self.head_k_inter(x_inter)[:, :n_amr, :]
        c_inter = (q_inter @ k_inter.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
        c_inter = c_inter.masked_fill(self.copy_mask[:n_amr, :n_amr] == 0, 0)

        softmax_attn = torch.zeros_like(c_inter)
        for b in range(B):
            sum_mask = int(torch.sum(num_seq_mask[b]).item())
            sub_attn = c_inter[b, :sum_mask, :sum_mask]
            sub_attn = F.softmax(sub_attn, dim=-1)
            softmax_attn[b, :sum_mask, :sum_mask] = sub_attn
        weighted_x = softmax_attn @ intra_amr_atten
        # global
        g_attn = self.global_attn(weighted_x)
        weighted_x = weighted_x + g_attn
        weighted_x = torch.max(weighted_x, dim=1)[0]
        output = self.ln_flatten(weighted_x)

        loss = self.criterion(output, targets)

        contrastive_loss = 0
        for b1 in range(B):
            for b2 in range(b1 + 1, B):
                label = 1 if targets[b1] != targets[b2] else 0
                contrastive_loss += self.contra_loss(weighted_x[b1], weighted_x[b2], torch.tensor(label, dtype=torch.float))
        contrastive_loss /= B * (B - 1) / 2

        return weighted_x, output, loss, contrastive_loss




