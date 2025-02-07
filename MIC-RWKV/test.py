import torch
from torch.utils.data.dataloader import DataLoader
import pickle
from easydict import EasyDict
import yaml
from src.dna_utils import collate_fn, DataGenerator
from src.model import MIC_RWKV
import os
import json


def test(model, loader, test_result_path):
    model.eval()
    mapping_tensor = torch.tensor([cls2val[i] for i in range(len(cls2val))], device='cuda:0')
    pbar = enumerate(loader)
    
    counter = 0
    nbCorrect = 0
    nbRaw = 0
    f_res = open(test_result_path, "w")
    f_res.write("PATRIC_ID, MIC, Predict MIC")

    for i, (x, y, mask, num_sequence_mask) in pbar:
        x = x.to('cuda')
        y = y.to('cuda')
        mask = mask.to('cuda')
        num_sequence_mask = num_sequence_mask.to('cuda')

        with torch.no_grad():
            _, pred, _, _ = model(x, y, mask, num_sequence_mask) 
            pred = torch.max(pred, 1)[1]
            pred = mapping_tensor[pred]
            y = mapping_tensor[y]

        for j in range(config.batch_size):
            if (pred[j] >= y[j] / 2 ) and pred[j] <= y[j] * 2:
                nbCorrect += 1
                if pred[j].item() == y[j].item():
                    nbRaw += 1
            f_res.write("\n" + str(loader.dataset.lst_data[i * config.batch_size + j]['PATRIC ID']) + "," + str(y[j].item()) + "," + str(pred[j].item()))
            counter += 1
                
    acc = nbCorrect / counter
    raw_acc = nbRaw / counter
    print(f'1-tier ACC: {acc}')
    print(f'Raw ACC: {raw_acc}')
    f_res.write("\n1-tier ACC: " + str(acc))
    f_res.write("\nRaw ACC: " + str(raw_acc))
    f_res.close()


if __name__ == '__main__':
    with open("./MIC-RWKV/config.yml", 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    cls2val = {v: int(k) if k.isdigit() else float(k) if k.replace('.', '', 1).isdigit() else k for k, v in config.val2cls[config.ANTIBIOTIC].items()}
    with open(f'./testset/{config.ANTIBIOTIC}_lstTest.json', 'r') as fr:
        lstTest = json.load(fr)
    test_generator = DataGenerator(config, lstTest)
    loader = DataLoader(test_generator, shuffle=False,
                    batch_size=config.batch_size,
                    num_workers=0,
                    drop_last=True,
                    collate_fn=lambda batch: collate_fn(batch, config.n_amr, config.ctx_len))
    ckpt = ''
    model_path = f'./{config.ckpt_path}/{config.ANTIBIOTIC}/{ckpt}'
    test_result_path = "results/" + config.ANTIBIOTIC.replace("/", "").replace(" ", "_") + "_test.csv"
    model = MIC_RWKV(config).cuda()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    
    test(model, loader, test_result_path)
