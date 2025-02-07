########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
import json
from src.model import MIC_RWKV
from src.trainer import Trainer
import torch
import numpy as np
from src.dna_utils import getListData, split_dataset, DataGenerator
from easydict import EasyDict
import yaml
import random

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


with open("./MIC-RWKV/config.yml", 'r') as f:
    config = EasyDict(yaml.safe_load(f))

config.epoch_save_path = f'./{config.ckpt_path}/{config.ANTIBIOTIC}/trained-'

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

lstData = getListData(config, config.ANTIBIOTIC)
NB_SAMPLES = len(lstData)

train_index, val_index, test_index = split_dataset(NB_SAMPLES)    

lstTrain = [lstData[i] for i in train_index]
lstVal = [lstData[i] for i in val_index]
lstTest = [lstData[i] for i in test_index]
# Save the test set randomly split during each training and load this file during testing.
with open(f'./testset/{config.ANTIBIOTIC}_lstTest.json', 'w') as f:
    f.write(json.dumps(lstTest))

print("lstTrain: ", len(lstTrain))
print("lstVal: ", len(lstVal))
print("lstTest: ", len(lstTest))
lstTrain = lstTrain[:20]
lstVal = lstVal[:8]
training_generator = DataGenerator(config, lstTrain)
validation_generator = DataGenerator(config, lstVal)


if __name__ == '__main__':

    TEST_RESULT_FILE = config.ANTIBIOTIC.replace("/", "").replace(" ", "_") + "_test.csv"

    model = MIC_RWKV(config).cuda()

    print('epoch', config.max_epochs, 'batchsz', config.batch_size, 'ctx', config.ctx_len, 'layer', config.n_layer, 'embd', config.n_embd)
    trainer = Trainer(model, training_generator, validation_generator, config)
    trainer.train()
