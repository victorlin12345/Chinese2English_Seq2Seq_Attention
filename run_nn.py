from __future__ import absolute_import, division, print_function, unicode_literals

import os
import io
import re
import sys
import time
import numpy as np
import unicodedata
import tensorflow as tf

from data_loader import load_dataset, save_index2word
from model import NNConfig

PATH_TO_FILE = "./dataset/CMN_TRAD_SEG.txt"
SAVE_DIR = 'checkpoints/training_checkpoints'
SAVE_PATH = os.path.join(SAVE_DIR, 'ckpt')  # 最佳验证结果保存路径

def train():
    print("Loading training data...")
    # Get sentences to tensors
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(PATH_TO_FILE, num_examples=None)
    
    # NNConfig
    config = NNConfig(inp_lang, targ_lang)
    
    # Save i2w file for test and translate
    save_index2word(inp_lang, "input_dict.txt")
    save_index2word(targ_lang, "target_dict.txt")

def translate():
    print("translate")
    # translate

if __name__ == '__main__':

    if len(sys.argv) != 2 or sys.argv[1] not in ['train']:
        raise ValueError("""usage: python run_nn.py [train]""")

    if sys.argv[1] == 'train':
        train()