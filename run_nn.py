from __future__ import absolute_import, division, print_function, unicode_literals

import os
import io
import re
import sys
import time
import numpy as np
import unicodedata
import tensorflow as tf

from data_loader import load_dataset

PATH_TO_FILE = "./dataset/CMN_TRAD_SEG.txt"
SAVE_DIR = 'checkpoints/training_checkpoints'
SAVE_PATH = os.path.join(SAVE_DIR, 'ckpt')  # 最佳验证结果保存路径

def train():
    print("Loading training data...")
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(PATH_TO_FILE, num_examples=None)
    print(len(input_tensor))

def translate():
    print("translate")
    # translate

if __name__ == '__main__':

    if len(sys.argv) != 2 or sys.argv[1] not in ['train']:
        raise ValueError("""usage: python run_nn.py [train]""")

    if sys.argv[1] == 'train':
        train()