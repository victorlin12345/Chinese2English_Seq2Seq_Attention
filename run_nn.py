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
from model import NNConfig, Encoder, BahdanauAttention, Decoder

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

    
    # Setup the trainning data batch
    BUFFER_SIZE = len(input_tensor)
    steps_per_epoch = len(input_tensor)//config.BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(config.BATCH_SIZE, drop_remainder=True)
    
    # Setup the NN Flow
    example_input_batch, example_target_batch = next(iter(dataset))
    print("Input Batch Tensor Shape:",example_input_batch.shape, example_target_batch.shape)

    encoder = Encoder(config.VOCAB_INP_SIZE, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE)
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
    print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    
    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(config.VOCAB_TARG_SIZE, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)), sample_hidden, sample_output)
    print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
    
    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def translate():
    print("translate")
    # translate

if __name__ == '__main__':

    if len(sys.argv) != 2 or sys.argv[1] not in ['train']:
        raise ValueError("""usage: python run_nn.py [train]""")

    if sys.argv[1] == 'train':
        tf.enable_eager_execution()
        train()