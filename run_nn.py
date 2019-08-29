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
    save_index2word(inp_lang, "./dataset/input_dict.txt")
    save_index2word(targ_lang, "./dataset/target_dict.txt")

    
    # Setup the trainning data batch
    BUFFER_SIZE = len(input_tensor)
    steps_per_epoch = len(input_tensor)//config.BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(config.BATCH_SIZE, drop_remainder=True)
    
    print("Setting Seq2Seq model...")
    # Setup the NN Structure
    encoder = Encoder(config.VOCAB_INP_SIZE, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE)
    decoder = Decoder(config.VOCAB_TARG_SIZE, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE)
    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    # Setup Checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(inp, targ, enc_hidden, optimizer):
        loss = 0 
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * config.BATCH_SIZE, 1)
            
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
    
        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    print("Start training ...")
    for epoch in range(config.EPOCHS):
        
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            
            batch_loss = train_step(inp, targ, enc_hidden, optimizer)
            total_loss += batch_loss
        
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
  
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = SAVE_PATH)
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def translate():
    print("translate")
    # translate

if __name__ == '__main__':

    if len(sys.argv) != 2 or sys.argv[1] not in ['train']:
        raise ValueError("""usage: python run_nn.py [train]""")

    if sys.argv[1] == 'train':
        train()