from __future__ import absolute_import, division, print_function, unicode_literals

import os
import io
import re
import sys
import time
import numpy as np
import unicodedata
import tensorflow as tf

from data_loader import preprocess_ch_sentence, load_dataset, save_index2word, load_word2index, load_index2word, \
    save_max_length, load_max_length, init_jieba_dict, seg2words
from model import NNConfig, Encoder, BahdanauAttention, Decoder

PATH_TO_FILE = "./dataset/CMN_TRAD_SEG.txt"
SAVE_DIR = './checkpoints/training_checkpoints'
SAVE_PATH = os.path.join(SAVE_DIR, 'ckpt')  # 最佳验证结果保存路径

def train():
    print("Loading training data...")
    # Get sentences to tensors
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(PATH_TO_FILE, num_examples=None)
    
    # NNConfig
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_targ_size = len(targ_lang.word_index)+1
    config = NNConfig(vocab_inp_size, vocab_targ_size)
    
    # Save i2w file for test and translate
    save_index2word(inp_lang, "./dataset/input_dict.txt")
    save_index2word(targ_lang, "./dataset/target_dict.txt")
    save_max_length(input_tensor, target_tensor, vocab_inp_size, vocab_targ_size, "./dataset/max_len.txt")
    
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
    # preprocessing the original_sentence
    init_jieba_dict()
    original_sentence = input("\nPlease enter chinese sentence:")
    sentence = seg2words(original_sentence)
    sentence = preprocess_ch_sentence(sentence)
    print()
    INP_MAX_LEN, INP_VOCAB_SIZE, TARG_MAX_LEN, TARG_VOCAB_SIZE = load_max_length("./dataset/max_len.txt")
    INP_W2I = load_word2index("./dataset/input_dict.txt")
    TARG_I2W = load_index2word("./dataset/target_dict.txt")
    # __, __, inp_lang, targ_lang = load_dataset(PATH_TO_FILE, num_examples=None)
    # vocab_inp_size = len(inp_lang.word_index)+1
    # vocab_targ_size = len(targ_lang.word_index)+1
    
    # # NNConfig
    config = NNConfig(INP_VOCAB_SIZE, TARG_VOCAB_SIZE)
    encoder = Encoder(config.VOCAB_INP_SIZE, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE)
    decoder = Decoder(config.VOCAB_TARG_SIZE, config.EMBEDDING_DIM, config.UNITS, config.BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    #restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(SAVE_DIR))
    
    inputs = []
    
    for word in sentence.split(' '):
        if word in INP_W2I:
            inputs.append(INP_W2I[word])
        else:
            inputs.append(INP_W2I["ukn"])
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=INP_MAX_LEN, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, config.UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    # TARGET 1 <start>
    dec_input = tf.expand_dims([1], 0)

    for t in range(TARG_MAX_LEN):
        predictions, dec_hidden, __ = decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += TARG_I2W[predicted_id] + ' '

        if TARG_I2W[predicted_id] == '<end>':
            break

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    
    print("\n"+original_sentence+"->"+result.rstrip(" <end>")+"\n")

if __name__ == '__main__':

    if len(sys.argv) != 2 or sys.argv[1] not in ['train','translate']:
        raise ValueError("""usage: python run_nn.py [train/translate]""")

    if sys.argv[1] == 'train':
        train()
    
    if sys.argv[1] == 'translate':
        translate()

        