from __future__ import absolute_import, division, print_function, unicode_literals

import re
import os
import io
import tensorflow as tf

def preprocess_eng_sentence(w):
    w = w.lower().strip()

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?'.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

def preprocess_ch_sentence(w):

    w = w.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, CHINESE]
def create_dataset(path, num_examples):
    
    word_pairs = []
    
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
   
    for line in lines[:num_examples]:
        eng, ch = line.strip().split("\t")
        word_pairs.append([preprocess_eng_sentence(eng), preprocess_ch_sentence(ch)])

    return zip(*word_pairs)

# Get the max length of sentence in the list 
# def max_length(tensor):
#     return max(len(t) for t in tensor)

# Transform the word to token.
def tokenize(lang):
    # Tokenizer: notice word token pairs always same for same corpus 
    # and token start from 1, 0 will be keep
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # padding: 'post': pad after each sequence.
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


# Output either input or target token seqence and their Tokenizer
def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

# Print word sequence convert from token sequence
def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, lang.index_word[t]))


# Prepare word2index file for test or translate
def save_index2word(lang, file_name):
    file = open(file_name, "w")
    for index, word in lang.index_word.items():
        file.write(str(index)+"\t"+word+"\n")
    file.close()

