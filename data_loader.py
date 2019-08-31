from __future__ import absolute_import, division, print_function, unicode_literals

import re
import os
import io
import jieba
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

def load_word2index(file_name):
    word2index = {}
    for line in open(file_name, "r"):
        index, word = line.strip().split("\t")
        word2index[word] = int(index)
    return word2index

def load_index2word(file_name):
    index2word = {}
    for line in open(file_name, "r"):
        index, word = line.strip().split("\t")
        index2word[int(index)] = word
    return index2word

# Get the max length of sentence in the list 
def save_max_length(inp_tensor, targ_tensor, vocab_inp_size, vocab_targ_size, file_name):
    file = open(file_name, "w")
    inp_len = max(len(t) for t in inp_tensor)
    targ_len = max(len(t) for t in targ_tensor)
    file.write("INPUT\t"+str(inp_len)+"\n")
    file.write("INPUT_VOCAB\t"+str(vocab_inp_size)+"\n")
    file.write("TARGET\t"+str(targ_len)+"\n")
    file.write("TARGET_VOCAB\t"+str(vocab_targ_size)+"\n")
    file.close()

def load_max_length(file_name):
    MAX_LEN = {}
    for line in open(file_name, "r"):
        k, v = line.strip().split("\t")
        MAX_LEN[k] = int(v)
    return MAX_LEN["INPUT"], MAX_LEN["INPUT_VOCAB"], MAX_LEN["TARGET"], MAX_LEN["TARGET_VOCAB"]

def init_jieba_dict():
    jieba.load_userdict("dataset/jieba_dict.txt")

# seg the chinese sentence
def seg2words(original_sentense):
    words = jieba.cut(original_sentense, cut_all=False)
    return_word=''
    for w in words:
        return_word = return_word+' '+w
    return return_word.lstrip()



