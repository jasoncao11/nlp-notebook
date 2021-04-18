# -*- coding: utf-8 -*-
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK_TAG = "<UNK>"
PAD_TAG = "<PAD>"

EMBEDDING_DIM = 300
HIDDEN_DIM = 64
BATCH_SIZE = 512
EPOCHS = 100

CHAR_VOCAB_PATH = "./data/char_vocabs.txt" # 字典文件
TRAIN_DATA_PATH = "./data/train_data" # 训练数据
TEST_DATA_PATH = "./data/test_data" # 测试数据
