import re
import os
import random
import tarfile
import urllib
from torchtext import data
import jieba


def tokenizer(x):
    res = [w for w in jieba.cut(x)]
    return res


def load_dataset(args):
    stop_words = []
    print('build stop words set')
    with open('dataset/stopwords.dat', encoding='UTF-8') as f:
        for l in f.readlines():
            stop_words.append(l.strip())

    TEXT = data.Field(sequential=True, tokenize=tokenizer, fix_length=1000, stop_words=stop_words)
    LABEL = data.Field(sequential=False, use_vocab=False)
    #
    train, valid, test = data.TabularDataset.splits(path='dataset', train='train.csv',
                                                    validation='valid.csv', test='test.csv',
                                                    format='csv',
                                                    skip_header=True, csv_reader_params={'delimiter': ','},
                                                    fields=[(None, None), ('class_label', LABEL), ('content', TEXT)])
    TEXT.build_vocab(train)

    train_iter, val_iter, test_iter = data.Iterator.splits((train, valid, test),
                                                           batch_sizes=(
                                                           args.batch_size, args.batch_size, args.batch_size),
                                                           device=args.device,
                                                           sort_key=lambda x: len(x.content),
                                                           sort_within_batch=False,
                                                           repeat=False)
    return TEXT, LABEL, train_iter, val_iter, test_iter
