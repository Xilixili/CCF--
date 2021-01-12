# -*- coding:utf-8 -*-
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import pandas as pd

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=128, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=list, default=[3, 4, 5],
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default="cuda:0", help='cuda_id')
parser.add_argument('-cuda', action='store_true', default=True, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None,
                    help='filename of model snapshot [default: None]')
# parser.add_argument('-snapshot', type=str, default="snapshot/2020`-11-16_21-41-43/best_steps_300.pt", help='filename of model snapshot [default: None]')
# parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-predict', action='store_true', default=False, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-class_num', type=int, default=10, help='train or test')
args = parser.parse_args()



# load data
print("\nLoading data...")
# args.embed_num = 100
# cnn = model.CNN_Text(args)
# print(cnn)
# text_field = data.Field(lower=True)
# label_field = data.Field(sequential=False)
# train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)

text_field, label_field, train_iter, val_iter, test_iter = mydatasets.load_dataset(args)

# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
args.embed_num = len(text_field.vocab)
print(label_field)
# 打印配置
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

print("\nParameters:")
# torch.cuda.set_device(0)
# 设置device
device_id = args.device
device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
cnn = cnn.to(device)

# train or predict
# 预测
if args.predict:
    text = pd.read_csv('dataset/test_data.csv', encoding='UTF-8')['content']
    # id = pd.read_csv('dataset/train/unlabeled_data.csv', encoding='UTF-8')['id']
    # print(id)
    label = train.predict(text, cnn, text_field, label_field, args.cuda)
    # res = text
    # print(text)
    # print(label)
    # text.to_csv('text.csv', encoding='UTF-8')
    # label.to_csv('label.csv', encoding='UTF-8')
    res = pd.DataFrame(columns=['content', 'label'])
    res['content'] = text
    res['label'] = label
    print(res)
    res.to_csv('res2.csv', encoding='UTF-8')
    # print(label)
    # print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
# 测试
elif args.test:
    try:
        train.eval(test_iter, cnn, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
# 训练
else:
    print()
    try:
        train.train(train_iter, test_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
