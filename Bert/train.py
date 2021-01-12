import csv
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader
from model import BertClassifier
import os
labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
num_labels = len(labels)


def get_bert_input(text, tokenizer, max_len=512):
    '''
        生成单个句子的BERT模型的三个输入
        参数:
            text: 文本(单个句子)
            tokenizer: 分词器
            max_len: 文本分词后的最大长度
        返回值:
            input_id, attention_mask, token_type_id
    '''
    cls_token = '[CLS]'
    sep_token = '[SEP]'

    word_piece_list = tokenizer.tokenize(text)  # 分词
    input_id = tokenizer.convert_tokens_to_ids(word_piece_list)  # 把分词结果转成id
    if len(input_id) > max_len - 2:  # 如果input_id的长度大于max_len，则进行前后截取
        pre = input_id[:128]
        post = input_id[-382:]
        input_id = pre
        input_id.extend(post)
        # input_id = input_id[0:128].extend(input_id[-382:])
    # print(input_id)
    input_id = tokenizer.build_inputs_with_special_tokens(input_id)  # 对input_id补上[CLS]、[SEP]
    # print(input_id)

    attention_mask = []  # 注意力的mask，把padding部分给遮蔽掉, 也可记为position_id
    for i in range(len(input_id)):
        attention_mask.append(1)  # 句子的原始部分补1
    while len(attention_mask) < max_len:
        attention_mask.append(0)  # padding部分补0

    while len(input_id) < max_len:  # 如果句子长度小于max_len, 做padding，在句子后面补0
        input_id.append(0)

    token_type_id = [0] * max_len  # 第一个句子为0，第二个句子为1，第三个句子为0 ..., 也可记为segment_id

    assert len(input_id) == len(token_type_id) == len(attention_mask)

    return input_id, attention_mask, token_type_id


def load_data(filename):
    '''
    读取数据文件，将数据封装起来返回，便于BERT输入
    输入：
        数据文件名
    输出：
        已经处理好的适合BERT输入的dataset
    '''

    # print('加载数据ing')
    with open(filename, 'r', encoding='utf-8') as rf:
        data = csv.DictReader(rf)
        # data = rf.readlines()

        label_ids = []
        input_ids = []
        attention_mask = []
        token_type_ids = []

        tokenizer = BertTokenizer(vocab_file='./chinese_wwm_pytorch/vocab.txt')

        for item in data:
            label, text = item['class_label'], item['content']
            # print(label,text)
            # label转为对应的序号
            label_id = labels.index(label)
            # 得到bert的输入
            input_id, attention_mask_id, token_type_id = get_bert_input(text, tokenizer)

            # df拼接
            input_ids.append(input_id)
            attention_mask.append(attention_mask_id)
            token_type_ids.append(token_type_id)
            label_ids.append(label_id)
        # 生成用于训练的dataset
        input_ids = torch.tensor([i for i in input_ids], dtype=torch.long)
        attention_mask = torch.tensor([i for i in attention_mask], dtype=torch.long)
        token_type_ids = torch.tensor([i for i in token_type_ids], dtype=torch.long)
        label_ids = torch.tensor([i for i in label_ids], dtype=torch.long)
        dataset = TensorDataset(input_ids, attention_mask, token_type_ids, label_ids)
    return dataset


def main():
    device = torch.device('cuda:3')
    # 获取到dataset
    print('加载训练数据')
    train_data = load_data('dataset/train.csv')
    print('加载验证数据')
    valid_data = load_data('dataset/test.csv')
    # test_data = load_data('cnews/cnews.test.txt')

    batch_size = 16

    # 生成Batch
    print('生成batch')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=3)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=3)
    # test_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained('./chinese_wwm_pytorch')
    bert_config.num_labels = num_labels
    print(bert_config)

    # 初始化模型
    model = BertClassifier(bert_config)
    # model.to(device)

    # 参数设置
    EPOCHS = 20
    learning_rate = 5e-6  # Learning Rate不宜太大
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 损失函数采用交叉熵
    criterion = nn.CrossEntropyLoss()

    with open('output.txt', 'w') as wf:
        wf.write('Batch Size: ' + str(batch_size) + '\tLearning Rate: ' + str(learning_rate) + '\n')

    best_acc = 0
    # 设置并行训练,模型默认是把参数放在device[0]对应的gpu编号的gpu上，所以这里应该和上面设置的cuda:2对应
    net = torch.nn.DataParallel(model, device_ids=[3, 4])
    net.to(device)
    # model.module.avgpool = nn.AdaptiveAvgPool2d(7)
    # 开始训练 
    for Epoch in range(1, EPOCHS + 1):
        losses = 0  # 损失
        accuracy = 0  # 准确率
        print('Epoch:', Epoch)

        model.train()
        for batch_index, batch in enumerate(train_dataloader):
            # print(batch_index)
            # print(batch)
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            label_ids = batch[3].to(device)
            # 将三个输入喂到模型中
            output = net(  # forward
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = criterion(output, label_ids)
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_ids.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc
            # 打印训练过程中的准确率以及loss
            # print('Epoch: %d ｜ Train: | Batch: %d / %d | Acc: %f | Loss: %f' % (Epoch, batch_index + 1, len(train_dataloader), acc, loss.item()))
            # 模型梯度置零，损失函数反向传播，优化更新
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)
        # 打印该epoch训练结果的
        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)
        # with open('output.txt', 'a') as rf:
        #     output_to_file = '\nEpoch: ' + str(Epoch) + '\tTrain ACC:' + str(average_acc) + '\tLoss: ' + str(
        #         average_loss)
        #     rf.write(output_to_file)

        # 验证
        model.eval()
        losses = 0  # 损失
        accuracy = 0  # 准确率
        # 在验证集上进行验证
        for batch_index, batch in enumerate(valid_dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            label_ids = batch[3].to(device)
            with torch.no_grad():
                output = model(  # forward
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            loss = criterion(output, label_ids)
            losses += loss.item()
            # 这里的两部操作都是直接对生成的结果张量进行操作
            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_ids.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc

        average_loss = losses / len(valid_dataloader)
        average_acc = accuracy / len(valid_dataloader)

        print('\tValid ACC:', average_acc, '\tLoss:', average_loss)
        # with open('output.txt', 'a') as rf:
        #     output_to_file = '\nEpoch: ' + str(Epoch) + '\tValid ACC:' + str(average_acc) + '\tLoss: ' + str(
        #         average_loss) + '\n'
        #     rf.write(output_to_file)

        if average_acc > best_acc:
            best_acc = average_acc
            torch.save(model.state_dict(), 'best_model_on_trainset.pkl')


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
    main()
