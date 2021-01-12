import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import pandas as pd
import visdom


def train(train_iter, test_iter, model, args):
    if args.cuda:
        model.cuda()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs + 1):
        print(train_iter)
        for batch in train_iter:
            model.train()

            feature, target = batch.content.to(device), batch.class_label.to(device)
            feature.t_(), target.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            # 计算loss
            loss = F.cross_entropy(logit, target)
            # loss反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            steps += 1
            # 进行一次当前训练效果的日志打印
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                # print(
                #     '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                #                                                              loss.item(),
                #                                                              accuracy.item(),
                #                                                              corrects.item(),
                #                                                              batch.batch_size))
            # 进行一次测试集上的评估
            if steps % args.test_interval == 0:
                dev_acc = eval(test_iter, model, args, epoch)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            # 进行一次模型的保存
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args, epoch):
    loss_list = []
    acc_list = []
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.content, batch.class_label
        feature.t_(), target.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        # print(target,logit)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    loss_list.append(avg_loss)
    acc_list.append(accuracy)
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    # visdom是用来数据可视化的
    # vis = visdom.Visdom(env='first')
    # for i in range(epoch):
    #     vis.line(X=torch.FloatTensor([i]),
    #              Y=torch.FloatTensor(acc_list[i]),
    #              win='first', update='append')
    return accuracy


# text预测的句子
def predict(text, model, text_field, label_feild, cuda_flag):
    # assert isinstance(text, str)
    model.eval()
    df = pd.DataFrame(columns=['label'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # text = text_field.tokenize(text)
    # i = 0
    for text in text:

        # 预处理文本
        text = text_field.preprocess(text)
        # print(text)
        text = [[text_field.vocab.stoi[x] for x in text]]
        x = torch.tensor(text)
        x = autograd.Variable(x)
        if cuda_flag:
            x = x.to(device)
        # print(x)
        output = model(x).to(device)
        # max是从几个类别的置信数组中得到置信度最大的
        _, predicted = torch.max(output, 1)
        new = {"label": no2label(predicted[0].item() + 1)}
        df = df.append(new, ignore_index=True)

    return df


def no2label(label_no):
    label_text = {
        1: "房产",
        2: "财经",
        3: "教育",
        4: "科技",
        5: "时尚",
        6: "家居",
        7: "时政",
        8: "娱乐",
        9: "体育",
        10: "游戏",
    }
    return label_text.get(label_no, None)


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
