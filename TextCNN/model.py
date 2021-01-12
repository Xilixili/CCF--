import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        # 设置为词的个数
        V = args.embed_num
        #
        D = args.embed_dim
        # 输出类别数
        C = args.class_num
        Ci = 1
        # 卷积核的数目，TextCNN中的是100
        Co = args.kernel_num
        # 卷积核的大小，TextCNN用的是[3,4,5]
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        # 这里利用ModuleList容器构建了三个卷积层，其卷积核大小分别是3*D,4*D,5*D(其中D代表了隐含层的大小),注意这里的三个convs是并行的
        # 注意
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        # self.fc1 = nn.Linear(len(Ks) * Co, C)
        # 这个trick来观察增加分类层是否会对模型带来增益,分别测试2-5层
        self.fc1 = nn.Sequential(
            nn.Linear(len(Ks) * Co, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, C)
            # nn.Linear(len(Ks) * Co, 1024)
        )
        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
