import torch
import torch.nn as nn
import torch.distributions.dirichlet as dirichlet

class Attentions(nn.Module):
    def __init__(self, input_dim):
        super(Attentions, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        weighted_x = x * attn_weights
        return attn_weights, weighted_x

class MLP(nn.Module):
    def __init__(self, dims, num_classes, num_views):
        super(MLP, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Flatten(),
                    nn.Linear(dims[0], 256),
                    nn.ReLU(),
                    nn.Linear(256, num_views), nn.Softplus())
    def forward(self, x):
        h = self.net(x)
        return h

# 输出一个概率分布，表示属于每个类别的概率
class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes,ratio=4):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()

        self.attentions = Attentions(dims[0])

        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(dims[self.num_layers - 1], 256, bias=False))
        self.net.append(nn.ReLU())
        self.net.append(nn.Linear(256,  num_classes, bias=False))
        self.net.append(nn.Softplus())  # 添加一个softplus层保证结果非负

    def forward(self, x):
        attn_weights, weighted_x = self.attentions(x)
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        # return h
        return h, attn_weights

#迪利克雷分布
class TrustCL(nn.Module):
    def __init__(self, num_views, dims, num_classes):     # num_views（视图的数量），dims（每个视图的特征维度列表），num_classes（类别的数量）
        super(TrustCL, self).__init__()
        self.name = 'TrustCL'
        self.num_views = num_views
        #self.atten = atten
        self.num_classes = num_classes
        self.e_parameters = nn.Parameter(
            torch.tensor([0.2910, 0.1555, 0.1661, 0.1386, 0.1071, 0.1417]))

        # 包含num_views + 1个模块。
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)]   # 前num_views个模块是EvidenceCollector实例，用于从每个视图中提取证据，
        + [MLP(dims[0], num_classes, num_views)])   # 最后一个模块是一个多层感知机（MLP），用于融合证据

    # X，这是一个包含多个视图数据的列表
    def forward(self, X):
        # get evidence
        evidences = dict()
        attn_weights = dict()
        weighted_evidences = dict()

        evidences['wg'] = self.EvidenceCollectors[-1](X[0])    # 使用最后一个模块（MLP）处理第一个视图的数据，得到全局证据 D

        fuse_weight = torch.FloatTensor(evidences['wg'].size()).normal_()    # 创建一个与全局证据相同大小的张量，并用正态分布初始化
        std = evidences['wg'].mul(0.5).exp_()       # 计算全局证据的标准差
        fuse_weight = nn.functional.softplus(fuse_weight.mul(self.e_parameters).add_(std))
        poster = torch.tensor([1, 0.15120968, 0.64112903, 0, 0.55645161, 0.1733871], dtype=torch.float32)

        Dirichlet = dirichlet.Dirichlet((fuse_weight + poster))  # 创建一个Dirichlet分布对象，使用α
        dir_fuse_weight = Dirichlet.sample()  # 从Dirichlet分布中采样得到融合权重

        # 遍历每个视图，使用采样得到的融合权重调整每个视图的证据 c * e
        for v in range(self.num_views):
            evidences[v], attn_weights[v] = self.EvidenceCollectors[v](X[v])
            # evidences[v] = self.EvidenceCollectors[v](X[v])
            weighted_evidences[v] = evidences[v] * dir_fuse_weight[:, v:v + 1]
            # evidences[v] = self.EvidenceCollectors[v](X[v]) * dir_fuse_weight[:, v:v + 1]

        evidence_a = evidences[0].clone() * dir_fuse_weight[:, 0:0 + 1]    # 初始化一个变量来存储累积的证据
        for i in range(1, self.num_views):   # 累加过程
            evidence_a += weighted_evidences[i]
        # e_parameters 固有重要性, dir_fuse_weight权重
        return evidences, evidence_a, fuse_weight.tolist(), self.e_parameters.detach(), attn_weights
