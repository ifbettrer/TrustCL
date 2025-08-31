import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import csv
from data import CT
from model import TrustCL
from loss_function import get_MRR, get_f1_recall_pre,get_lossYsingle

def train1(model_lst):
    models = dict()
    for i, model in enumerate(model_lst):
        models[i] = model(num_views, dims, num_classes)  # 模型实例化
        models[i] = models[i].to(device)
    epochs = 260    # 原先是250
    for epoch in range(1, epochs + 1):
        print(f'====> {epoch}')
        for X, Y, indexes in train_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
                Y[v] = Y[v].to(device)
            Y['syn'] = Y['syn'].to(device)

            for model in models.values():
                train2(X, Y, model, epoch, device)
    return models

def train2(X, Y, model, epoch, device):
    loss_f = get_lossYsingle
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    gamma = 1
    model.train()

    evidences, evidence_a, Dirichlet_weight, _, attn_weights = model(X)  # , attn_weights
    loss = loss_f(evidences, evidence_a, Y, epoch, num_classes,
                      annealing_step=50, gamma=gamma, device=device)
    optimizer.zero_grad()
    loss.backward()

    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()

def eval_model(mrr, recall, X, Y, a, model, device, times):
    model.eval()
    for v in range(num_views):
        X[v] = X[v].to(device)
        Y[v] = Y[v].to(device)
    Y['syn'] = Y['syn'].to(device)
    with torch.no_grad():
        evidences, evidence_a, Dirichlet_weight, _, attn_weights = model(X)  # , attn_weights

        MRR = get_MRR(evidence_a, Y['syn'])
        Recall = get_f1_recall_pre(evidence_a, Y['syn'])
        mrr[model.name] += MRR
        recall[model.name] += Recall
        for i in range(num_views):
            _, Y_pre = torch.max(evidences[i], dim=1)    ###
            a[model.name][0][i] += (Y_pre == Y[i]).sum().item()
            a[model.name][1][i] += Y[i].shape[0]
        _, Y_pre = torch.max(evidence_a, dim=1)     ###
        a[model.name][0]['syn'] += (Y_pre == Y['syn']).sum().item()
        a[model.name][1]['syn'] += Y['syn'].shape[0]

def get_mean_final(MRRs):
    mean_MRR, final_MRR = torch.split(MRRs, [6, 1], dim=1)
    mean_MRR = torch.mean(mean_MRR, dim=1)
    final_MRR = final_MRR.squeeze()

    return mean_MRR, final_MRR

def main_eval(a, models, MRRs, Recall, acc, times=0):
    for i, model in enumerate(models.values()):
        num = 0
        model.eval()
        c, d = dict(), dict()
        for v in range(6):
            c[v], d[v] = 0, 0
        c['syn'], d['syn'] = 0, 0
        a[model.name] = [c, d]

        # print("c和d的值分别为：", c, d)

        for X, Y, indexes in test_loader:
            num += 1
            for v in range(num_views):
                X[v] = X[v].to(device)
                Y[v] = Y[v].to(device)
            Y['syn'] = Y['syn'].to(device)

            with torch.no_grad():
                evidences, evidence_a, Dirichlet_weight, _, attn_weights = model(X)  # , attn_weights

                for v in range(num_views):
                    MRRs[i][v] = MRRs[i][v] + get_MRR(evidences[v], Y[v])
                    Recall[i][v] = Recall[i][v] + get_f1_recall_pre(evidences[v], Y[v])

                MRRs[i][-1] = MRRs[i][-1] + get_MRR(evidence_a, Y['syn'])
                Recall[i][-1] = Recall[i][-1] + get_f1_recall_pre(evidence_a, Y['syn'])

                # 单个视图预测
                for v in range(num_views):
                    _, Y_pre = torch.max(evidences[v], dim=1)
                    a[model.name][0][v] += (Y_pre == Y[v]).sum().item()
                    a[model.name][1][v] += Y[v].shape[0]

                _, Y_pre = torch.max(evidence_a, dim=1)
                a[model.name][0]['syn'] += (Y_pre == Y['syn']).sum().item()
                a[model.name][1]['syn'] += Y['syn'].shape[0]

        # print("c和d的值分别为：", c, d)

        MRRs[i] = MRRs[i] / num
        Recall[i] = Recall[i] / num
        sum_acc = 0
        for v in range(num_views):
            sum_acc += (a[model.name][0][v] / a[model.name][1][v])
        print('====> mean_self_acc: {:.4f}'.format(sum_acc / num_views))
        acc[i][0] = sum_acc / num_views  # 平均模态预测
        print('====> total_acc: {:.4f}'.format(a[model.name][0]['syn'] / a[model.name][1]['syn']))
        acc[i][1] = a[model.name][0]['syn'] / a[model.name][1]['syn']  # 融合预测

    mean_MRR, final_MRR = get_mean_final(MRRs)
    mean_Recall, final_Recall = get_mean_final(Recall)

    return acc, mean_MRR, final_MRR, mean_Recall, final_Recall


dataset = CT()
num_samples = len(dataset)
num_classes = dataset.num_classes
num_views = dataset.num_views
dims = dataset.dims
index = np.arange(num_samples)
np.random.shuffle(index)
train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

# create a test set with conflict instances
dataset.postprocessing(test_index, addNoise=False, addConflict=False)

dataset.postprocessing(train_index, addNoise=True, addConflict=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(Subset(dataset, train_index), batch_size=200, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_index), batch_size=200, shuffle=False)

model_lst = [TrustCL]

def main():
    models = train1(model_lst)
    a, mrr, recall = dict(), dict(), dict()
    Recalls, MRRs = torch.zeros(9, 7), torch.zeros(9, 7)
    acc = torch.zeros(9, 2)
    acc, mean_MRR, final_MRR, mean_Recall, final_Recall = main_eval(a, models, MRRs, Recalls, acc)

    # torch.save(models.state_dict(), 'trustcl.pth')

    print('====> mean_MRR:{:.4f}, final_MRR:{:.4f}, mean_Recall:{:.4f}, final_Recall:{:.4f}'.format(
        mean_MRR[0], final_MRR[0], mean_Recall[0], final_Recall[0]))


np.set_printoptions(precision=4, suppress=True)
torch.autograd.set_detect_anomaly(True)
main()

