import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score

def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    # 计算证据匹配项（直接影响准确率）交叉熵损失
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    # kl_alpha = y * (alpha - 1) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)

# def view

def get_syn_loss(evidence_a, target, epoch_num, num_classes, annealing_step, gamma, device):
    target = F.one_hot(target, num_classes).float()  # 类别和y
    alpha_a = evidence_a + 1     # 所有evidence和之后 +1
    loss_syn = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    return loss_syn

# 自身Y和Y['syn']一起监督
def get_lossYsingle(evidences, evidence_a, targets, epoch_num, num_classes, annealing_step, gamma, device):

    loss_syn = get_syn_loss(evidence_a, targets['syn'], epoch_num, num_classes, annealing_step, gamma,
                            device)
    # 对于每个视图v，其损失计算
    for v in range(1, len(evidences)-1):
        target_v = F.one_hot(targets[v], num_classes).float()
        alpha = evidences[v] + 1     # 计算第 v 个视图的参数 α ，可能用于 EDL（Evidence Deep Learning）损失函数
        loss_syn += edl_digamma_loss(alpha, target_v, epoch_num, num_classes, annealing_step, device)  # !!
    total_loss = loss_syn / (len(evidences))

    return total_loss


def get_MRR(evidence,Y_true):  #是top1的
    MRR = 0
    _, idx = torch.sort(evidence, descending=True)  # 找到对应的最大evidence的idx
    for i in range(len(Y_true)):
        a = torch.where(idx[i]==Y_true[i])[0]
        mrr = 1/(a+1)
        MRR += mrr
    MRR = MRR/len(Y_true)
    return MRR

def get_f1_recall_pre(evidence,Y_true): #是所有的
    _, Y_pre = torch.max(evidence, dim=1)
    average = 'micro'
    recall = recall_score(Y_true.cpu(), Y_pre.cpu(), average=average) # 整体召回率

    return recall