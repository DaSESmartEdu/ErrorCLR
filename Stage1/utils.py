import torch
from sklearn import metrics
import numpy as np

def euclidean_distance(x, y):
    return torch.sum((x - y) ** 2, dim=-1)


def approximate_hamming_similarity(x, y):
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


def pairwise_loss(x, y, labels, loss_type='margin', margin=1.0):
    labels = labels.float()

    if loss_type == 'margin':
        return torch.relu(margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == 'hamming':
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)


def exact_hamming_similarity(x, y):
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)


def compute_similarity(x, y, loss_type='margin'):
    if loss_type == 'margin':
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif loss_type == 'hamming':
        return exact_hamming_similarity(x, y)

def auc(scores, labels):
    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)
    labels = (labels + 1) / 2

    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return metrics.auc(fpr, tpr)

def localization_loss(probs, labels):
    loss_func = torch.nn.BCELoss()
    labels = (labels > 0).float()
    loss = loss_func(probs[:, 1:].sum(dim=1), labels)
    return loss

def classification_loss(probs, labels):
    loss_func = torch.nn.NLLLoss()
    loss = loss_func(torch.log(probs), labels)
    return loss


def calculate_metrics(probs_1, probs_2, labels, k, graphs):
    suspicious_score = probs_1[:, 1:]
    suspicious_score = suspicious_score.sum(dim=1)
    possible_error_type_1 = probs_1[:, 1:].argmax(dim=1) + 1
    possible_error_type_2 = probs_2[:, 1:].argmax(dim=1) + 1
    lineids = [i['lineid'] for i in graphs]
    ptr = 0
    top1_count = 0
    top5_count = 0
    top10_count = 0
    error_count = 0
    type_1_count = 0
    type_2_count = 0
    predict_label_1 = []
    predict_label_2 = []
    true_label_1 = []
    true_label_2 = []


    for g in lineids:
        # localization ranking
        g_score = suspicious_score[ptr:ptr+len(g)]
        label = labels[0][ptr:ptr+len(g)]

        top_lines = torch.topk(g_score, k=min(k, len(g))).indices.cpu().detach().numpy().tolist()
        error_lines_pos = torch.nonzero(label).view(-1).cpu().detach().numpy().tolist()
        hit_lines = list(set(top_lines).intersection(set(error_lines_pos)))

        top5_count += len(hit_lines)
        error_count += len(error_lines_pos)

        # classification score
        type_1_count += (label[hit_lines] == possible_error_type_1[ptr:ptr+len(g)][hit_lines]).sum()
        type_2_count += (labels[1][ptr:ptr+len(g)][hit_lines] == possible_error_type_2[ptr:ptr+len(g)][hit_lines]).sum()

        predict_label_1.append(possible_error_type_1[ptr:ptr+len(g)][hit_lines].cpu().detach().numpy().tolist())
        predict_label_2.append(possible_error_type_2[ptr:ptr+len(g)][hit_lines].cpu().detach().numpy().tolist())
        true_label_1.append(label[error_lines_pos].cpu().detach().numpy().tolist())
        true_label_2.append(labels[1][ptr:ptr+len(g)][error_lines_pos].cpu().detach().numpy().tolist())

        # top1 & top 10
        top_lines_1 = torch.topk(g_score, k=min(1, len(g))).indices.cpu().detach().numpy().tolist()
        error_lines_pos_1 = torch.nonzero(label).view(-1).cpu().detach().numpy().tolist()
        hit_lines_1 = list(set(top_lines_1).intersection(set(error_lines_pos_1)))
        top1_count += len(hit_lines_1)

        top_lines_10 = torch.topk(g_score, k=min(10, len(g))).indices.cpu().detach().numpy().tolist()
        error_lines_pos_10 = torch.nonzero(label).view(-1).cpu().detach().numpy().tolist()
        hit_lines_10 = list(set(top_lines_10).intersection(set(error_lines_pos_10)))
        top10_count += len(hit_lines_10)

        ptr += len(g)

    return top5_count, error_count, type_1_count, type_2_count, top1_count, top10_count, \
           [predict_label_1, predict_label_2, true_label_1, true_label_2]