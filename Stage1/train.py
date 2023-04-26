from utils import *
from dataset import GraphDataset
from model import GraphMatchingNet
import numpy as np
import torch.nn as nn
import collections
from tqdm import tqdm
import time
import os
import random
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs')

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
# device = 'cpu'

# NOTE: Similarity Task

# Set random seeds
seed = 3407
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Build Datasets
epochs = 5
batch_size = 32

train_pair_path = "./data/train_similar_pairs.npy"
valid_pair_path = "./data/valid_similar_pairs.npy"
test_pair_path = "./data/test_similar_pairs.npy"
graph_dict = "./data/graph_data.npy"

training_set = GraphDataset(train_pair_path, graph_dict, device)
validation_set = GraphDataset(valid_pair_path, graph_dict, device)
test_set = GraphDataset(test_pair_path, graph_dict, device)

training_data_iter = training_set.pairs(batch_size)
validation_data_iter = validation_set.pairs(batch_size)
test_data_iter = test_set.pairs(batch_size)
print("Finished Dataset Preparation.")

# Build Model
node_type_num = 1193+1
node_type_dim = 32
node_content_dim = 100
node_state_dim = 128
graph_state_dim = 128
similarity_name = 'cosine'
learning_rate = 5e-4
weight_decay = 1e-5
graph_vec_regularizer_weight = 1e-6
clip_value = 1e-6

model = GraphMatchingNet(node_type_num, node_type_dim, node_content_dim, node_state_dim,
                         graph_state_dim, similarity_name, device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

print("Finished Model Preparation.")

global_count = 0
t_start = time.time()
for epoch in range(epochs):
    train_loss = []
    train_sim_diff = []
    train_auc = []
    model.train()
    for i_iter in tqdm(range(0, training_set.num, batch_size)):
        batch_g1, batch_g2, labels, node_line_length = next(training_data_iter)
        labels = labels.to(device)

        x, y, _ = model(batch_g1, batch_g2, True)
        loss = pairwise_loss(x, y, labels)

        is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
        is_neg = 1 - is_pos
        n_pos = torch.sum(is_pos)
        n_neg = torch.sum(is_neg)
        sim = compute_similarity(x, y)
        sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8) 
        sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8)  


        graph_vectors = torch.cat([x, y], dim=0)
        graph_vec_scale = torch.mean(graph_vectors ** 2)
        loss += graph_vec_regularizer_weight * 0.5 * graph_vec_scale

        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))  #
        nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()

        sim_diff = sim_pos - sim_neg
        batch_auc = auc(sim, labels)

        train_auc.append(batch_auc)
        train_loss.append(loss.cpu().detach().numpy().mean())
        train_sim_diff.append(sim_diff.cpu().detach().numpy())


        global_count += 1
        t_start = time.time()

    writer.add_scalar('Loss/Train', np.mean(train_loss), epoch)
    writer.add_scalar('Similarity Diff/Train', np.mean(train_sim_diff), epoch)
    writer.add_scalar('AUC Score/Train', np.mean(train_auc), epoch)


    valid_loss = []
    valid_sim_diff = []
    valid_auc = []
    model.eval()
    with torch.no_grad():
        for i_iter in tqdm(range(0, validation_set.num, batch_size)):
            batch_g1, batch_g2, labels, node_line_length = next(validation_data_iter)
            labels = labels.to(device)
            x, y, _ = model(batch_g1, batch_g2, True)

            loss = pairwise_loss(x, y, labels)


            is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
            is_neg = 1 - is_pos
            n_pos = torch.sum(is_pos)
            n_neg = torch.sum(is_neg)
            sim = compute_similarity(x, y)
            sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)  
            sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8) 
            sim_diff = sim_pos - sim_neg


            graph_vectors = torch.cat([x, y], dim=0)
            graph_vec_scale = torch.mean(graph_vectors ** 2)
            loss += graph_vec_regularizer_weight * 0.5 * graph_vec_scale

            valid_loss.append(loss.cpu().detach().numpy().mean())
            valid_sim_diff.append(sim_diff.cpu().detach().numpy())

            batch_auc = auc(sim, labels)
            valid_auc.append(batch_auc)

    writer.add_scalar('Loss/Validation', np.mean(valid_loss), epoch)
    writer.add_scalar('Similarity Diff/Validation', np.mean(valid_sim_diff), epoch)
    writer.add_scalar('AUC Score/Validation', np.mean(valid_auc), epoch)

    print('[Epoch: %3d/%3d] Training Loss: %.4f, Training similarity diff: %.4f,'
          ' Validation Loss: %.4f, Validation AUC: %.3f, Validation similarity diff: %.4f'
          % (epoch, epochs, np.mean(train_loss), np.mean(train_sim_diff), np.mean(valid_loss),
             np.mean(valid_auc),np.mean(valid_sim_diff)))

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict()
                }, f"./model/pretrained_nbl/epoch_{epoch}.pkl")


test_loss = []
test_sim_diff = []
test_auc = []
model.eval()
with torch.no_grad():
    for i_iter in tqdm(range(0, test_set.num, batch_size)):
        batch_g1, batch_g2, labels, node_line_length = next(test_data_iter)
        labels = labels.to(device)
        x, y, _ = model(batch_g1, batch_g2, True)

        loss = pairwise_loss(x, y, labels)


        is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
        is_neg = 1 - is_pos
        n_pos = torch.sum(is_pos)
        n_neg = torch.sum(is_neg)
        sim = compute_similarity(x, y)
        sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)  
        sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8) 
        sim_diff = sim_pos - sim_neg


        graph_vectors = torch.cat([x, y], dim=0)
        graph_vec_scale = torch.mean(graph_vectors ** 2)
        loss += graph_vec_regularizer_weight * 0.5 * graph_vec_scale

        test_loss.append(loss.cpu().detach().numpy().mean())
        test_sim_diff.append(sim_diff.cpu().detach().numpy())

        batch_auc = auc(sim, labels)
        test_auc.append(batch_auc)

print('Test Loss: %.4f, Test AUC: %.3f, Test similarity diff: %.4f')