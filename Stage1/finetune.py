from utils import *
from dataset import GraphDataset
from model import ErrorPredictionModel
import numpy as np
from sklearn import metrics
import torch.nn as nn
import collections
from tqdm import tqdm
import time
import os
import random
import torch
import warnings
warnings.filterwarnings("ignore")

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


seed = 3407
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Build Datasets
epochs = 100
batch_size = 16
n_tasks = 3
train_pair_path = "./data/train_error_pairs.npy" 
valid_pair_path = "./data/valid_error_pairs.npy"
test_pair_path = "./data/test_error_pairs.npy"
graph_dict = "./data/graph_data.npy"
pretrained_model_path = "./model/pretrained/epoch_best.pkl"


training_set = GraphDataset(train_pair_path, graph_dict, device, 'finetune', epochs)
validation_set = GraphDataset(valid_pair_path, graph_dict, device, 'finetune', epochs)
test_set = GraphDataset(test_pair_path, graph_dict, device, 'finetune', epochs)

training_data_iter = training_set.pairs(batch_size)
validation_data_iter = validation_set.pairs(batch_size)
test_data_iter = test_set.pairs(batch_size)
print("Finished Dataset Preparation.")

# Build Model
node_type_num = 1193+1 # tobe modify 1193  1166
node_type_dim = 32
node_content_dim = 100
node_state_dim = 128
graph_state_dim = 128
similarity_name = 'cosine'
learning_rate = 1e-3
weight_decay = 1e-5
graph_vec_regularizer_weight = 1e-6
clip_value = 1e-6

error_type_num_1 = 4+1
error_type_num_2 = 16+1

model = ErrorPredictionModel(node_type_num, node_type_dim, node_content_dim, node_state_dim,
                         graph_state_dim, similarity_name, device, error_type_num_1, error_type_num_2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

if pretrained_model_path is not None:
    checkpoint = torch.load(pretrained_model_path)
    model.node_state_model.load_state_dict(checkpoint['model_state_dict'])

print("Finished Model Preparation.")

t_start = time.time()
flag = True
for epoch in range(epochs):
    train_loss = []
    train_Top1_count = 0
    train_Top10_count = 0
    train_Top5_count = 0
    train_error_count = 0
    train_type_1_count = 0
    train_type_2_count = 0

    model.train()
    for i_iter in tqdm(range(0, training_set.num, batch_size)):
        batch_g1, batch_g2, labels, node_line_length = next(training_data_iter)

        output_1, output_2 = model(batch_g1, batch_g2, node_line_length)

        task_loss = torch.stack([localization_loss(output_1, labels[0]),
                            classification_loss(output_1, labels[0]),
                            classification_loss(output_2, labels[1])])

        weighted_task_loss = torch.mul(model.weights, task_loss)
        if flag:
            initial_task_loss = task_loss.data.cpu().numpy()
            flag = False

        loss = torch.sum(weighted_task_loss)

        hit_count, error_count, type_1_count, type_2_count, top1_count, top10_count, _\
            = calculate_metrics(output_1, output_2, labels, 5, batch_g1)
        #
        train_Top5_count += hit_count
        train_error_count += error_count
        train_type_1_count += type_1_count
        train_type_2_count += type_2_count
        train_Top1_count += top1_count
        train_Top10_count += top10_count


        optimizer.zero_grad()
        loss.backward(retain_graph=True)  #

        train_loss.append(loss.cpu().detach().numpy())
        # https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py
        model.weights.grad.data = model.weights.grad.data * 0.0
        W = model.get_last_shared_layer()
        norms = []
        for i in range(3):
            gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
            norms.append(torch.norm(torch.mul(model.weights[i], gygw[0])))
        norms = torch.stack(norms)
        loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)
        mean_norm = np.mean(norms.data.cpu().numpy())
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 0.1), requires_grad=False).cuda()
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
        model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]
        #
        normalize_coeff = n_tasks / torch.sum(model.weights.data, dim=0)
        model.weights.data = model.weights.data * normalize_coeff

        optimizer.step()

        t_start = time.time()


    val_loss = []
    valid_Top1_count = 0
    valid_Top10_count = 0
    valid_Top5_count = 0
    valid_error_count = 0
    valid_type_1_count = 0
    valid_type_2_count = 0
    model.eval()
    with torch.no_grad():
        for i_iter in tqdm(range(0, validation_set.num, batch_size)):
            batch_g1, batch_g2, labels, node_line_length = next(validation_data_iter)
            output_1, output_2 = model(batch_g1, batch_g2, node_line_length)

            task_loss = torch.stack([localization_loss(output_1, labels[0]),
                                     classification_loss(output_1, labels[0]),
                                     classification_loss(output_2, labels[1])])

            loss = torch.sum(torch.mul(model.weights, task_loss))


            hit_count, error_count, type_1_count, type_2_count, top1_count, top10_count,_ \
                = calculate_metrics(output_1, output_2, labels, 5, batch_g1)
            
            valid_Top5_count += hit_count
            valid_error_count += error_count
            valid_type_1_count += type_1_count
            valid_type_2_count += type_2_count
            valid_Top1_count += top1_count
            valid_Top10_count += top10_count

            val_loss.append(loss.cpu().detach().numpy())

    print('[Epoch: %3d/%3d] Training Loss: %.4f, Top-1 Localization: %.4f, Top-5 Localization: %.4f, '
          'Top-10 Localization: %.4f, ACC level-1: %.4f, ACC level-2: %.4f,\n'
          ' Validation Loss: %.4f,  Top-1 Localization: %.4f, Top-5 Localization: %.4f, '
          'Top-10 Localization: %.4f, ACC level-1: %.4f, ACC level-2: %.4f'
          % (epoch, epochs, np.mean(train_loss), train_Top1_count / train_error_count,
             train_Top5_count / train_error_count,train_Top10_count / train_error_count,
             train_type_1_count / train_error_count, train_type_2_count / train_error_count,
             np.mean(val_loss), valid_Top1_count / valid_error_count, valid_Top5_count / valid_error_count,
             valid_Top10_count / valid_error_count,
             valid_type_1_count / valid_error_count, valid_type_2_count / valid_error_count))

    
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict()
                }, f"./model/finetune_new/epoch_{epoch}.pkl")


test_loss = []
test_Top1_count = 0
test_Top10_count = 0
test_Top5_count = 0
test_error_count = 0
test_type_1_count = 0
test_type_2_count = 0
model.eval()
with torch.no_grad():
    for i_iter in tqdm(range(0,test_set.num, batch_size)):
        batch_g1, batch_g2, labels, node_line_length = next(test_data_iter)
        output_1, output_2 = model(batch_g1, batch_g2, node_line_length)

        task_loss = torch.stack([localization_loss(output_1, labels[0]),
                                 classification_loss(output_1, labels[0]),
                                 classification_loss(output_2, labels[1])])

        loss = torch.sum(torch.mul(model.weights, task_loss))


        hit_count, error_count, type_1_count, type_2_count, top1_count, top10_count,_ \
            = calculate_metrics(output_1, output_2, labels, 5, batch_g1)
        
        test_Top5_count += hit_count
        test_error_count += error_count
        test_type_1_count += type_1_count
        test_type_2_count += type_2_count
        test_Top1_count += top1_count
        test_Top10_count += top10_count

        test_loss.append(loss.cpu().detach().numpy())

print('Test Top-1 Localization: %.4f, Top-5 Localization: %.4f, '
      'Top-10 Localization: %.4f, ACC level-1: %.4f, ACC level-2: %.4f'
      % (test_Top1_count / test_error_count, test_Top5_count / test_error_count,
         test_Top10_count / test_error_count,
         test_type_1_count / test_error_count , test_type_2_count / test_error_count))
