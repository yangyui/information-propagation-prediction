# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:42:32 2021

@author: Ling Sun
"""

import argparse
import time
import numpy as np
import Constants
import torch
import torch.nn as nn
from graphConstruct import ConRelationGraph, ConHyperGraphList
from dataLoader import Split_data, DataLoader
from Metrics import Metrics
from HGAT import MSHGAT
from Optim import ScheduledOptim
from dataLoader import Split_data_1


torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

metric = Metrics()

parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='memetracker')
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-d_model', type=int, default=64)
parser.add_argument('-initialFeatureSize', type=int, default=64)
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.6)
parser.add_argument('-log', default=None)
parser.add_argument('-save_path', default= "./checkpoint/DiffusionPrediction.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-pos_emb', type=bool, default=True)
parser.add_argument('-preprocess', type=bool, default=False, help="preprocess dataset")    #when you first run code, you should set it to true.
parser.add_argument('-max_lenth', type=int, default=100)
parser.add_argument('-posSize', type=int, default=8, help= "the position embedding size")
parser.add_argument('--embSize', type=int, default=64, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default= 0.005, help='ssl task maginitude')  ###0.005  0.001  0.01 0.02  0.05 0.1 0.5
parser.add_argument('--beta2', type=float, default=0.005, help='ssl task maginitude')  ###0.005  0.001  0.01 0.02  0.05 0.1 0.5
parser.add_argument('-patience', type=int, default=5, help="control the step of early-stopping")
parser.add_argument('--norm', type=bool, default=True, metavar='Norm',
                    help='Need norm')
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)
parser.add_argument('-d_inner_hid', type=int, default=64)
parser.add_argument('-window_size', type=int, default=3)
parser.add_argument('-n_head', type=int, default=8)
opt = parser.parse_args()
opt.d_word_vec = opt.d_model
user_size, all_cascade, all_time = Split_data_1(opt.data_name,load_dict=True)
print('user_size:',user_size)
opt.n_node=user_size
# print(opt)


def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


def train_epoch(model, training_data, graph, hypergraph_list, loss_func, optimizer):
    # train

    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    batch_num = 0.0

    for i, batch in enumerate(
            training_data):  # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # data preparing
        tgt, tgt_timestamp, tgt_idx = (item.cuda() for item in batch)

        np.set_printoptions(threshold=np.inf)
        gold = tgt[:, 1:]

        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        # training
        optimizer.zero_grad()
        pred = model(tgt, tgt_timestamp, tgt_idx, graph, hypergraph_list)
        loss, n_correct = get_performance(loss_func, pred, gold)
        loss.backward()

        # parameter update
        optimizer.step()
        optimizer.update_learning_rate()

        n_total_correct += n_correct
        total_loss += loss.item()

    return total_loss / n_total_words, n_total_correct / n_total_words


def train_model(MSHGAT, data_path):
    # ========= Preparing DataLoader =========#
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    train_data = DataLoader(train, batch_size=opt.batch_size, load_dict=True, cuda=False)
    valid_data = DataLoader(valid, batch_size=opt.batch_size, load_dict=True, cuda=False)
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    opt.user_size = user_size

    # ========= Preparing Model =========#
    model = MSHGAT(opt, dropout=opt.dropout)

    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)

    params = model.parameters()
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-05)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    validation_history = 0.0
    best_scores = {}
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, train_data, relation_graph, hypergraph_list, loss_func, optimizer)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        if epoch_i >= 0:
            start = time.time()
            scores = test_epoch(model, valid_data, relation_graph, hypergraph_list)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print("Validation use time: ", (time.time() - start) / 60, "min")

            print('  - (Test) ')
            scores = test_epoch(model, test_data, relation_graph, hypergraph_list)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))

            if validation_history <= sum(scores.values()):
                print("Best Validation hit@100:{} at Epoch:{}".format(scores["hits@100"], epoch_i))
                validation_history = sum(scores.values())
                best_scores = scores
                print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)

    print(" -(Finished!!) \n Best scores: ")
    for metric in best_scores.keys():
        print(metric + ' ' + str(best_scores[metric]))


def test_epoch(model, validation_data, graph, hypergraph_list, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            # Prepare data
            tgt, tgt_timestamp, tgt_idx = batch
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

            # Forward
            pred = model(tgt, tgt_timestamp, tgt_idx, graph, hypergraph_list)
            y_pred = pred.detach().cpu().numpy()

            # Compute metrics
            scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len

            # Accumulate scores
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)]
                scores['map@' + str(k)] += scores_batch['map@' + str(k)]

        # Average the scores
        for k in k_list:
            scores['hits@' + str(k)] /= len(validation_data)
            scores['map@' + str(k)] /= len(validation_data)
    return scores


def test_model(MSHGAT, data_path):
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(train, user_size)

    opt.user_size = user_size

    model = MSHGAT(opt, dropout=opt.dropout)
    model.load_state_dict(torch.load(opt.save_path))
    model.cuda()

    scores = test_epoch(model, test_data, relation_graph, hypergraph_list)
    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))


if __name__ == "__main__":
    model = MSHGAT
    train_model(model, opt.data_name)



