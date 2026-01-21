# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:30:16 2021

@author: Ling Sun
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from layer import HGATLayer
from torch_geometric.nn import GCNConv
import torch.nn.init as init
import Constants
from TransformerBlock import TransformerBlock,Long_term_atention
from torch.autograd import Variable
from graphConstruct import ConHypergraph
from dataLoader import Split_data_1
from Optim import *
from torch_geometric.nn import  GCNConv, HypergraphConv
from Block import TransformerBlock_1
from Time import TimeEncoding
from torch_geometric.nn import LayerNorm


def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
    masked_seq = Variable(masked_seq, requires_grad=False)
    # print("masked_seq ",masked_seq.size())
    return masked_seq.cuda()

# Fusion gate
class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.6, use_layer_norm=True, use_residual=True):
        super(Fusion, self).__init__()
        self.use_residual = use_residual
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)

        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(input_size)  # 修改为 input_size
            self.layer_norm2 = nn.LayerNorm(out)  # 这里 out 的维度也要匹配输出的维度

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, cas_emb, dy_emb):
        emb = torch.cat([cas_emb.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb = self.layer_norm1(emb) if hasattr(self, 'layer_norm1') else emb

        emb_score = F.softmax(self.linear2(F.leaky_relu(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)

        if self.use_residual:
            out += cas_emb

        return out


'''Learn friendship network'''

class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5, is_norm=True):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
        # in:inp,out:nip*2
        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)
        self.is_norm = is_norm

        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(ninp)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        #print('graph_edge_index', graph_edge_index)
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)
        # print(graph_output.shape)
        return graph_output.cuda()


'''Learn diffusion network'''


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3, is_norm=True):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.is_norm = is_norm
        if self.is_norm:
            self.batch_norm1 = torch.nn.BatchNorm1d(output_size)
        self.gat1 = HGATLayer(input_size, output_size, dropout=self.dropout, transfer=False, concat=True, edge=True)
        self.fus1 = Fusion(output_size)

    def forward(self, x, hypergraph_list):
        root_emb = F.embedding(hypergraph_list[1].cuda(), x)

        hypergraph_list = hypergraph_list[0]
        embedding_list = {}
        for sub_key in hypergraph_list.keys():
            sub_graph = hypergraph_list[sub_key]
            sub_node_embed, sub_edge_embed = self.gat1(x, sub_graph.cuda(), root_emb)
            sub_node_embed = F.dropout(sub_node_embed, self.dropout, training=self.training)

            if self.is_norm:
                sub_node_embed = self.batch_norm1(sub_node_embed)
                sub_edge_embed = self.batch_norm1(sub_edge_embed)

            x = self.fus1(x, sub_node_embed)
            embedding_list[sub_key] = [x.cpu(), sub_edge_embed.cpu()]

        return embedding_list

class fusion_1(nn.Module):
    def __init__(self, input_size, dropout=0.5):
        super(fusion_1, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, input_size, bias=False)
        self.linear2 = nn.Linear(input_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.input_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, hidden, dy_emb, dy_emb2):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0), dy_emb2.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
class LSTMGNN(nn.Module):
    def __init__(self, hypergraphs, args, dropout=0.2):
        super(LSTMGNN, self).__init__()

        # parameters
        self.emb_size = args.embSize
        self.n_node = args.n_node
        self.layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.drop_rate = dropout
        self.n_channel = len(hypergraphs)  ## Hypergraph 2, Social graph
        self.win_size = 3

        # hypergraph
        # self.H_Time = hypergraphs[0]   #source-user hypergraph
        self.H_Item = hypergraphs[0]  # cascade-item hypergraph
        self.H_User = hypergraphs[1]

        ###### user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)

        ### channel self-gating parameters
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        ### channel self-supervised parameters
        self.ssl_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.ssl_bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        ### attention parameters
        self.att = nn.Parameter(torch.zeros(1, self.emb_size))
        self.att_m = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))

        # sequence model
        self.past_gru = nn.GRU(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)
        self.past_lstm = nn.LSTM(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)

        # multi-head attention
        self.past_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, attn_dropout=dropout)

        self.future_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, is_FFN=False,
                                                 is_future=True, attn_dropout=dropout)

        self.long_term_att = Long_term_atention(input_size=self.emb_size, attn_dropout=dropout)


        self.reset_parameters()

        #### optimizer and loss function
        self.optimizerAdam = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.optimizer = ScheduledOptim(self.optimizerAdam, self.emb_size, args.n_warmup_steps)
        self.loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def self_supervised_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.ssl_weights[channel]) + self.ssl_bias[channel]))

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim=-1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

    def _dropout_graph(self, graph, keep_prob):
        size = graph.size()
        index = graph.coalesce().indices().t()
        values = graph.coalesce().values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    '''social structure and hypergraph structure embeddding'''

    def structure_embed(self):
        if self.training:
            # H_Time = self._dropout_graph(self.H_Time, keep_prob=0.6)
            H_Item = self._dropout_graph(self.H_Item, keep_prob=0.6)
            H_User = self._dropout_graph(self.H_User, keep_prob=0.6)
        else:
            # H_Time = self.H_Time
            H_Item = self.H_Item
            H_User = self.H_User

        # u_emb_c1 = self.self_gating(self.user_embedding.weight, 0)

        u_emb_c2 = self.self_gating(self.user_embedding.weight, 0)
        u_emb_c3 = self.self_gating(self.user_embedding.weight, 1)
        # u_emb_c2 = self.user_embedding.weight
        # u_emb_c3 = self.user_embedding.weight

        # all_emb_c1 = [u_emb_c1]
        all_emb_c2 = [u_emb_c2]
        all_emb_c3 = [u_emb_c3]

        for k in range(self.layers):
            # Channel Source
            # u_emb_c1 = torch.sparse.mm(H_Time, u_emb_c1)
            # norm_embeddings1 = F.normalize(u_emb_c1, p=2, dim=1)
            # all_emb_c1 += [norm_embeddings1]
            #
            # Channel Item
            u_emb_c2 = torch.sparse.mm(H_Item, u_emb_c2)
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [norm_embeddings2]

            u_emb_c3 = torch.sparse.mm(H_User, u_emb_c3)
            norm_embeddings2 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [norm_embeddings2]

        # u_emb_c1 = torch.stack(all_emb_c1, dim=1)
        # u_emb_c1 = torch.sum(u_emb_c1, dim=1)
        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.sum(u_emb_c2, dim=1)
        u_emb_c3 = torch.stack(all_emb_c3, dim=1)
        u_emb_c3 = torch.sum(u_emb_c3, dim=1)

        # aggregating channel-specific embeddings
        # high_embs, attention_score = self.channel_attention(u_emb_c1, u_emb_c2, u_emb_c3)
        high_embs, attention_score = self.channel_attention(u_emb_c2, u_emb_c3)
        # high_embs = u_emb_c3
        return high_embs
    def forward(self, input):

        mask = (input == 0)
        '''structure embeddding'''
        HG_Uemb = self.structure_embed().cuda()
        input = input.cuda()
        cas_seq_emb = F.embedding(input, HG_Uemb)
        source_emb = cas_seq_emb[:, 0, :]
        L_cas_emb = self.long_term_att(source_emb, cas_seq_emb, cas_seq_emb, mask=mask.cuda())
        output = self.past_multi_att(L_cas_emb, L_cas_emb, L_cas_emb, mask)
        #print('output',output.shape)
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))

        return L_cas_emb,cas_seq_emb

class GraphNN_2(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(GraphNN_2, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)

        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.norm1 = LayerNorm(ninp * 2)

        self.gnn2 = GCNConv(ninp * 2, ninp)
        self.norm2 = LayerNorm(ninp)
        self.gnn3 = HypergraphConv(ninp, ninp * 2)
        self.norm3 = LayerNorm(ninp * 2)
        self.gnn4 = HypergraphConv(ninp, ninp)
        self.norm4 = LayerNorm(ninp)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)
        for layer in [self.gnn1, self.gnn2, self.gnn3, self.gnn4]:
            if hasattr(layer, 'weight'):
                init.xavier_normal_(layer.weight)

    def forward(self, heter_graph, hyper_graph):
        heter_graph_edge_index = heter_graph.edge_index.cuda()
        heter_graph_x_embeddings = self.gnn1(self.embedding.weight, heter_graph_edge_index)
        heter_graph_x_embeddings = self.norm1(heter_graph_x_embeddings)
        heter_graph_output = self.gnn2(heter_graph_x_embeddings, heter_graph_edge_index)
        heter_graph_output = self.norm2(heter_graph_output)

        hyper_graph_edge_index = hyper_graph.edge_index.cuda()
        hyper_graph_output = self.gnn4(heter_graph_output.data.clone(), hyper_graph_edge_index)
        hyper_graph_output = self.norm4(hyper_graph_output)

        return heter_graph_output.cuda(), hyper_graph_output.cuda()


class Decoder(nn.Module):
    def __init__(self, input_size, user_size, opt):
        super(Decoder, self).__init__()
        if opt.norm:
            self.decoder = Decoder2L(input_size, user_size, opt.dropout)
        else:
            self.decoder = Decoder1L(input_size, user_size, opt.dropout)

    def forward(self, outputs):
        return self.decoder(outputs)


class Decoder2L(nn.Module):
    def __init__(self, input_size, user_size, dropout=0.1):
        super(Decoder2L, self).__init__()

        self.linear2 = nn.Linear(input_size, input_size * 2)
        self.linear1 = nn.Linear(input_size * 2, user_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(input_size * 2)

        # 使用Kaiming初始化
        init.kaiming_normal_(self.linear1.weight, nonlinearity='sigmoid')
        init.kaiming_normal_(self.linear2.weight, nonlinearity='sigmoid')

    def forward(self, outputs):
        x = self.linear2(outputs)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return x


class Decoder1L(nn.Module):
    def __init__(self, input_size, user_size, dropout=0.1):
        super(Decoder1L, self).__init__()

        self.linear1 = nn.Linear(input_size, user_size)
        self.dropout = nn.Dropout(p=dropout)

        # 使用Kaiming初始化
        init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')

    def forward(self, outputs):
        x = self.dropout(outputs)
        x = self.linear1(x)
        return x


class MSHGAT(nn.Module):
    def __init__(self, opt, dropout=0.6):
        super(MSHGAT, self).__init__()
        self.hidden_size = opt.d_word_vec
        self.n_node = opt.user_size
        self.pos_dim =8

        self.dropout = nn.Dropout(dropout)
        self.initial_feature = opt.initialFeatureSize
        self.data_name = opt.data_name
        self.hgnn = HGNN_ATT(self.initial_feature, self.hidden_size * 2, self.hidden_size, dropout=dropout)
        self.gnn = GraphNN(self.n_node, self.initial_feature, dropout=dropout)
        self.fus = Fusion(self.hidden_size + self.pos_dim)
        self.fus2 = Fusion(self.hidden_size)
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)
        self.decoder_attention1 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)
        self.decoder_attention2 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)
        self.linear2 = nn.Linear(self.hidden_size + self.pos_dim, self.n_node)
        self.linear3 = nn.Linear(self.hidden_size, self.n_node)
        self.embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)
        self.reset_parameters()
        self.opt = opt
        self.transformer_dim = self.hidden_size + self.pos_dim
        self.gnn_layer = GraphNN_2(self.n_node, self.initial_feature)
        # Instantiate LSTMGNN
        user_size, all_cascade, all_time = Split_data_1(self.data_name, load_dict=True)
        HG_Item, HG_User = ConHypergraph(self.data_name, user_size)
        HG_Item = trans_to_cuda(HG_Item)
        HG_User = trans_to_cuda(HG_User)
        self.lstmgnn = LSTMGNN(hypergraphs=[HG_Item, HG_User], args=self.opt, dropout=0.3)
        ninp = opt.d_word_vec
        # Learnable weights for fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.user_hidden_fusion = TransformerBlock_1(d_q=self.transformer_dim ,d_k=self.transformer_dim, d_v=self.transformer_dim,d_e=self.transformer_dim,n_heads=1)
        # Residual connection module
        self.residual = nn.Sequential(
            nn.Linear( self.hidden_size
                       +self.pos_dim,self.n_node),
            nn.ReLU(),
            nn.Linear( self.n_node,self.hidden_size+self.pos_dim)
        )
        self.non_linear_transform = nn.Sequential(
            nn.Linear(self.hidden_size + self.pos_dim, 256),  # 第一层线性层
            nn.ReLU6(),  # 激活函数，可以替换为其他激活函数如 LeakyReLU, ELU, GELU 等
            nn.Linear(256, self.n_node),  # 第二层线性层
            nn.ReLU6(),
            nn.Linear(self.n_node, 256),  # 第一层线性层
            nn.ReLU6(),  # 激活函数，可以替换为其他激活函数如 LeakyReLU, ELU, GELU 等
            nn.Linear(256, self.n_node)  # 第二层线性层
        )
        self.decoder = Decoder(input_size=self.transformer_dim, user_size=self.n_node, opt=opt)
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, input_timestamp, input_idx, graph, hypergraph_list):

        input = input[:, :-1]
        input_timestamp = input_timestamp[:, :-1]
        hidden = self.dropout(self.gnn(graph))
        user_social_embedding_lookup, user_hyper_embedding_lookup = self.gnn_layer(graph, graph)
        user_social_embedding_lookup = self.dropout(user_social_embedding_lookup)
        user_hyper_embedding_lookup = self.dropout(user_hyper_embedding_lookup)
        batch_size, max_len = input.size()
        user_input = input.contiguous().view(batch_size * max_len, 1).cuda()
        user_social_embedding_one_hot = torch.zeros(batch_size * max_len, self.n_node).cuda()
        user_social_embedding_one_hot = user_social_embedding_one_hot.scatter_(1, user_input, 1)
        user_social_embedding = torch.einsum("bt,td->bd", user_social_embedding_one_hot,
                                             user_social_embedding_lookup).view(batch_size, max_len,self.hidden_size).cuda()
        user_hyper_embedding = torch.einsum("bt,td->bd", user_social_embedding_one_hot,
                                            user_hyper_embedding_lookup).view(batch_size, max_len, self.hidden_size).cuda()
        memory_emb_list = self.hgnn(hidden, hypergraph_list)
        mask_1 = (input == Constants.PAD)
        time_encoder = TimeEncoding(8)
        encoded_time = time_encoder(input_timestamp).cuda()
        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))
        batch_size, max_len = input.size()
        zero_vec = torch.zeros_like(input)
        dyemb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()
        cas_emb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()
        for ind, time in enumerate(sorted(memory_emb_list.keys())):
            if ind == 0:
                sub_input = torch.where(input_timestamp <= time, input, zero_vec)
                sub_emb = F.embedding(sub_input.cuda(), hidden.cuda())
                temp = sub_input == 0
                sub_cas = sub_emb.clone()
            else:
                cur = torch.where(input_timestamp <= time, input, zero_vec) - sub_input
                temp = cur == 0
                sub_cas = torch.zeros_like(cur)
                sub_cas[~temp] = 1
                sub_cas = torch.einsum('ij,i->ij', sub_cas, input_idx)
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())
                sub_emb = F.embedding(cur.cuda(), list(memory_emb_list.values())[ind - 1][0].cuda())
                sub_input = cur + sub_input
            sub_cas[temp] = 0
            sub_emb[temp] = 0
            dyemb += sub_emb
            cas_emb += sub_cas
            if ind == len(memory_emb_list) - 1:
                sub_input = input - sub_input
                temp = sub_input == 0
                sub_cas = torch.zeros_like(sub_input)
                sub_cas[~temp] = 1
                sub_cas = torch.einsum('ij,i->ij', sub_cas, input_idx)
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())
                sub_cas[temp] = 0
                sub_emb = F.embedding(sub_input.cuda(), list(memory_emb_list.values())[ind][0].cuda())
                sub_emb[temp] = 0
                dyemb += sub_emb
                cas_emb += sub_cas
        dyemb = self.fus2(dyemb, cas_emb)
        diff_embed = torch.cat([dyemb, order_embed], dim=-1).cuda()
        fri_embed = torch.cat([F.embedding(input.cuda(), hidden.cuda()), order_embed], dim=-1).cuda()
        diff_att_out = self.decoder_attention1(diff_embed.cuda(), diff_embed.cuda(), diff_embed.cuda(), mask=mask_1.cuda())
        diff_att_out = self.dropout(diff_att_out.cuda())
        fri_att_out = self.decoder_attention2(fri_embed.cuda(), fri_embed.cuda(), fri_embed.cuda(), mask=mask_1.cuda())
        fri_att_out = self.dropout(fri_att_out.cuda())
        att_out = self.fus(diff_att_out, fri_att_out)
        output_u = self.linear2(att_out.cuda())  # (bsz, user_len, |U|)
        L_cas_emb,outp = self.lstmgnn(input)
        L_cas_emb=torch.cat([L_cas_emb,order_embed],dim=-1)
        L_cas_emb_11= self.decoder_attention1(L_cas_emb.cuda(), L_cas_emb.cuda(), L_cas_emb.cuda(), mask=mask_1.cuda())
        L_cas_emb_1 = self.linear2(L_cas_emb)
        cas= torch.cat([outp, order_embed], dim=-1)
        cas_1 = self.decoder_attention1(cas.cuda(), cas.cuda(), cas.cuda(), mask=mask_1.cuda())
        user_social=torch.cat([user_social_embedding,encoded_time],dim=-1)
        user_social_1 = self.decoder_attention1(user_social.cuda(), user_social.cuda(), user_social.cuda(),
                                              mask=mask_1.cuda())
        user_hyper=torch.cat([user_hyper_embedding,encoded_time],dim=-1)
        user_hyper_1 = self.decoder_attention1(user_hyper.cuda(), user_hyper.cuda(), user_hyper.cuda(),
                                              mask=mask_1.cuda())
        #user_hidden = (self.user_hidden_fusion(user_social_1 , user_hyper_1, L_cas_emb_11,att_out, mask=mask_1))
        user_hidden=user_social_1+user_hyper_1+L_cas_emb_11+att_out
        user_hidden_1=user_hidden+0.33*cas_1
        final=self.decoder(user_hidden_1)
        mask_1 = get_previous_user_mask(input.cpu(), self.n_node)
        return (final+mask_1).view(-1, output_u.size(-1)).cuda()