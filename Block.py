import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class TransformerBlock_1(nn.Module):

    def __init__(self, d_q=64, d_k=64, d_v=64, d_e=64,n_heads=8, is_layer_norm=True, attn_dropout=0.1):
        super(TransformerBlock_1, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_q
        self.d_v = d_v if d_v is not None else d_q

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=d_v)

        self.W_q = nn.Parameter(torch.Tensor(d_q, n_heads * d_v))
        self.W_k = nn.Parameter(torch.Tensor(d_k, n_heads * d_v))
        self.W_v = nn.Parameter(torch.Tensor(d_v, n_heads * d_v))
        self.W_e = nn.Parameter(torch.Tensor(d_v, n_heads * d_v))  # New parameter for E

        self.W_o = nn.Parameter(torch.Tensor(d_v * n_heads, d_v))
        self.linear1 = nn.Linear(d_v, d_v)
        self.linear2 = nn.Linear(d_v, d_v)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_e)  # Initialize W_e
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, E, mask, episilon=1e-6):
        temperature = self.d_k ** 0.5

        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().cuda()
            mask_ = mask.cuda() + pad_mask.cuda()
            Q_K = Q_K.masked_fill(mask_, -1e2)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)

        # Incorporate E into the attention output
        E_att = Q_K_score.bmm(E)  # (*, max_q_words, input_size)
        combined_att = V_att + E_att  # Combine V and E attention

        return combined_att

    def multi_head_attention(self, Q, K, V, E, mask):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()
        bsz, e_len, _ = E.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_v)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_v)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)
        E_ = E.matmul(self.W_e).view(bsz, e_len, self.n_heads, self.d_v)  # Transform E

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, k_len, self.d_v)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, v_len, self.d_v)
        E_ = E_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, e_len, self.d_v)  # Reshape E

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # For head axis broadcasting.
            mask = mask.reshape(-1, mask.size(-1))

        combined_att = self.scaled_dot_product_attention(Q_, K_, V_, E_, mask)
        combined_att = combined_att.view(bsz, self.n_heads, q_len, self.d_v)
        combined_att = combined_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)

        output = self.dropout(combined_att.matmul(self.W_o))  # (batch_size, max_q_words, input_size)
        return output

    def forward(self, Q, K, V, E, S=0, mask=None):
        V_att = self.multi_head_attention(Q, K, V, E, mask)

        if self.is_layer_norm:
            X = self.layer_norm(V + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_norm(self.FFN(X) + X)
        else:
            X = V + V_att
            output = self.FFN(X) + X
        return output
