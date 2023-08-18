import math
import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_units, n_layers, n_heads, dropout, device,
                 final_act='relu', eps=1e-8, bidirectional=False):
        super(TransformerEncoder, self).__init__()
        
        self.device = device
        self.bidirectional = bidirectional

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=eps)

        for _ in range(n_layers):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=eps)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(hidden_units, n_heads, dropout, batch_first=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=eps)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout, final_act)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs, input_emb):
        # log2feats
        timeline_mask = (seqs == 0)
        input_emb *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        if self.bidirectional:
            # For BERT4Rec
            attention_mask = ~(seqs > 0).unsqueeze(1).repeat(1, seqs.size(1), 1)
        else:
            # For SASRec
            tl = input_emb.shape[1]
            attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](input_emb)
            mha_outputs, attn_weights = self.attention_layers[i](
                Q, input_emb, input_emb, attn_mask=attention_mask
            )

            input_emb = Q + mha_outputs

            input_emb = self.forward_layernorms[i](input_emb)
            input_emb = self.forward_layers[i](input_emb)
            input_emb *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(input_emb) # (U, T, C) -> (U, -1, C)

        return log_feats


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout, final_act='relu'):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units * 4, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout)
        self.create_final_activation(final_act)
        self.conv2 = nn.Conv1d(hidden_units * 4, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))
        elif final_act == 'gelu':
            self.final_activation = GeLU()
        elif final_act == 'swish':
            self.final_activation = Swish()


    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.final_activation(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class GeLU(nn.Module):
    """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)



class GNN(nn.Module):
    """Gated GNN"""
    def __init__(self, embedding_dim, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = embedding_dim
        self.input_size = embedding_dim * 2
        self.gate_size = 3 * embedding_dim
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden
