import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_, normal_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.layers import GNN
from models.layers import TransformerEncoder
from utils.utils import get_last_item


class SRGNN(nn.Module):
    def __init__(self, input_size, args, device):
        super(SRGNN, self).__init__()
        self.n_items = input_size
        self.args = args
        self.device = device

        # Embedding
        self.item_embedding = nn.Embedding(self.n_items, args.embedding_dim, padding_idx=0)

        # Model Architecture
        self.gnn = GNN(args.embedding_dim, step=args.n_layers)
        self.linear1 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=True)
        self.linear2 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=True)
        self.linear3 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=True)
        self.linear_transform = nn.Linear(args.embedding_dim * 2, args.embedding_dim, bias=True)

        self._init_weights()
    
    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        seqs = batch[input_str]
        lengths_t = torch.as_tensor(batch[len_str]).to(self.device)
        alias_inputs, A, items, mask = self._get_slice(seqs)
        hidden = self.item_embedding(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.args.embedding_dim
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = get_last_item(seq_hidden, lengths_t)
        q1 = self.linear1(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear2(seq_hidden)

        alp = self.linear3(torch.sigmoid(q1 + q2))
        a = torch.sum(alp * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        
        return seq_hidden, seq_output

    def _get_slice(self, seqs):
        mask = seqs.gt(0)
        items, A, alias_inputs = [], [], []
        max_n_nodes = seqs.size(1)
        seqs = seqs.cpu().numpy()
        for seq in seqs:
            node = np.unique(seq)
            items.append(node.tolist() + (max_n_nodes - len(node)) * [0])
            u_A = np.zeros((max_n_nodes, max_n_nodes))
            for i in np.arange(len(seq) - 1):
                if seq[i+1] == 0:
                    break
                u = np.where(node == seq[i])[0][0]
                v = np.where(node == seq[i+1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in seq])
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(np.array(A)).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask


class NARM(nn.Module):
    """Neural Attentive Session Based Recommendation Model Class

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        embedding_dim(int): the dimension of item embedding
        batch_size(int): 
        n_layers(int): the number of gru layers

    """
    def __init__(self, input_size, args, device):
        super(NARM, self).__init__()
        self.n_items = input_size
        self.args = args
        self.device = device

        self.item_embedding = nn.Embedding(input_size, args.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(args.dropout0)
        self.gru = nn.GRU(args.embedding_dim, args.hidden_size, args.n_layers, batch_first=True)
        self.a_1 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.a_2 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.v_t = nn.Linear(args.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(args.dropout1)
        self.b = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        seqs = batch[input_str]
        hidden = self.init_hidden(seqs.size(0))
        item_embedding = self.emb_dropout(self.item_embedding(seqs))
        item_embedding = pack_padded_sequence(item_embedding, batch[len_str], batch_first=True)
        self.gru.flatten_parameters()
        gru_out, hidden = self.gru(item_embedding, hidden)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=self.args.maxlen)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        # gru_out = gru_out.permute(1, 0, 2) # (Session Length, Batch, Hidden) -> (B, S, H)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.args.hidden_size)).view(gru_out.size())  
        q2 = self.a_2(ht)

        mask = torch.where(seqs > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alp = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.args.hidden_size)).view(mask.size())
        c_local = torch.sum(alp.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        predictions = self.b(c_t)
        
        return gru_out, predictions

    def init_hidden(self, batch_size):
        return torch.zeros((self.args.n_layers, batch_size, self.args.hidden_size),
                           requires_grad=True, device=self.device)
    

class STAMP(nn.Module):
    def __init__(self, input_size, args, device):
        super(STAMP, self).__init__()
        self.n_items = input_size
        self.args = args
        self.device = device

        self.item_embedding = nn.Embedding(input_size, args.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(args.dropout0)
        self.w1 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=False)
        self.w2 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=False)
        self.w3 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=False)
        self.w0 = nn.Linear(args.embedding_dim, 1, bias=False)
        self.b_a = normal_(nn.Parameter(torch.zeros(args.embedding_dim), requires_grad=True), 0, 0.05) # Based on original code
        self.mlp_a = nn.Linear(args.embedding_dim, args.embedding_dim, bias=False)
        self.mlp_b = nn.Linear(args.embedding_dim, args.embedding_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.zi_dropout = nn.Dropout(args.dropout1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
     
    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        seqs = batch[input_str]
        lengths_t = torch.as_tensor(batch[len_str]).to(self.device)
        item_embedding = self.item_embedding(seqs) # item sequence embedding
        item_embedding = self.emb_dropout(item_embedding)
        mask = seqs.ne(0).to(self.device)
        xt = get_last_item(item_embedding, lengths_t) # last_item
        xi = item_embedding

        # Attention Net
        ms = torch.div(torch.sum(xi, dim=1), lengths_t.unsqueeze(1).float())
        alp = self.get_alpha(xi, xt, ms, mask)
        ma = torch.matmul(alp.unsqueeze(1), item_embedding).squeeze(1) # (B, 1, S) * (B, S, H) = (B, 1, H).squeeze(1) -> (B, H)
        ma += ms # based on original code

        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(xt))

        # Trilinear Composition
        # z_i = torch.matmul((ht * xt), hs.transpose(1, 0))
        z_i = hs * ht # based on original code
        predictions = self.zi_dropout(z_i)

        return xi, predictions

    def get_alpha(self, memory, last_items, general_interest, mask):
        expand_last = last_items.unsqueeze(1).expand_as(memory)
        expand_gi = general_interest.unsqueeze(1).expand_as(memory)

        res_memory = self.w1(memory)
        res_last = self.w2(expand_last)
        res_gi = self.w3(expand_gi)

        res_sum = res_memory + res_last + res_gi
        res_act = self.sigmoid(res_sum)

        b_a = self.b_a.repeat(memory.shape[0], 1).view(-1, self.b_a.shape[0], 1)
        res_act = torch.matmul(res_act, b_a).squeeze(2)
        alp = res_act * mask

        return alp

class GCSAN(nn.Module):
    def __init__(self, input_size, args, device):
        super(GCSAN, self).__init__()
        self.n_items = input_size
        self.args = args
        self.device = device

        # Embedding
        self.item_embedding = nn.Embedding(self.n_items, args.embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(args.maxlen, args.hidden_size)

        # Model Architecture
        self.gnn = GNN(args.embedding_dim, step=args.n_layers)
        self.transformer_enc = TransformerEncoder(args.hidden_size,
                                                  args.n_layers,
                                                  args.n_heads,
                                                  args.dropout1,
                                                  self.device,
                                                  bidirectional=False)
        self.linear1 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=True)
        self.linear2 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=True)
        self.linear3 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=True)
        self.linear_transform = nn.Linear(args.embedding_dim * 2, args.embedding_dim, bias=True)

        self.weight = self.args.lamb

        self._init_weights()
    
    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):
        seqs = batch[input_str]
        lengths_t = torch.as_tensor(batch[len_str]).to(self.device)
        alias_inputs, A, items, mask = self._get_slice(seqs)
        hidden = self.item_embedding(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.args.embedding_dim
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = get_last_item(seq_hidden, lengths_t)
        # q1 = self.linear1(ht).view(ht.size(0), 1, ht.size(1))
        # q2 = self.linear2(seq_hidden)

        # alp = self.linear3(torch.sigmoid(q1 + q2))
        # a = torch.sum(alp * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        # seq_output = self.linear_transform(torch.cat([a, ht], dim=1))

        pos = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        position_embedding = self.position_embedding(torch.LongTensor(pos).to(self.device))

        # Input Embedding
        input_emb = seq_hidden + position_embedding

        # Transformer Architecture
        log_feats = self.transformer_enc(seqs, input_emb)

        # predictions = log_feats[:, -1, :]
        output = get_last_item(log_feats, batch['lens'])

        seq_output = self.weight * output + (1 - self.weight) * ht

        return seq_output

    def _get_slice(self, seqs):
        mask = seqs.gt(0)
        items, A, alias_inputs = [], [], []
        max_n_nodes = seqs.size(1)
        seqs = seqs.cpu().numpy()
        for seq in seqs:
            node = np.unique(seq)
            items.append(node.tolist() + (max_n_nodes - len(node)) * [0])
            u_A = np.zeros((max_n_nodes, max_n_nodes))
            for i in np.arange(len(seq) - 1):
                if seq[i+1] == 0:
                    break
                u = np.where(node == seq[i])[0][0]
                v = np.where(node == seq[i+1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in seq])
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        A = torch.FloatTensor(np.array(A)).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask
