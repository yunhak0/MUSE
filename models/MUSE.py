import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from embedder import Embedder
from models.backbone import NARM, STAMP, SRGNN
from models.metric import evaluate


class MUSE_Trainer(Embedder):
    def __init__(self, args):
        Embedder.__init__(self, args)

    def load_model(self):
        self.model = VICReg(self.n_items, self.args, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        batch_losses = []
        shuffle_rec_losses = []
        nonshuffle_rec_losses = []
        epoch_loss = 0
        self.train_before_epoch_start()
        train_batch_iter = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, batch in train_batch_iter:
            batch = self.after_epoch_start(batch)
            batch['aug1'] = batch['aug1'].to(self.device, non_blocking=True)

            v1_hidden, v1_preds = self.model(batch,
                                             input_str='orig_sess',
                                             len_str='lens',
                                             get_last=True)
            v2_hidden, v2_preds = self.model(batch,
                                             input_str='aug1',
                                             len_str='aug_len1',
                                             get_last=True)
            
            matching_loss = self.model.compute_finegrained_matching_loss(
                batch, v1_hidden, v2_hidden, v1_preds, v2_preds, epoch
            )

            rec_loss = self.calculate_loss(v1_preds, batch)
            loss = rec_loss.mean() + matching_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            batch_losses.append(loss.item())
            epoch_loss += loss.item()

            tmp_loss = rec_loss.clone().detach().cpu().tolist()
            nonshuffle_loss = [l for i, l in enumerate(tmp_loss) if batch['shuffle'][i] == self.shuffle_key_idx['nonshuffle']]
            nonshuffle_rec_losses.append(np.mean(nonshuffle_loss))
            shuffle_loss = [l for i, l in enumerate(tmp_loss) if batch['shuffle'][i] == self.shuffle_key_idx['shuffle']]
            shuffle_rec_losses.append(np.mean(shuffle_loss))

        avg_epoch_loss = epoch_loss / i
        avg_non_rec_loss = np.mean(nonshuffle_rec_losses)
        if len(shuffle_rec_losses) != 0:
            avg_shu_rec_loss = np.mean(shuffle_rec_losses)
        else:
            avg_shu_rec_loss = 0

        return avg_epoch_loss, avg_non_rec_loss, avg_shu_rec_loss
    
    @torch.no_grad()
    def validate(self, dataloder, topk, best_model=None):
        if best_model != None:
            self.model = best_model
        recalls = {}
        mrrs = {}
        ndcgs = {}
        if isinstance(topk, int):
            topk = [topk]
        for k in topk:
            recalls[f'@{k}'] = []
            mrrs[f'@{k}'] = []
            ndcgs[f'@{k}'] = []

        with torch.no_grad():
            valid_batch_iter = tqdm(enumerate(dataloder), total=len(dataloder))
            for i, batch in valid_batch_iter:
                batch = self.after_epoch_start(batch)
                
                _, predictions = self.model(batch, 'orig_sess')
                # batch, predictions = self.after_epoch_start(batch)

                logits = self.predict(predictions)

                for k in topk:
                    recall, mrr, ndcg = evaluate(logits, batch['labels'], k=k)
                    recalls[f'@{k}'].append(recall)
                    mrrs[f'@{k}'].append(mrr)
                    ndcgs[f'@{k}'].append(ndcg)

        for k in topk:
            recalls[f'@{k}'] = np.mean(recalls[f'@{k}'])
            mrrs[f'@{k}'] = np.mean(mrrs[f'@{k}'])
            ndcgs[f'@{k}'] = np.mean(ndcgs[f'@{k}'])

        return recalls, mrrs, ndcgs
    
    def calculate_loss(self, predictions, batch):
        all_embs = self.model.backbone.item_embedding.weight
        logits = torch.matmul(predictions, all_embs.transpose(0, 1))
        loss = self.loss_func(logits, batch['labels'])

        return loss

    def predict(self, predictions):
        all_embs = self.model.backbone.item_embedding.weight
        logits = torch.matmul(predictions, all_embs.transpose(0, 1))
        logits = F.softmax(logits, dim=1)

        return logits

class VICReg(nn.Module):
    def __init__(self, input_size, args, device):
        super().__init__()
        self.n_items = input_size
        self.args = args
        self.device = device

        self.num_features = args.hidden_size
        self.backbone = SRGNN(input_size, args, device)

        self.mask_default = self.mask_correlated_samples(batch_size=args.batch_size)

    def forward(self, batch, input_str='orig_sess', len_str='lens', get_last=True):

        hidden, preds = self.backbone(batch, input_str, len_str, get_last)

        return hidden, preds
    
    def _inv_loss(self, x, y, loss_type):
        if loss_type.lower() == 'mse':
            repr_loss = F.mse_loss(x, y)
        elif loss_type.lower() == 'infonce':
            repr_logits, repr_labels = self.info_nce(x, y, self.args.temperature, x.size(0))
            repr_loss = F.cross_entropy(repr_logits, repr_labels)
        else:
            raise ValueError
        return repr_loss
    
    def _vicreg_loss(self, x, y):
        repr_loss = self.args.inv_coeff * F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.args.var_coeff * (
            torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss = self.args.cov_coeff * cov_loss

        return repr_loss, std_loss, cov_loss

    def _finegrained_matching_loss(
        self, maps_1, maps_2, location_1, location_2, mask1, mask2, j, epoch
    ):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # item-based matching
        if epoch > self.args.warm_up_epoch:
            num_matches_on_l2 = self.args.num_matches

            maps_1_filtered, maps_1_nn = self.item_based_matching(
                maps_1, maps_2, num_matches=num_matches_on_l2[0], mask1=mask1, mask2=mask2
            )
            maps_2_filtered, maps_2_nn = self.item_based_matching(
                maps_2, maps_1, num_matches=num_matches_on_l2[1], mask1=mask1, mask2=mask2
            )

            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)
            inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        maps_1_filtered, maps_1_nn = self.similarity_based_matching(
            location_1, location_2, maps_1, maps_2, mask1, mask2, j
        )

        inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
        var_loss = var_loss + var_loss_1
        cov_loss = cov_loss + cov_loss_1
        inv_loss = inv_loss + inv_loss_1

        return inv_loss, var_loss, cov_loss

    def finegrained_matching_loss(self, maps_embedding, locations, mask, epoch):
        num_views = len(maps_embedding)
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss_this, var_loss_this, cov_loss_this = self._finegrained_matching_loss(
                    maps_embedding[i], maps_embedding[j], locations[i], locations[j], mask[i], mask[j], j, epoch
                )
                inv_loss = inv_loss + inv_loss_this
                var_loss = var_loss + var_loss_this
                cov_loss = cov_loss + cov_loss_this
                iter_ += 1

        inv_loss = inv_loss / iter_
        var_loss = var_loss / iter_
        cov_loss = cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def global_loss(self, embedding, maps=False):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.args.inv_coeff * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = embedding[i]
            x = x - x.mean(dim=0)
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + off_diagonal(cov_x).pow_(2).sum().div(
                self.args.embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.args.var_coeff * var_loss / iter_
        cov_loss = self.args.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def compute_finegrained_matching_loss(self, batch,
                                          seq_hidden1, seq_hidden2,
                                          seq_pred1, seg_pred2, epoch):
        loss = 0.0
        mask1 = batch['orig_sess'].gt(0)
        mask2 = batch['aug1'].gt(0)
        mask = torch.cat([mask1.unsqueeze(0), mask2.unsqueeze(0)], dim=0)

        seq_hidden = torch.cat([seq_hidden1.unsqueeze(0),
                                seq_hidden2.unsqueeze(0)], dim=0)
        seq_pred = torch.cat([seq_pred1.unsqueeze(0), seg_pred2.unsqueeze(0)], dim=0)

        v1_position = torch.arange(self.args.maxlen).unsqueeze(0).repeat(
            batch['position_labels'].size(0), 1)
        v2_position = batch['position_labels'].masked_fill(
            batch['position_labels'] < 0, 0)
        locations = torch.cat([v1_position.unsqueeze(0),
                               v2_position.unsqueeze(0)], dim=0).to(self.device)

        # Global criterion
        if self.args.alpha < 1.0:
            inv_loss, var_loss, cov_loss = self.global_loss(seq_pred)
            loss = loss + (1 - self.args.alpha) * (inv_loss + var_loss + cov_loss)

        # Local criterion
        # Maps shape: B, C, H, W
        # With convnext actual maps shape is: B, H * W, C
        if self.args.alpha > 0.0:
            (maps_inv_loss, maps_var_loss, maps_cov_loss) = self.finegrained_matching_loss(seq_hidden, locations, mask, epoch)
            loss = loss + (self.args.alpha) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )

        return loss
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def info_nce(self, z_i, z_j, temperature, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temperature
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.args.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def nearest_neighbores(self, input_maps, candidate_maps, distances, num_matches):
        batch_size = input_maps.size(0)

        if num_matches is None or num_matches == -1:
            num_matches = input_maps.size(1)

        topk_values, topk_indices = distances.topk(k=1, largest=False)
        topk_values = topk_values.squeeze(-1)
        topk_indices = topk_indices.squeeze(-1)

        sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
        sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

        mask = torch.stack(
            [
                torch.where(sorted_indices_indices[i] < num_matches, True, False)
                for i in range(batch_size)
            ]
        )
        topk_indices_selected = topk_indices.masked_select(mask)
        topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

        indices = (
            torch.arange(0, topk_values.size(1))
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(topk_values.device)
        )
        indices_selected = indices.masked_select(mask)
        indices_selected = indices_selected.reshape(batch_size, num_matches)

        filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
        filtered_candidate_maps = batched_index_select(
            candidate_maps, 1, topk_indices_selected
        )

        return filtered_input_maps, filtered_candidate_maps


    def item_based_matching(self, input_maps, candidate_maps, num_matches, mask1, mask2):
        """
        input_maps: (B, H * W, C)
        candidate_maps: (B, H * W, C)
        """
        distances = torch.cdist(input_maps, candidate_maps)
        mask_tensor1 = mask1.unsqueeze(1) * mask1.unsqueeze(-1)
        mask_tensor2 = mask2.unsqueeze(1) * mask2.unsqueeze(-1)
        mask_tensor = mask_tensor1 * mask_tensor2
        distances = distances.masked_fill(~mask_tensor, np.Inf)
        return self.nearest_neighbores(input_maps, candidate_maps, distances, num_matches)

    def similarity_based_matching(
            self, input_location, candidate_location, input_maps, candidate_maps, mask1, mask2, j
        ):
        # mask_tensor1 = mask1.unsqueeze(1) * mask1.unsqueeze(-1)
        # mask_tensor2 = mask2.unsqueeze(1) * mask2.unsqueeze(-1)
        if j == 1:
            perm_mat = candidate_location
            coverted_maps = candidate_maps
            mask = mask2
            # mask_tensor = mask_tensor2
        elif j == 0:
            perm_mat = input_location
            coverted_maps = input_maps
            mask = mask1
            # mask_tensor = mask_tensor1

        perm_mat = F.one_hot(perm_mat, num_classes=self.args.maxlen) * mask.unsqueeze(-1)
        zeros = torch.zeros_like(perm_mat).to(self.device)
        ones = (~mask).long()
        r = torch.arange(self.args.maxlen).to(self.device)
        zeros[:, r, r] = ones
        perm_mat += zeros

        candidate_maps = torch.matmul(coverted_maps.transpose(2, 1),
                                      perm_mat.float()).transpose(2, 1)
        return input_maps, candidate_maps


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)