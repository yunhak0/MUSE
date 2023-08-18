import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self, loss_type='TOP1'):
        """ An abstract loss function that can supports custom loss functions compatible with PyTorch."""
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'CE':
            self._loss_fn = CrossEntropyLoss()
        elif loss_type == 'sampledCE':
            self._loss_fn = SampledCrossEntropyLoss()
        elif loss_type == 'TOP1':
            self._loss_fn = TOP1Loss()
        elif loss_type == 'BPR':
            self._loss_fn = BPRLoss()
        elif loss_type == 'TOP1-max':
            self._loss_fn = TOP1_max()
        elif loss_type == 'BPR-max':
            self._loss_fn = BPR_max()
        elif loss_type == 'BCE':
            self._loss_fn = BinaryCrossEntropyLoss()
        else:
            raise NotImplementedError

    def forward(self, logit, target=None):
        if self.loss_type in ['CE', 'BCE']:
            return self._loss_fn(logit, target)
        else:
            return self._loss_fn(logit)


class CrossEntropyLoss(nn.Module):
    """ original CrossEntropyLoss """
    def __init__(self):
        """
        Args:
             use_cuda (bool): whether to use cuda or not
        """
        super(CrossEntropyLoss, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logit, target):
        return self.xe_loss(logit, target)


class SampledCrossEntropyLoss(nn.Module):
    """ CrossEntropyLoss with n_classes = batch_size = the number of samples in the session-parallel mini-batch """
    def __init__(self):
        """
        Args:
             use_cuda (bool): whether to use cuda or not
        """
        super(SampledCrossEntropyLoss, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss()

    def forward(self, logit):
        batch_size = logit.size(1)
        target = Variable(torch.arange(batch_size).long()).to(logit.device)

        return self.xe_loss(logit, target)


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logit, target):
        return self.bce_loss(logit, target)


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # differences between the item scores
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        # final loss
        loss = -torch.mean(F.logsigmoid(diff))
        return loss


class BPR_max(nn.Module):
    def __init__(self):
        super(BPR_max, self).__init__()
    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1)
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        loss = -torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))
        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()
    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
        return loss


class TOP1_max(nn.Module):
    def __init__(self):
        super(TOP1_max, self).__init__()

    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1)
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))
        return loss


class NCELoss(nn.Module):
    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)
        
    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two):
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=batch_sample_one.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss

"""
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07,
                 contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class L2NormLoss(nn.Module):
    def __init__(self):
        super(L2NormLoss, self).__init__()

    def forward(self, logit, target):
        return torch.norm(logit - target, dim=1).sum() / logit.shape[0]
