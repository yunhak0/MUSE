import torch

def evaluate(indices, targets, k=20):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
        ndcg (float): the ndcg score
    """

    _, indices = torch.topk(indices, k, -1)
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    
    recall = get_recall(hits, targets)
    mrr = get_mrr(hits, targets)
    ndcg = get_ndcg(hits, targets)

    return recall, mrr, ndcg

@torch.jit.script
def get_recall(hits, targets):
    n_hits = hits[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall

@torch.jit.script
def get_ndcg(hits, targets):
    log_ranks = torch.log2(hits[:, -1] + 2)
    log_ranks = log_ranks.float()
    r_log_ranks = torch.reciprocal(log_ranks)
    ndcg = float(torch.sum(r_log_ranks).detach()) / targets.size(0)
    return ndcg

@torch.jit.script
def get_mrr(hits, targets):
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = float(torch.sum(rranks).detach()) / targets.size(0)
    return mrr
