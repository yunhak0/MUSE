import os
import pickle
import random
import numpy as np
import torch
from copy import deepcopy
import logging
import sys
import os.path as osp

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # torch.multiprocessing.set_start_method('spawn')

def setup_logger(name, log_dir=None, filename='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(4)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(osp.join(log_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb'))

def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        ## Hyperparameters
        if name not in ['device', 'val_epoch', 'topk', 'theanorc']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)

def get_last_item(X, lengths):
    if isinstance(lengths, list):
        lengths = torch.as_tensor(lengths).to(X.device)
    idx = lengths - 1
    idx = idx.view(-1, 1, 1).expand(-1, -1, X.shape[-1])
    out = X.gather(dim=1, index=idx)
    return out.squeeze(1)
