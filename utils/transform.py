import random
import numpy as np
import torch
import copy
from scipy.special import softmax
from numpy.random import choice
import time

class Random_CMR(object):
    """Randomly pick one data augmentation type every time call"""
    def __init__(self, transition_r=0.2, reorder_r=0.2, maxlen=19, transition_dict=None, device=None):
            self.data_augmentation_methods = [Transition(prob=transition_r, maxlen=maxlen, transition_dict=transition_dict, device=device),
                                             Reorder(reorder_r=reorder_r)]

    def __call__(self, sequence, shuffle='shuffle'):
        #randint generate int x in range: a <= x <= b
        if shuffle == 'shuffle':
            augment_method_idx = 0
            augment_method = self.data_augmentation_methods[augment_method_idx]
        else:
            augment_method_idx = 1
            augment_method = self.data_augmentation_methods[augment_method_idx]

        return augment_method(sequence, shuffle)

class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""
    def __init__(self, reorder_r=0.2):
        self.reorder_r = reorder_r

    def __call__(self, sequence, shuffle='shuffle'):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        length = len(copied_sequence)
        if length > 1:
            sub_seq_length = int(self.reorder_r*length)
            if sub_seq_length == 0:
                reordered_seq = copied_sequence
                reordered_pos = [range(len(copied_sequence))]
            else:
                if self.reorder_r == 1.0:
                    start_index = 0
                else:
                    start_index = random.randint(0, length-sub_seq_length-1)

                sub_seq = copied_sequence[start_index:start_index+sub_seq_length]
                x = list(enumerate(sub_seq))
                random.shuffle(x)
                sub_index, sub_item = zip(*x)
                sub_index = [i + start_index for i in sub_index]
                
                reordered_seq = copied_sequence[:start_index] + list(sub_item) + \
                                copied_sequence[start_index+sub_seq_length:]
                reordered_pos = list(range(start_index)) + sub_index + \
                    list(range(start_index+sub_seq_length, length))

                assert length == len(reordered_seq)
                assert length == len(reordered_pos)
        else:
            reordered_seq = copied_sequence
            reordered_pos = [0]
        return reordered_seq, reordered_pos

class Transition(object):
    """Replace based on transition_frequency"""
    def __init__(self, prob=0.2, maxlen=19, transition_dict=None, device=None):
        self.prob = prob
        self.maxlen = maxlen
        self.transition_dict = transition_dict
        self.device = device

    def __call__(self, sequence, shuffle='shuffle', og_idx=None):
        # make a deep copy to avoid original sequence be modified
        transition_source = self.transition_dict['source']
        transition_target = self.transition_dict['target']
        
        copied_sequence = copy.deepcopy(sequence)
        if og_idx != None:
            pos = og_idx
        else:
            pos = np.arange(len(copied_sequence))

        sources = copied_sequence[:-1]
        targets = copied_sequence[1:]

        candidate_mat = transition_source[sources].multiply(transition_target[targets])
        idx = np.array(candidate_mat.sum(1) > 0).reshape(-1)

        if sum(idx) > 0:
            mat = candidate_mat[idx]
            sampled = np.array([np.random.choice(mat.indices[mat.indptr[i]:mat.indptr[i+1]],
                                        p=softmax(mat.data[mat.indptr[i]:mat.indptr[i+1]])) for i in range(sum(idx))])

            remainig = self.maxlen - len(copied_sequence)
            if remainig == 0:
                return copied_sequence, pos
            else:
                if remainig < len(sampled):
                    mask = np.zeros_like(sampled, dtype=bool)
                    chosen = np.random.choice(np.arange(len(mask)), remainig, replace=False)
                    mask[chosen] = True
                else:
                    mask = np.random.binomial(np.ones_like(sampled), p=self.prob) == 1

                new_sequence = np.zeros_like(copied_sequence * 2)[:-1]
                new_sequence[np.arange(0, len(new_sequence), 2)] = copied_sequence
                new_sequence[np.arange(1, len(new_sequence), 2)[idx][mask]] = sampled[mask]
                pos_seq = np.zeros_like(new_sequence) -1
                pos_seq[np.arange(0, len(new_sequence), 2)] = pos

                nnz_idx = new_sequence != 0

                copied_sequence = list(new_sequence[nnz_idx])
                pos = list(pos_seq[nnz_idx])
 
        return copied_sequence, pos
