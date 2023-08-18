from copy import deepcopy

class ValidChecker:
    def __init__(self, max_tolerate=2, eval_mode=1):
        self.max_tolerate = max_tolerate
        self.cur_tolerate = 0
        self.best_val = -1
        self.best_results = None
        self.best_epoch = 0
        self.best_ptr_model = None
        self.best_model = None
        self.earlystop = False
        self.eval_mode = eval_mode

    def __call__(self, cur_valid, epoch, model, ptr_model=None):
        """
        For main training
        """
        if self.best_val < cur_valid[self.eval_mode]: # cur_valid: recall, mrr, ndcg, acc
            self.best_val = cur_valid[self.eval_mode]
            if ptr_model is not None:
                self.best_ptr_model = deepcopy(ptr_model)
            self.best_model = deepcopy(model)
            self.best_epoch = epoch
            self.best_results = cur_valid
            self.cur_tolerate = 0
            self.earlystop = False
        else:
            self.cur_tolerate += 1
            if self.cur_tolerate >= self.max_tolerate:
                self.earlystop = True
            else:
                self.earlystop = False
