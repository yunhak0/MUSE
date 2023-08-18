import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.data import load_data, BaseData, collate_fn
from torch.utils.data import DataLoader
from models.valid_checker import ValidChecker
from models.metric import evaluate
from scipy.sparse import csr_matrix


class Embedder:
    def __init__(self, args):
        self.args = args
        self.set_configs(args)
        self.topk = eval(self.args.topk)

        # GPU
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"

    def set_configs(self, args):
        self.trial = f'{self.args.dataset}_{self.args.embedder}'

        # PATH SETTING
        base_path = '.'
        self.data_dir = f'{base_path}/data/{self.args.dataset}-{self.args.seed}/all'  # Data
        # self.log_path = f'{base_path}/logs'  # All Performance tracking regardless of Trial
        # self.log_info_path = f'{base_path}/logs/{self.trial}'  # Logging
        self.ckpt_path = f'{base_path}/checkpoints/{self.trial}'
        # self.result_path = f'{base_path}/results/{self.trial}'

        # os.makedirs(self.log_path, exist_ok=True)
        # os.makedirs(self.log_info_path, exist_ok=True)
        # os.makedirs(self.result_path, exist_ok=True)

        if self.args.save_ckpt:
            os.makedirs(self.ckpt_path, exist_ok=True)
        

    def get_context_key(self, context_col):
        key_idx = {}
        for i, v in enumerate(list(set(self.sess_map[context_col].to_list()))):
            key_idx[v] = i
        return key_idx

    def get_sess_context(self, data, context_col, key_idx):
        try:
            df = pd.DataFrame({'session_id': data[2]}).merge(
                self.sess_map[self.sess_map['train'].isin(['train_valid', 'test'])],
                on='session_id', how='left'
            )
            context_list = [key_idx[x] for x in list(df[context_col])]
        except:
            df = pd.DataFrame({'session_id': data[2]}).merge(
                self.sess_map[self.sess_map['train'].isin(['valid'])],
                on='session_id', how='left'
            )
            context_list = [key_idx[x] for x in list(df[context_col])]

        return context_list

    def load_dataset(self, load_type='train'):
        train_data, valid_data, train_full_data, test_data, \
            self.n_items, self.sess_map = load_data(root=self.data_dir)
        # self.n_items = self.item_map.shape[0] + 1
        self.n_sesss = self.sess_map.shape[0]

        if ('muse' in self.args.embedder.lower()):
            try:
                # load
                with open(f'{self.data_dir}/adj_train.pickle', 'rb') as f:
                    self.adj_train_dict = pickle.load(f)
                print(f'Successfully loaded adj train ...!')
            
                with open(f'{self.data_dir}/adj_train_full.pickle', 'rb') as f:
                    self.adj_train_full_dict = pickle.load(f)
                print(f'Successfully loaded adj train full ...!')
            except:
                # save
                self.adj_train_dict = self.transition_adj(train_data, self.sess_map, self.n_items)
                self.adj_train_full_dict = self.transition_adj(train_full_data, self.sess_map, self.n_items)
        
                with open(f'{self.data_dir}/adj_train.pickle', 'wb') as f:
                    pickle.dump(self.adj_train_dict, f, pickle.HIGHEST_PROTOCOL)
                with open(f'{self.data_dir}/adj_train_full.pickle', 'wb') as f:
                    pickle.dump(self.adj_train_full_dict, f, pickle.HIGHEST_PROTOCOL)

        else:
            self.adj_train_dict = None
            self.adj_train_full_dict = None


        # session_info
        self.sess_map['context_shuffle'] = self.sess_map['context_session'] + '_' + \
            self.sess_map['shuffle_session']
        
        # self.shuffle_key_idx = self.get_context_key('shuffle_session')
        self.shuffle_key_idx = {}
        self.shuffle_key_idx['nonshuffle'] = 0
        self.shuffle_key_idx['hybrid'] = 1
        self.shuffle_key_idx['shuffle'] = 2
        self.context_key_idx = self.get_context_key('context_session')
        self.hybrid_key_idx = self.get_context_key('context_shuffle')

        shuffle_idx_train = self.get_sess_context(train_data, 'shuffle_session', self.shuffle_key_idx)
        shuffle_idx_valid = self.get_sess_context(valid_data, 'shuffle_session', self.shuffle_key_idx)
        shuffle_idx_train_full = self.get_sess_context(train_full_data, 'shuffle_session', self.shuffle_key_idx)
        shuffle_idx_test = self.get_sess_context(test_data, 'shuffle_session', self.shuffle_key_idx)

        context_idx_train = self.get_sess_context(train_data, 'context_session', self.context_key_idx)
        context_idx_valid = self.get_sess_context(valid_data, 'context_session', self.context_key_idx)
        context_idx_train_full = self.get_sess_context(train_full_data, 'context_session', self.context_key_idx)
        context_idx_test = self.get_sess_context(test_data, 'context_session', self.context_key_idx)

        hybrid_idx_train = self.get_sess_context(train_data, 'context_shuffle', self.hybrid_key_idx)
        hybrid_idx_valid = self.get_sess_context(valid_data, 'context_shuffle', self.hybrid_key_idx)
        hybrid_idx_train_full = self.get_sess_context(train_full_data, 'context_shuffle', self.hybrid_key_idx)
        hybrid_idx_test = self.get_sess_context(test_data, 'context_shuffle', self.hybrid_key_idx)


        transform = None
        
        c_collate_fn_train = collate_fn(self.args, self.n_items, transform, transition_adj=self.adj_train_dict, device=self.device, train=True)
        c_collate_fn_val = collate_fn(self.args, self.n_items, transform, transition_adj=self.adj_train_dict, device=self.device, train=False)
        c_collate_fn_train_full = collate_fn(self.args, self.n_items, transform, transition_adj=self.adj_train_full_dict, device=self.device, train=True)
        c_collate_fn_test = collate_fn(self.args, self.n_items, transform, transition_adj=self.adj_train_full_dict, device=self.device, train=False)

        # Train Data Loader
        self.train_data = BaseData(train_data, shuffle_idx_train, context_idx_train, hybrid_idx_train,
                                   ranshu=True)
        self.train_loader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True,
                                       collate_fn=c_collate_fn_train, num_workers=4)
        self.train_loader.data_type = 'train'

        self.valid_data = BaseData(valid_data, shuffle_idx_valid, context_idx_valid, hybrid_idx_valid)
        self.valid_loader = DataLoader(self.valid_data, batch_size=self.args.batch_size,
                                       shuffle=False, collate_fn=c_collate_fn_val, num_workers=4)
        self.valid_loader.data_type = 'train'

        self.train_full_data = BaseData(train_full_data, shuffle_idx_train_full, context_idx_train_full, hybrid_idx_train_full,
                                        ranshu=True)
        self.train_full_loader = DataLoader(self.train_full_data, batch_size=self.args.batch_size,
                                            shuffle=True, collate_fn=c_collate_fn_train_full, num_workers=4)
        self.train_full_loader.data_type = 'train_full'
        
        self.test_data = BaseData(test_data, shuffle_idx_test, context_idx_test, hybrid_idx_test)
        self.test_loader = DataLoader(self.test_data, batch_size=self.args.batch_size,
                                      shuffle=False, collate_fn=c_collate_fn_test, num_workers=4)
        self.test_loader.data_type = 'train_full'

        self.ns_test_data = BaseData(
            data=(
                np.array(test_data[0], dtype=object)[np.array(shuffle_idx_test) == self.shuffle_key_idx['nonshuffle']].tolist(),
                np.array(test_data[1], dtype=object)[np.array(shuffle_idx_test) == self.shuffle_key_idx['nonshuffle']].tolist(),
                np.array(test_data[2], dtype=object)[np.array(shuffle_idx_test) == self.shuffle_key_idx['nonshuffle']].tolist(),
                np.array(test_data[3], dtype=object)[np.array(shuffle_idx_test) == self.shuffle_key_idx['nonshuffle']].tolist()
            ),
            shuffle_idx=[x for x in shuffle_idx_test if x == self.shuffle_key_idx['nonshuffle']],
            context_idx=[x for i, x in enumerate(context_idx_test) if shuffle_idx_test[i] == self.shuffle_key_idx['nonshuffle']],
            hybrid_idx=[x for i, x in enumerate(hybrid_idx_test) if shuffle_idx_test[i] == self.shuffle_key_idx['nonshuffle']]
        )
        self.ns_test_loader = DataLoader(self.ns_test_data, batch_size=self.args.batch_size,
                                         shuffle=False, collate_fn=c_collate_fn_test, num_workers=4)
        self.ns_test_loader.data_type = 'train_full'

        self.sh_test_data = BaseData(
            data=(
                np.array(test_data[0], dtype=object)[np.array(shuffle_idx_test) != self.shuffle_key_idx['nonshuffle']].tolist(),
                np.array(test_data[1], dtype=object)[np.array(shuffle_idx_test) != self.shuffle_key_idx['nonshuffle']].tolist(),
                np.array(test_data[2], dtype=object)[np.array(shuffle_idx_test) != self.shuffle_key_idx['nonshuffle']].tolist(),
                np.array(test_data[3], dtype=object)[np.array(shuffle_idx_test) != self.shuffle_key_idx['nonshuffle']].tolist()
            ),
            shuffle_idx=[x for x in shuffle_idx_test if x != self.shuffle_key_idx['nonshuffle']],
            context_idx=[x for i, x in enumerate(context_idx_test) if shuffle_idx_test[i] != self.shuffle_key_idx['nonshuffle']],
            hybrid_idx=[x for i, x in enumerate(hybrid_idx_test) if shuffle_idx_test[i] != self.shuffle_key_idx['nonshuffle']]
        )
        self.sh_test_loader = DataLoader(self.sh_test_data, batch_size=self.args.batch_size,
                                        shuffle=False, collate_fn=c_collate_fn_test, num_workers=4)
        self.sh_test_loader.data_type = 'train_full'

    def load_model(self):
        raise NotImplementedError

    def train_before_epoch_start(self):
        pass

    def after_epoch_start(self, batch):
        batch['orig_sess'] = batch['orig_sess'].to(self.device, non_blocking=True)
        batch['right_padded_sesss'] = batch['right_padded_sesss'].to(self.device, non_blocking=True)
        batch['labels'] = batch['labels'].to(self.device, non_blocking=True)

        return batch

    def train(self):
        print('[TRAIN] Process ...')
        self.valid_checker = ValidChecker(max_tolerate=self.args.patience)
        self.load_model()

        for epoch in range(1, self.args.n_epochs+1):
            start = time.time()
            epoch_loss, nonshuffle_loss, shuffle_loss = self.train_epoch(dataloader=self.train_loader, epoch=epoch)

            st = '[TRAIN][Seed {}/{}][Epoch {}/{}] Loss: {:.4f} - NonShuffle Loss: {:.4f} - Shuffle Loss: {:.4f} (spent: {:.2f})'.format(
                self.args.seed, self.args.n_runs-1, epoch, self.args.n_epochs, epoch_loss, nonshuffle_loss, shuffle_loss, time.time() - start
            )
            print(st)

            if epoch % self.args.val_epoch == 0:
                valid_recall, valid_mrr, valid_ndcg = self.validate(self.valid_loader, 5)
                valid_recall = valid_recall['@5']
                valid_mrr = valid_mrr['@5']
                valid_ndcg = valid_ndcg['@5']
                valid_results = [valid_recall, valid_mrr, valid_ndcg]
                self.valid_checker(valid_results, epoch, self.model)
                st_val = f"[VALID][Seed {self.args.seed}/{self.args.n_runs-1}]"
                st_val += f"[Epoch {epoch}/{self.args.n_epochs}] "
                st_val += f"Recall@5: {valid_results[0]:.4f} | " 
                st_val += f"MRR@5: {valid_results[1]:.4f} | " 
                st_val += f"NDCG@5 {valid_results[2]:.4f}"
                print(st_val)

                if self.valid_checker.earlystop:
                    st = f'** Early Stopping: {epoch} out of {self.args.n_epochs}!! **'
                    print(st)
                    break
        

        print(f'[TRAIN_FULL] Process ... Best Epoch is {self.valid_checker.best_epoch}/{self.args.n_epochs}')
        self.load_model()

        for epoch in range(1, self.valid_checker.best_epoch+1): # range(1, self.args.n_epochs+1)
            start = time.time()
            epoch_loss, nonshuffle_loss, shuffle_loss = self.train_epoch(dataloader=self.train_full_loader, epoch=epoch)

            st = '[TRAIN_FULL][Seed {}/{}][Epoch {}/{}] Loss: {:.4f} - NonShuffle Loss: {:.4f} - Shuffle Loss: {:.4f} (spent: {:.2f})'.format(
                self.args.seed, self.args.n_runs-1, epoch, self.valid_checker.best_epoch, epoch_loss, nonshuffle_loss, shuffle_loss, time.time() - start
            )

        test_recall, test_mrr, test_ndcg = self.validate(self.test_loader, topk=self.topk)
        print_test_recall = [float(f"{x:.4f}") for x in list(test_recall.values())]
        print_test_mrr = [float(f"{x:.4f}") for x in list(test_mrr.values())]
        print_test_ndcg = [float(f"{x:.4f}") for x in list(test_ndcg.values())]

        test_recall_ns, test_mrr_ns, test_ndcg_ns = self.validate(self.ns_test_loader, topk=self.topk)
        print_test_recall_ns = [float(f"{x:.4f}") for x in list(test_recall_ns.values())]
        print_test_mrr_ns = [float(f"{x:.4f}") for x in list(test_mrr_ns.values())]
        print_test_ndcg_ns = [float(f"{x:.4f}") for x in list(test_ndcg_ns.values())]

        test_recall_sh, test_mrr_sh, test_ndcg_sh = self.validate(self.sh_test_loader, topk=self.topk)
        print_test_recall_sh = [float(f"{x:.4f}") for x in list(test_recall_sh.values())]
        print_test_mrr_sh = [float(f"{x:.4f}") for x in list(test_mrr_sh.values())]
        print_test_ndcg_sh = [float(f"{x:.4f}") for x in list(test_ndcg_sh.values())]

        # Return dict
        overall_performance = {}
        overall_performance['all'] = {}
        overall_performance['nonshuffle'] = {}
        overall_performance['shuffle'] = {}

        overall_performance['nonshuffle']['recall'] = print_test_recall_ns
        overall_performance['nonshuffle']['mrr'] = print_test_mrr_ns
        overall_performance['nonshuffle']['ndcg'] = print_test_ndcg_ns

        overall_performance['all']['recall'] = print_test_recall
        overall_performance['all']['mrr'] = print_test_mrr
        overall_performance['all']['ndcg'] = print_test_ndcg

        overall_performance['shuffle']['recall'] = print_test_recall_sh
        overall_performance['shuffle']['mrr'] = print_test_mrr_sh
        overall_performance['shuffle']['ndcg'] = print_test_ndcg_sh

        return overall_performance

    def fit(self):
        self.start_time = time.time()

        self.load_dataset()

        overall_performance = self.train()

        # Save Model
        if self.args.save_ckpt:
            torch.save(self.valid_checker.best_model,
                       os.path.join(self.ckpt_path, f'{self.args.dataset}_{self.args.embedder}_valid_best_model_seed_{self.args.seed}.pt'))
            torch.save(self.model,
                       os.path.join(self.ckpt_path, f'{self.args.dataset}_{self.args.embedder}_final_model_seed_{self.args.seed}.pt'))

        return overall_performance

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        rec_losses = []
        shuffle_rec_losses = []
        nonshuffle_rec_losses = []
        epoch_loss = 0
        self.train_before_epoch_start()
        train_batch_iter = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, batch in train_batch_iter:
            batch = self.after_epoch_start(batch)

            predictions = self.model(batch, 'orig_sess', get_last=True)
            rec_loss = self.calculate_loss(predictions, batch)
            loss = rec_loss.mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            tmp_loss = rec_loss.clone().detach().cpu()
            if self.args.loss_type == 'CE':
                if (batch['shuffle'] == self.shuffle_key_idx['nonshuffle']).sum() > 0:
                    nonshuffle_loss = [l for i, l in enumerate(tmp_loss) if batch['shuffle'][i] == self.shuffle_key_idx['nonshuffle']]
                    nonshuffle_rec_losses.append(np.mean(nonshuffle_loss))
                    shuffle_loss = [l for i, l in enumerate(tmp_loss) if batch['shuffle'][i] == self.shuffle_key_idx['shuffle']]
                    shuffle_rec_losses.append(np.mean(shuffle_loss))

        avg_epoch_loss = epoch_loss / i
        if len(nonshuffle_rec_losses) != 0:
            avg_non_rec_loss = np.mean(nonshuffle_rec_losses)
        else:
            avg_non_rec_loss = 0

        if len(shuffle_rec_losses) != 0:
            avg_shu_rec_loss = np.mean(shuffle_rec_losses)
        else:
            avg_shu_rec_loss = 0

        return avg_epoch_loss, avg_non_rec_loss, avg_shu_rec_loss

    @torch.no_grad()
    def validate(self, dataloder, topk):
        self.model.eval()
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
                
                predictions = self.model(batch, 'orig_sess')
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
        all_embs = self.model.item_embedding.weight
        logits = torch.matmul(predictions, all_embs.transpose(0, 1))
        loss = self.loss_func(logits, batch['labels'])

        return loss

    def predict(self, predictions):
        all_embs = self.model.item_embedding.weight
        logits = torch.matmul(predictions, all_embs.transpose(0, 1))
        logits = F.softmax(logits, dim=1)

        return logits
    
    def transition_adj(self, train_data, sess_map, n_items):
        seq_dict = {}
        seq_dict['seq'] = train_data[0]
        seq_dict['target'] = train_data[1]
        seq_dict['seq_target'] = [train_data[0][i] + [train_data[1][i]] for i in range(len(train_data[0]))]
        seq_dict['session_id'] = train_data[2]
        df_seq = pd.DataFrame(seq_dict)

        df_tmp = sess_map.merge(df_seq, how='inner', on='session_id')
        df_sess =  df_tmp.drop_duplicates(['session_id'], keep='first').reset_index(drop=True)
        df_sess['seq_length'] = [len(seq) for seq in df_sess.seq]

        ## Generate New all_seqs 
        # all_seqs = dict(zip(df_sess.session_id, df_sess.seq_target)) # including target
        all_seqs = dict(zip(df_sess.session_id, df_sess.seq)) # excluding target
        df_all_seqs = pd.DataFrame(list(all_seqs.items()), columns=['session_id', 'ItemId'])
        df_all_seqs['Played'] = list(df_sess.not_skipped)
        df_all_seqs = df_all_seqs.merge(sess_map[sess_map['train'] == 'train_valid'], on='session_id', how='left')
        
        ids = []
        items = []
        played = []

        ns_ids = []
        ns_items = []
        ns_played = []

        s_ids = []
        s_items = []
        s_played = []
        
        # all
        for row in tqdm(df_all_seqs.iterrows(), total=len(df_all_seqs)):
            for i in range(len(row[1]['ItemId'])):
                ids.append(row[1]['session_id'])
                items.append(row[1]['ItemId'][i])
                played.append(row[1]['Played'][i])

        # from all
        for row in tqdm(df_all_seqs[df_all_seqs['shuffle_session'] == 'nonshuffle'].iterrows(), total=sum(df_all_seqs['shuffle_session'] == 'nonshuffle')):
            for i in range(len(row[1]['ItemId'])):
                ns_ids.append(row[1]['session_id'])
                ns_items.append(row[1]['ItemId'][i])
                ns_played.append(row[1]['Played'][i])
        
        for row in tqdm(df_all_seqs[df_all_seqs['shuffle_session'] != 'nonshuffle'].iterrows(), total=sum(df_all_seqs['shuffle_session'] != 'nonshuffle')):
            for j in range(len(row[1]['ItemId'])):
                s_ids.append(row[1]['session_id'])
                s_items.append(row[1]['ItemId'][j])
                s_played.append(row[1]['Played'][j])

        transition_dict = {}

        for i in range(3):
            if i == 0:
                ids = ids
                items = items
                played = played
            
            elif i == 1:
                ids = ns_ids
                items = ns_items
                played = ns_played

            elif i == 2:
                ids = s_ids
                items = s_items
                played = s_played

            org_df = pd.DataFrame({'session_id': ids, 'item_id': items, 'played': played})
            
            org_df['next_item_id'] = org_df.groupby(['session_id'])['item_id'].shift(-1)
            org_df['next_played'] = org_df.groupby(['session_id'])['played'].shift(-1)

            org_df = org_df.dropna().reset_index(drop=True)
            org_df['next_item_id'] = org_df['next_item_id'].astype(np.int64)
            org_df['next_played'] = org_df['next_played'].astype(bool)

            transition_matrix = org_df.groupby(['item_id', 'next_item_id', 'played', 'next_played'])['session_id'].count().reset_index(drop=False).rename(columns={'session_id': 'cnt'}).sort_values(['cnt'], ascending=False)
            transition_matrix['source'] = transition_matrix['item_id'].astype(int)
            transition_matrix['target'] = transition_matrix['next_item_id'].astype(int)
            transition_matrix['source_played'] = transition_matrix['played'].astype(bool)
            transition_matrix['target_played'] = transition_matrix['next_played'].astype(bool)

            transition_matrix = transition_matrix[['source', 'target', 'cnt', 'source_played', 'target_played']]
            transition_matrix = transition_matrix[transition_matrix.cnt>1]
            transition_matrix.index = np.arange(transition_matrix.shape[0])

            soto = transition_matrix[(transition_matrix['source_played'] == True) & (transition_matrix['target_played'] == True)][['source', 'target', 'cnt']]

            # torch.sparse.FloatTensor(torch.LongTensor(soto.source))

        
            soto_csr = csr_matrix((np.array(np.log(soto.cnt)), (np.array(soto.source),np.array(soto.target))), shape=[n_items, n_items])
            idx = (soto_csr.indptr == soto_csr.indptr.max()).nonzero()[0][0]
            soto_csr.data = soto_csr.data / np.repeat(np.add.reduceat(soto_csr.data, soto_csr.indptr[:idx]), np.diff(soto_csr.indptr[:idx+1]))

            soto_csr_t = csr_matrix((np.array(np.log(soto.cnt)), (np.array(soto.target),np.array(soto.source))), shape=[n_items, n_items])
            idx = (soto_csr_t.indptr == soto_csr_t.indptr.max()).nonzero()[0][0]
            soto_csr_t.data = soto_csr_t.data / np.repeat(np.add.reduceat(soto_csr_t.data, soto_csr_t.indptr[:idx]), np.diff(soto_csr_t.indptr[:idx+1]))

            if i == 0:
                transition_dict['source'] = soto_csr
                transition_dict['target'] = soto_csr_t

            elif i == 1:
                transition_dict['ns_source'] = soto_csr
                transition_dict['ns_target'] = soto_csr_t

            elif i == 2:
                transition_dict['s_source'] = soto_csr
                transition_dict['s_target'] = soto_csr_t

        return transition_dict