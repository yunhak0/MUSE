from utils.argument import parse_args
from utils.utils import set_random_seeds, setup_logger, config2string
from datetime import datetime
import numpy as np
import os
import torch
import yaml

def main():
    args = parse_args()
    torch.set_num_threads(4)
    model_name = args.embedder.lower()

    ns_recall_list = []
    ns_mrr_list = []
    ns_ndcg_list = []

    s_recall_list = []
    s_mrr_list = []
    s_ndcg_list = []

    all_recall_list = []
    all_mrr_list = []
    all_ndcg_list = []

    if 'muse' in args.embedder.lower():
        log_info_path = f'./logs_n_runs_{args.n_runs}/{args.dataset}'  # Logging

    os.makedirs(log_info_path, exist_ok=True)
    logger = setup_logger(name=args.embedder, log_dir=log_info_path, filename=f'./{args.embedder}.txt')
    logger_all = setup_logger(name=args.embedder, log_dir=log_info_path, filename=f'all_logs.txt')
    config_str = config2string(args)
    inference = 'all'
    
    for seed in range(args.seed, args.seed+args.n_runs):
        print(f'Dataset: {args.dataset}, Inference: {inference}, Seed: {seed}, Model: {model_name}')
        set_random_seeds(seed)
        args.seed = seed
    
        if 'best' in model_name:
            with open(f'config/{args.embedder[:-5]}_mssd.yaml', 'r') as f:
                hyperparams = yaml.safe_load(f)
                for k, v in hyperparams.items():
                    setattr(args, k, v)
            model_name = model_name[:-5]

        if model_name == 'muse':
            from models.MUSE import MUSE_Trainer as trainer
        else:
            raise NotImplementedError

        embedder = trainer(args)
        
        if args.inference:
            with open(f'config/{args.embedder}_mssd.yaml', 'r') as f:
                hyperparams = yaml.safe_load(f)
                for k, v in hyperparams.items():
                    setattr(args, k, v)
            embedder.load_dataset()
            embedder.load_model()
            embedder.model.load_state_dict(torch.load(f'{embedder.ckpt_path}/{args.embedder}_final_model.pt'))
        else:
            overall_performance = embedder.fit()

        _ns_recall = overall_performance['nonshuffle']['recall']
        _ns_mrr = overall_performance['nonshuffle']['mrr']
        _ns_ndcg = overall_performance['nonshuffle']['ndcg']

        ns_recall_list.append(_ns_recall)
        ns_mrr_list.append(_ns_mrr)
        ns_ndcg_list.append(_ns_ndcg)

        _s_recall = overall_performance['shuffle']['recall']
        _s_mrr = overall_performance['shuffle']['mrr']
        _s_ndcg = overall_performance['shuffle']['ndcg']

        s_recall_list.append(_s_recall)
        s_mrr_list.append(_s_mrr)
        s_ndcg_list.append(_s_ndcg)

        _all_recall = overall_performance['all']['recall']
        _all_mrr = overall_performance['all']['mrr']
        _all_ndcg = overall_performance['all']['ndcg']

        all_recall_list.append(_all_recall)
        all_mrr_list.append(_all_mrr)
        all_ndcg_list.append(_all_ndcg)

        st = f'{datetime.now()}\n'
        st += f'[Config] {config_str}\n'
        st += f'-------------------- Top K: {args.topk} --------------------\n'
        st += f'***** [Seed {args.seed}] Test Results (Non-Shuffle) *****\n'
        st += f'Recall: {_ns_recall}\n'
        st += f'MRR: {_ns_mrr}\n'
        st += f'NDCG: {_ns_ndcg}\n'
        
        st += f'***** [Seed {args.seed}] Test Results (Shuffle) *****\n'
        st += f'Recall: {_s_recall}\n'
        st += f'MRR: {_s_mrr}\n'
        st += f'NDCG: {_s_ndcg}\n'

        st += f'***** [Seed {args.seed}] Test Results (All) *****\n'
        st += f'Recall: {_all_recall}\n'
        st += f'MRR: {_all_mrr}\n'
        st += f'NDCG: {_all_ndcg}\n'

        st += f'================================='

        logger.info(st)
        logger_all.info(st)

    ns_recall = np.stack(ns_recall_list)
    ns_mrr = np.stack(ns_mrr_list)
    ns_ndcg = np.stack(ns_ndcg_list)

    ns_recall_mean = np.mean(ns_recall, 0)
    ns_recall_std = np.std(ns_recall, 0)
    ns_mrr_mean = np.mean(ns_mrr, 0)
    ns_mrr_std = np.std(ns_mrr, 0)
    ns_ndcg_mean = np.mean(ns_ndcg, 0)
    ns_ndcg_std = np.std(ns_ndcg, 0)

    s_recall = np.stack(s_recall_list)
    s_mrr = np.stack(s_mrr_list)
    s_ndcg = np.stack(s_ndcg_list)

    s_recall_mean = np.mean(s_recall, 0)
    s_recall_std = np.std(s_recall, 0)
    s_mrr_mean = np.mean(s_mrr, 0)
    s_mrr_std = np.std(s_mrr, 0)
    s_ndcg_mean = np.mean(s_ndcg, 0)
    s_ndcg_std = np.std(s_ndcg, 0)

    all_recall = np.stack(all_recall_list)
    all_mrr = np.stack(all_mrr_list)
    all_ndcg = np.stack(all_ndcg_list)

    all_recall_mean = np.mean(all_recall, 0)
    all_recall_std = np.std(all_recall, 0)
    all_mrr_mean = np.mean(all_mrr, 0)
    all_mrr_std = np.std(all_mrr, 0)
    all_ndcg_mean = np.mean(all_ndcg, 0)
    all_ndcg_std = np.std(all_ndcg, 0)

    st = f'{datetime.now()}\n'
    st += f'[Config] {config_str}\n'
    st += f'-------------------- Top K: {args.topk} --------------------\n'
    st += f'***** [Final] Test Results (Non-Shuffle) *****\n'
    st += f'Recall: {ns_recall.tolist()}\n'
    st += '{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}\n'.format(ns_recall_mean[0], ns_recall_std[0], ns_recall_mean[1], ns_recall_std[1], ns_recall_mean[2], ns_recall_std[2], ns_recall_mean[3], ns_recall_std[3])
    st += f'MRR: {ns_mrr.tolist()}\n'
    st += '{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}\n'.format(ns_mrr_mean[0], ns_mrr_std[0], ns_mrr_mean[1], ns_mrr_std[1], ns_mrr_mean[2], ns_mrr_std[2], ns_mrr_mean[3], ns_mrr_std[3])
    st += f'NDCG: {ns_ndcg.tolist()}\n'
    st += '{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}\n'.format(ns_ndcg_mean[0], ns_ndcg_std[0], ns_ndcg_mean[1], ns_ndcg_std[1], ns_ndcg_mean[2], ns_ndcg_std[2], ns_ndcg_mean[3], ns_ndcg_std[3])
    st += '\n'
    
    st += f'***** [Final] Test Results (Shuffle) *****\n'
    st += f'Recall: {s_recall.tolist()}\n'
    st += '{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}\n'.format(s_recall_mean[0], s_recall_std[0], s_recall_mean[1], s_recall_std[1], s_recall_mean[2], s_recall_std[2], s_recall_mean[3], s_recall_std[3])
    st += f'MRR: {s_mrr.tolist()}\n'
    st += '{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}\n'.format(s_mrr_mean[0], s_mrr_std[0], s_mrr_mean[1], s_mrr_std[1], s_mrr_mean[2], s_mrr_std[2], s_mrr_mean[3], s_mrr_std[3])
    st += f'NDCG: {s_ndcg.tolist()}\n'
    st += '{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}\n'.format(s_ndcg_mean[0], s_ndcg_std[0], s_ndcg_mean[1], s_ndcg_std[1], s_ndcg_mean[2], s_ndcg_std[2], s_ndcg_mean[3], s_ndcg_std[3])
    st += '\n'

    st += f'***** [Final] Test Results (All) *****\n'
    st += f'Recall: {all_recall.tolist()}\n'
    st += '{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}\n'.format(all_recall_mean[0], all_recall_std[0], all_recall_mean[1], all_recall_std[1], all_recall_mean[2], all_recall_std[2], all_recall_mean[3], all_recall_std[3])
    st += f'MRR: {all_mrr.tolist()}\n'
    st += '{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}\n'.format(all_mrr_mean[0], all_mrr_std[0], all_mrr_mean[1], all_mrr_std[1], all_mrr_mean[2], all_mrr_std[2], all_mrr_mean[3], all_mrr_std[3])
    st += f'NDCG: {all_ndcg.tolist()}\n'
    st += '{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}\n'.format(all_ndcg_mean[0], all_ndcg_std[0], all_ndcg_mean[1], all_ndcg_std[1], all_ndcg_mean[2], all_ndcg_std[2], all_ndcg_mean[3], all_ndcg_std[3])
    st += '\n'

    st += f'================================='

    logger.info(st)
    logger_all.info(st)

if __name__ == '__main__':
    main()
