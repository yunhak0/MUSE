import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='mssd-3d-all',
                        help='[mssd-1d-all, mssd-3d-all, mssd-5d-all, mssd-7d-all]')
    parser.add_argument('--embedder', default='MUSE',
                        help="Model name")
    parser.add_argument('--device', default='7', type=str,
                        help='GPU')
    parser.add_argument('--topk', default='[1,5,10,20]',
                        help="Top-K for performance metrics")
    parser.add_argument('--seed', default=0, type=int,
                        help='Start log file number: 0~9')
    parser.add_argument('--n_runs', default=1, type=int,
                        help='No. of seed to run')

    parser.add_argument('--n_epochs', default=30, type=int,
                        help='No. of training epoch')
    parser.add_argument('--val_epoch', default=1, type=int,
                        help='No. training epoch for validation')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Training batch size')
    parser.add_argument('--embedding_dim', default=100, type=int,
                        help='Item embedding size')
    parser.add_argument('--hidden_size', default=100, type=int,
                         help='Hidden size of model')
    parser.add_argument('--n_layers', default=1, type=int,
                        help='Hidden layer of model')
    parser.add_argument('--dropout0', default=0.0, type=float,
                        help='Dropout rate for embedding')
    parser.add_argument('--dropout1', default=0.0, type=float,
                        help='Dropout rate for model')
    parser.add_argument('--final_act', default='relu', type=str,
                        help='[tanh, relu, softmax, softmax_logit, elu-{:.d}, leaky-{:.d}]')
    parser.add_argument('--maxlen', default=19, type=int,
                        help='Max length of input sequence/session')

    parser.add_argument('--loss_type', default='CE', type=str,
                        help='Loss for Recommendation - CE / sampledCE / BPR-max / BPR / TOP1-max / TOP1')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--decay', default=0.0, type=float,
                        help="weight decay or l2 (fixed as 0.0)")
    parser.add_argument('--patience', default=5, type=int,
                        help="Tolerance for early stop")

    # For output
    parser.add_argument('--save_ckpt', action='store_true', default=True,
                        help='Save checkpoint of best model')
    parser.add_argument('--inference', action='store_true',
                        help='For inference only')
    
    # For MUSE
    parser.add_argument('--prob', default=1.0, type=float,
                        help='Controls how much candidates can be inserted in transition-based augmentation (fixed as 1.0)')
    parser.add_argument('--warm_up_epoch', default=1, type=int,
                        help='No. warm up epoch for Similarity-based Matching')
    parser.add_argument('--repr_loss_type', default='MSE', type=str,
                        help='Loss for self-supervised learning')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='Temperature for contrastive loss (InfoNCE, SupCon) (fixed as 1.0)')
    parser.add_argument("--inv-coeff", type=float, default=1.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--var-coeff", type=float, default=1.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=10.0,
                        help='Covariance regularization loss coefficient')
    parser.add_argument(
        "--num_matches",
        type=int,
        nargs="+",
        default=[5, 5], # [3, 3]
        help="Number of spatial matches in a feature map",
    )
    parser.add_argument("--alpha", type=float, default=0.2,
                        help='Loss balancing coefficient for matching loss (range: 0 ~ 1)')
    parser.add_argument('--reorder_r', default=0.5, type=float,
                        help='Reordering augmentation ratio within a sequence/session')

    args = parser.parse_args()
    return args
