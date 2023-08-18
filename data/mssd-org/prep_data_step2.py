import os
import argparse
import numpy as np
import pandas as pd
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--data_type', default='5d')
parser.add_argument('--log_num', default='0')
parser.add_argument('--session_key', default='session_id')
args = parser.parse_args()

def main(args):
    task = f'mssd-{args.data_type}-{args.log_num}'
    folder = f'mssd-{args.data_type}-all-{args.log_num}'
    print(f'{task} start!')
    df = pd.read_parquet(os.path.join(args.data_dir, 'mssd-org', 'processed', f'{task}.parquet'))
    save_dir = os.path.join(args.data_dir, folder, 'all')

    train_base = df[df['train'] != 'test'].reset_index(drop=True)
    test_base = df[(df['train'] == 'test')].reset_index(drop=True)

    os.makedirs(save_dir, exist_ok=True)

    df_train = train_base.copy()

    df_train_valid, train_valid_itemmap = assign_itemids(df_train)
    df_train_valid['train'] = 'train_valid'

    # TEST set
    df_test, _ = assign_itemids(test_base, itemmap=train_valid_itemmap)
    df_test['train'] =  'test'

    df_train_only = df_train[df_train['train'] == 'train'].reset_index(drop=True)
    df_train_only, train_only_itemmap = assign_itemids(df_train_only, itemmap=train_valid_itemmap)
    df_train_only['train'] = 'train'

    # VALID set
    df_valid_only = df_train[(df_train['train'] == 'valid')].reset_index(drop=True)
    df_valid_only, _ = assign_itemids(df_valid_only, itemmap=train_only_itemmap)
    df_valid_only['train'] = 'valid'
    
    tmp = pd.concat([df_train_valid, df_train_only, df_valid_only, df_test]).reset_index(drop=True)
    try:
        tmp['Time'] = tmp['new_position']
    except:
        tmp['Time'] = tmp['session_position']    
        
    df_seq = tmp[
        ['session_id', 'session_length', 'train', 'shuffle_session', 'context_session', 'not_skipped', 'ItemId', 'hour_of_day', 'day']
    ].groupby(['session_id', 'session_length', 'train', 'shuffle_session', 'context_session'],
            sort=False).agg(list).reset_index()
    df_seq.index = df_seq['session_id']
    df_seq = df_seq[df_seq['session_length'] > 2]

    sess_map = df_seq.drop(columns='ItemId').reset_index(drop=True)
    
    train_valid_itemmap.to_csv(os.path.join(save_dir, 'item_map.csv'), index=None)
    sess_map.to_csv(os.path.join(save_dir, 'sess_map.csv'), index=None)
    df_seq.to_csv(os.path.join(save_dir, 'seq_new.csv'), index=None)
    print('done!')


def assign_itemids(df, item_key='track_id_clean', session_key='session_id', itemmap=None):
    df_copy = df.copy()
    if itemmap is None:
        df_copy['ItemId'] = df_copy.groupby([item_key]).ngroup() + 1
        itemmap = df_copy[[item_key, 'ItemId']].drop_duplicates(ignore_index=True)
    else:
        df_copy = df_copy.merge(itemmap, on=item_key, how='left')
        df_copy.dropna(inplace=True)
        df_copy['ItemId'] = df_copy['ItemId'].astype(np.int64)
        df_copy['session_length'] = df_copy.groupby([session_key])['session_length'].transform('count')
        df_copy['session_position'] = df_copy.groupby([session_key])['session_position'].cumcount() + 1
        itemmap = df_copy[[item_key, 'ItemId']].drop_duplicates(ignore_index=True)

    return df_copy, itemmap


if __name__ == '__main__':
    main(args)
