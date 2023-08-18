import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--org_path', default='./data/mssd/', help='Original Raw Log Path')
parser.add_argument('--save_path', default='./data/mssd/processed', help='Save Path')
parser.add_argument('--data_type', default='5d', help='[3d, 5d, 7d]')
parser.add_argument('--log_num', default='1', type=str)
parser.add_argument('--n_cores', default=5, type=int)
args = parser.parse_args()


def main(args):
    print(f'mssd-{args.data_type}-{args.log_num} start!')
    df = pd.read_parquet(osp.join(args.org_path, f'log_{args.log_num}_aug.parquet'))
    if args.data_type == '1d':
        train_days = [20180801]
        valid_days = [20180802]
        test_days = [20180803]
    elif args.data_type == '3d':
        train_days = [20180801, 20180802, 20180803]
        valid_days = [20180804]
        test_days = [20180805]
    elif args.data_type == '5d':
        train_days = [20180806, 20180807, 20180808, 20180809, 20180810]
        valid_days = [20180811]
        test_days = [20180812]
    elif args.data_type == '7d':
        train_days = [201808013, 20180814, 20180815, 20180816, 20180817, 20180818, 20180819]
        valid_days = [20180820]
        test_days = [20180821]
    else:
        raise ValueError
    inscope_days = train_days + valid_days + test_days
    train_days = [f'log_{args.log_num}_{d}' for d in train_days]
    valid_days = [f'log_{args.log_num}_{d}' for d in valid_days]
    test_days = [f'log_{args.log_num}_{d}' for d in test_days]
    inscope_days = [f'log_{args.log_num}_{d}' for d in inscope_days]

    df = df[df['file'].isin(inscope_days)].reset_index(drop=True)

    shuffle_info = df.groupby(['session_id', 'hist_user_behavior_is_shuffle', 'session_length'])['session_position'].count().reset_index()
    shuffle_info['shuffle'] = np.where(shuffle_info['session_length'] == shuffle_info['session_position'], shuffle_info['hist_user_behavior_is_shuffle'], 'hybrid')
    shuffle_info['shuffle'] = np.where(shuffle_info['shuffle'] == 'True', 'shuffle',
                                    np.where(shuffle_info['shuffle'] == 'False', 'nonshuffle', 'hybrid'))
    shuffle_info = shuffle_info[['session_id', 'shuffle']].drop_duplicates(ignore_index=True)

    context_info = df.groupby(['session_id', 'context_type', 'session_length'])['session_position'].count().reset_index()
    context_info['context'] = np.where(context_info['session_length'] == context_info['session_position'], context_info['context_type'], 'hybrid')
    context_info = context_info[['session_id', 'context']].drop_duplicates(ignore_index=True)

    session_info = pd.merge(shuffle_info, context_info, on=['session_id'])

    # session_info.to_csv(osp.join(new_processed_path, 'session_info.csv'), index=None)


    df = filter_k_core(df, args.n_cores, 'session_id', 'track_id_clean')
    df = df.sort_values(['session_id', 'session_position'], ignore_index=True)

    df = df[['session_id', 'session_position', 'session_length',
            'track_id_clean', 'hour_of_day', 'date',
            'skip_1', 'skip_2', 'skip_3', 'not_skipped',
            'hist_user_behavior_is_shuffle', 'context_type', 'file']]
    df['timestamp'] = df['date'] + ' ' + df['hour_of_day'].astype(str) + ':00:00'
    df['day'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df.rename(columns={'hist_user_behavior_is_shuffle':'shuffle_track',
                    'context_type': 'context_track'}, inplace=True)
    df['shuffle_track'] = np.where(df['shuffle_track'], 'shuffle', 'nonshuffle')
    df.sort_values(['session_id', 'date', 'hour_of_day', 'session_position'], inplace=True, ignore_index=True)
    df.drop(columns=['date'], inplace=True)
    df['session_length'] = df.groupby(['session_id'])['session_length'].transform('count')
    df['session_position'] = df.groupby(['session_id'])['session_position'].cumcount() + 1

    df = df.merge(session_info.rename(columns={'shuffle':'shuffle_session', 'context':'context_session'}), on='session_id', how='left')

    df['train'] = np.where(df['file'].isin(test_days), 'test', 
                           np.where(df['file'].isin(valid_days), 'valid', 'train'))

    # save_path = osp.join(args.save_path, f'mssd-{args.data_type}-{args.log_num}')
    os.makedirs(args.save_path, exist_ok=True)
    df.to_parquet(osp.join(args.save_path, f'mssd-{args.data_type}-{args.log_num}.parquet'),
                  compression=None, index=None)
    # print(df.head())
    print('done!')


# From: https://github.com/microsoft/recommenders
def _get_column_name(name, col_user, col_item):
    if name == "user":
        return col_user
    elif name == "item":
        return col_item
    else:
        raise ValueError("name should be either 'user' or 'item'.")

def min_rating_filter_pandas(
    data,
    min_rating=1,
    filter_by="user",
    col_user='user',
    col_item='item',
):
    """Filter rating DataFrame for each user with minimum rating.
    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.
    Args:
        data (pandas.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating,
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.
    Returns:
        pandas.DataFrame: DataFrame with at least columns of user and item that has been filtered by the given specifications.
    """
    split_by_column = _get_column_name(filter_by, col_user, col_item)

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    return data.groupby(split_by_column).filter(lambda x: len(x) >= min_rating)


def filter_k_core(data, core_num=0, col_user="user", col_item="item"):
    """Filter rating dataframe for minimum number of users and items by
    repeatedly applying min_rating_filter until the condition is satisfied.
    """
    num_users, num_items = len(data[col_user].unique()), len(data[col_item].unique())
    print(f"Original: {num_users} users and {num_items} items")
    df_inp = data.copy()
    df_inp[f'new_{col_user}'] = df_inp.groupby(col_user).ngroup(ascending=True)
    df_inp[f'new_{col_item}'] = df_inp.groupby(col_item).ngroup(ascending=True)

    if core_num > 0:
        while True:
            df_inp = min_rating_filter_pandas(
                df_inp, min_rating=core_num, filter_by='item', col_user=f'new_{col_user}', col_item=f'new_{col_item}'
            )
            df_inp = min_rating_filter_pandas(
                df_inp, min_rating=core_num, filter_by='user', col_user=f'new_{col_user}', col_item=f'new_{col_item}'
            )
            count_u = df_inp.groupby(f'new_{col_user}')[col_item].count()
            count_i = df_inp.groupby(col_item)[f'new_{col_user}'].count()
            if (
                len(count_i[count_i < core_num]) == 0
                and len(count_u[count_u < core_num]) == 0
            ):
                break
    
    df_inp = df_inp.sort_values(by=[f'new_{col_user}'])
    num_users = len(df_inp[f'new_{col_user}'].unique())
    num_items = len(df_inp[col_item].unique())
    print(f"Final: {num_users} users and {num_items} items")

    df_inp = df_inp.drop(columns=[f'new_{col_user}', f'new_{col_item}'])

    return df_inp


if __name__ == '__main__':
    main(args)