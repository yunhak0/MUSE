import os
import os.path as osp
import re
import pandas as pd
from tqdm import tqdm


raw_path = './raw/'
processed_path = './'

for log in range(10):
    files = [f for f in os.listdir(raw_path) if osp.isfile(osp.join(raw_path, f)) and re.match(f'log_{log}_201808', f)]
    files.sort()

    df = pd.DataFrame()
    for f in tqdm(files, desc=f'{log}/9'):
        tmp = pd.read_csv(osp.join(raw_path, f))
        tmp['file'] = f[:14]
        non_prem_sess = tmp[~tmp.premium]['session_id'].unique()
        tmp = tmp[~tmp.session_id.isin(non_prem_sess)].reset_index(drop=True)
        df = pd.concat([df, tmp])
    df.to_parquet(osp.join(processed_path, f'log_{log}_aug.parquet'), compression=None, index=None)
