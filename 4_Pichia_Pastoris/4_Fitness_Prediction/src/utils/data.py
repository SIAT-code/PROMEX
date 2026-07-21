import os
import pandas as pd
import torch
from collections import defaultdict
from itertools import chain
from sklearn.preprocessing import StandardScaler

def make_dir(path):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def trunc_sequence(protein, max_len):
    L = len(protein['wild_type'])
    if L <= max_len:
        protein['offset'] = 0
        return
    
    df = protein['df']
    positions = list(chain(*df['positions']))
    max_pos, min_pos = max(positions), min(positions)
    gap = max_pos - min_pos + 1
    
    if max_pos < max_len:
        protein['wild_type'] = protein['wild_type'][:max_len]
        protein['offset'] = 0
        return
    
    if gap <= max_len:
        window_l = max(min_pos - (max_len - gap) // 2, 0)
        window_r = min(max_pos + (max_len - gap) // 2, L - 1)
        seq_lr = protein['wild_type'][window_l: window_l + max_len]
        seq_rl = protein['wild_type'][window_r - max_len + 1: window_r + 1]
        
        if len(seq_lr) > len(seq_rl):
            protein['wild_type'] = seq_lr
            left, right = window_l, window_l + max_len
        else:
            protein['wild_type'] = seq_rl
            left, right = window_r - max_len + 1, window_r + 1
    else:
        n = 0
        left, right = min_pos, max_len
        window_l, window_r = min_pos, max_len
        while window_r < L:
            window_n = df['positions'].apply(
                lambda positions: all(window_l <= pos < window_r for pos in positions)).sum()
            if window_n > n:
                left, right = window_l, window_r
                n = window_n
            window_l += 1
            window_r += 1
        
        if right - left + 1 < max_len:
            left = right - max_len
        protein['wild_type'] = protein['wild_type'][left:right]
    
    df_bool = df.apply(lambda row: all(left <= pos < right for pos in row['positions']), axis=1)
    df = df.loc[df_bool].copy()
    df.loc[:, 'positions'] = df['positions'].apply(lambda positions: tuple(pos - left for pos in positions))
    protein['df'] = df
    protein['offset'] = left
    return

def process_dms(file_path, shuffle=True, max_len=1022, wild_type=None):
    df = pd.read_csv(file_path, index_col='mutant')
    if shuffle:
        df = df.sample(frac=1)
   
    new_df, n_sites = defaultdict(list), set()
    for mutant, row in df.iterrows():
        wt_aas, mt_aas, positions = '', '', []
        for site in mutant.split(':'): # handle multi-site mutants
            wt_aa, position, mt_aa = site[0], int(site[1:-1]) - 1, site[-1]
            if wild_type is None:
                assert row['mutated_sequence'][position] == mt_aa
            else:
                assert wild_type[position] == wt_aa
            wt_aas += wt_aa
            mt_aas += mt_aa
            positions.append(position)
        
        new_df['wt_aas'].append(wt_aas)
        new_df['mt_aas'].append(mt_aas)
        new_df['positions'].append(tuple(positions))
        n_sites.add(len(positions))
    
    # new_df = pd.concat([pd.DataFrame(new_df, index=df.index),
    #                     df[['DMS_score', 'DMS_score_bin']]], axis=1)  # mark No DMS_score_bin on VenusMutHub 
    new_df = pd.concat([pd.DataFrame(new_df, index=df.index),
                        df[['DMS_score']]], axis=1)
    if wild_type is None:
        wild_type = list(row['mutated_sequence'])
        for wt_aa, position in zip(wt_aas, positions): # recover wild type sequence
            wild_type[position] = wt_aa
        wild_type = ''.join(wild_type)
    protein = dict(wild_type=wild_type, df=new_df)
    trunc_sequence(protein, max_len)
    protein['n_sites'] = sorted(n_sites)
    protein['name'] = os.path.basename(file_path).split('.')[0]
    return protein

def merge_files(data_dir, shuffle=True, max_len=1022, save_path=None):
    file_names = os.listdir(data_dir)
    proteins = defaultdict(list)
    for file_name in file_names:
        if 'indels' in file_name:
            continue
        protein = process_dms(f'{data_dir}/{file_name}/{file_name}.csv', shuffle, max_len)  # mark
        name = '_'.join(file_name.split('_')[:2])
        proteins[name].append(protein)
    
    if save_path is not None:
        make_dir(save_path)
        torch.save(proteins, save_path)
    return proteins

# Add mark
def new_merge_files(raw_data_dir, shuffle=True, max_len=1022, save_path=None):
    file_names = os.listdir(raw_data_dir)
    proteins = defaultdict(list)
    for file_name in file_names:
        if 'indels' in file_name:
            continue
        protein = process_dms(f'{raw_data_dir}/{file_name}', shuffle, max_len)  # mark
        name = file_name.replace('.csv', '')
        proteins[name].append(protein)
            
    if save_path is not None:
        make_dir(save_path)
        torch.save(proteins, save_path)
    return proteins

def normalize(df):  # mark
    scores = df['DMS_score'].to_numpy()[:,None]
    scaler = StandardScaler()
    df['DMS_score'] = scaler.transform(scores).squeeze(1)

def split_data(protein, train_size=0.8, shuffle=False, n_sites=None, neg_train=False,
               scale=False, train_ids=None):
    df = protein['df']
    train, test = protein.copy(), protein.copy()
    
    if train_ids is not None:
        train['df'] = df.loc[train_ids]
        test['df'] = df.loc[df.index.difference(train_ids, sort=False)]
    else:
        N = len(df)
        if train_size < 1:
            train_size = int(N * train_size)
        if shuffle:
            df = df.sample(frac=1)
        if n_sites is not None:
            n_sites = set(n_sites)
    
        df_bool = df.apply(lambda row: (not n_sites or len(row['positions']) in n_sites) and \
                                       (not neg_train or row['DMS_score_bin'] == 0), axis=1)
        train['df'] = df.loc[df_bool].iloc[:train_size]
        test['df'] = df.loc[df.index.difference(train['df'].index, sort=False)]
    
    if scale:
        normalize(train['df'])  # mark
        normalize(test['df'])  # mark
    return train, test


# Add mark
def split_train_test_data(protein, train_size, config, scale=False):
    df = protein['df']
    train_test_dict = {}
    for split in range(1, 6):
        train, compound_multi, local_single, cross_single = protein.copy(), protein.copy(), protein.copy(), protein.copy()
        splits_dir = f'{config.raw_data_dir}/{protein["name"]}/splits/{train_size}_{split}'
        if not os.path.isdir(splits_dir):
            continue
        train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'), index_col='mutant')
        train['df'] = df.loc[train_df.index]
        
        if os.path.isfile(os.path.join(splits_dir, 'compound_multi.csv')):
            compound_multi_df = pd.read_csv(os.path.join(splits_dir, 'compound_multi.csv'), index_col='mutant')
            compound_multi['df'] = df.loc[compound_multi_df.index]
        else:
            compound_multi['df'] = None
            
        if os.path.isfile(os.path.join(splits_dir, 'local_single.csv')): 
            local_single_df = pd.read_csv(os.path.join(splits_dir, 'local_single.csv'), index_col='mutant')
            local_single['df'] = df.loc[local_single_df.index]
        else:
            local_single['df'] = None
            
        if os.path.isfile(os.path.join(splits_dir, 'cross_single.csv')):
            cross_single_df = pd.read_csv(os.path.join(splits_dir, 'cross_single.csv'), index_col='mutant')
            cross_single['df'] = df.loc[cross_single_df.index]
        else:
            cross_single['df'] = None
            
        if scale:
            normalize(train['df'])
            if compound_multi['df'] is not None: normalize(compound_multi['df']) 
            if local_single['df'] is not None: normalize(local_single['df'])
            if cross_single['df'] is not None: normalize(cross_single['df'])

        train_test_dict[f'{train_size}_{split}'] = {'train': train, 'compound_multi': compound_multi, 'local_single': local_single, 'cross_single': cross_single}

    return train_test_dict