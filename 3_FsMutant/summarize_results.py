import os
import torch
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def process_reports(reports, metric='max'):
    all_compound_multi, all_local_single, all_cross_single = [], [], []
    for protein_name in reports['protein_name'].unique():
        compound_multi = reports.loc[reports['protein_name']==protein_name, '[compound_multi]multi_combined']
        local_single = reports.loc[reports['protein_name']==protein_name, '[local_single]single_local']
        cross_single = reports.loc[reports['protein_name']==protein_name, '[cross_single]single_cross']
        if not compound_multi.isna().all():
            all_compound_multi.append(compound_multi.max() if metric=='max' else compound_multi.mean())
        if not local_single.isna().all():
            all_local_single.append(local_single.max() if metric=='max' else local_single.mean())
        if not cross_single.isna().all():
            all_cross_single.append(cross_single.max() if metric=='max' else cross_single.mean())

    return np.mean(all_compound_multi), np.mean(all_local_single), np.mean(all_cross_single)


def results_to_df(results):
    rows = []
    for n_samples, metrics in sorted(results.items()):
        row = {'n_samples': n_samples}
        for metric_name, metric_values in metrics.items():
            for split_name, value in metric_values.items():
                row[f'{metric_name}_{split_name}'] = value
        rows.append(row)

    columns = [
        'n_samples',
        'spearmanr_compound_multi',
        'spearmanr_local_single',
        'spearmanr_cross_single',
        'ndcg_compound_multi',
        'ndcg_local_single',
        'ndcg_cross_single',
    ]
    return pd.DataFrame(rows, columns=columns)


def save_results(results, output_prefix='results-summary'):
    df_results = results_to_df(results)
    if df_results.empty:
        print('No result files found.')
        return df_results

    csv_file = f'{output_prefix}.csv'
    df_results.to_csv(csv_file, index=False, float_format='%.4f')
    print(df_results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nSaved results to {csv_file}')
    return df_results


predictions_dir = './predictions'
results = {}
for n_samples in [20, 40, 80, 160, 320]:
    result_file = os.path.join(predictions_dir, f"meta-transfer/esm2/all/r16_ts{n_samples}_cv5_cosine_mt3.pkl")
    if not os.path.isfile(result_file):
        continue
    reports=torch.load(os.path.join(predictions_dir, f"meta-transfer/esm2/all/r16_ts{n_samples}_cv5_cosine_mt3.pkl"))
    compound_multi_spearmanr, local_single_spearmanr, cross_single_spearmanr = process_reports(reports['spearmanr'], metric='mean')
    compound_multi_ndcg, local_single_ndcg, cross_single_ndcg = process_reports(reports['ndcg'], metric='mean')
    results[n_samples] = {
                        'spearmanr' : {'compound_multi': compound_multi_spearmanr, 'local_single': local_single_spearmanr, 'cross_single': cross_single_spearmanr},
                        'ndcg': {'compound_multi': compound_multi_ndcg, 'local_single': local_single_ndcg, 'cross_single': cross_single_ndcg}    
                    }

df_results = save_results(results)
