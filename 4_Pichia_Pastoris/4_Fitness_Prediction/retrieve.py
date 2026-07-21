import argparse
import torch
import pandas as pd
from src import config
from src.retrieval import Retriever

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, choices=['vectorize', 'retrieve'], default='retrieve',
                        help='compute embedding vectors, or retrieve top-k dms datasets using the saved vectors')
    parser.add_argument('--model', '-md', type=str, choices=config.model_dir.keys(), default='esm1v-1',
                        help='name of the foundation model for vectorizing')
    parser.add_argument('--pooling', '-p', type=str, choices=['average', 'max', 'last'], default='average',
                        help='the method to reduce the model hidden states when vectorizing')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='batch size for vectorizing and retrieving')
    parser.add_argument('--top_k', '-k', type=int, default=10,
                        help='number of similar datasets to retrieve')
    parser.add_argument('--metric', '-mt', type=str, choices=['cosine', 'l2'], default='cosine',
                        help='similarity metric')
    parser.add_argument('--force_cpu', '-cpu', action='store_true',
                        help='use cpu for vectorizing and retrieving even if gpu is available')
    parser.add_argument('--save_postfix', '-sp', type=str, default='',
                        help='a custom string to append to the data path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    auxiliary_data_path = config.auxiliary_data_path.replace('.pkl', f'{args.save_postfix}.pkl')
    target_data_path = config.target_data_path.replace('.pkl', f'{args.save_postfix}.pkl')
    
    if args.mode == 'vectorize':
        data_dict = {'auxiliary': auxiliary_data_path, 'target': target_data_path}
        sequences = {}
        for key, data_path in data_dict.items():
            proteins = torch.load(data_path, weights_only=False)
            if args.model == 'saprot':
                struc_seqs = pd.read_csv(config.struc_seq_path, index_col='protein')
                proteins_dict = {}
                for name, datasets in proteins.items():
                    offset = datasets[0]['offset']
                    sequence = struc_seqs.loc[name, 'struc_sequence']
                    proteins_dict[name] = sequence[offset * 2: (offset + 1022) * 2]
            else:
                proteins_dict = {name: datasets[0]['wild_type'] for name, datasets in proteins.items()}
            sequences[key] = proteins_dict
    else:
        sequences = None
    retriever = Retriever(args)
    retriever(sequences)
    