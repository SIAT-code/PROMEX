import torch
from torch.utils.data import Dataset
import json
import random
import numpy as np

from ..data_interface import register_dataset
from transformers import EsmTokenizer
from ..lmdb_dataset import *
from ..lmdb_dataset import *
from utils.others import setup_seed


@register_dataset
class SaprotRegressionDataset(LMDBDataset):
	def __init__(self,
	             tokenizer: str,
	             max_length: int = 1024,
	             min_clip: [float, float] = None,
	             mix_max_norm: [float, float] = None,
				 mask_struc_ratio: float = None,
	             plddt_threshold: float = None,
	             **kwargs):
		"""
		
		Args:
			tokenizer: ESM tokenizer

			max_length: Maximum length of the sequence

			min_clip: [given_value, clip_value]
					  Set the fitness value to a fixed value if it is less than a given value
			
			mix_max_norm: [min_norm, max_norm]
						  Normalize the fitness value to [0, 1] by min-max normalization

			mask_struc_ratio: Ratio of masked structure tokens, replace structure tokens with "#"
			
			plddt_threshold: If not None, mask structure tokens with pLDDT < threshold

			**kwargs:
		"""
		
		super().__init__(**kwargs)
		self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
		self.max_length = max_length
		self.min_clip = min_clip
		self.mix_max_norm = mix_max_norm
		self.mask_struc_ratio = mask_struc_ratio
		self.plddt_threshold = plddt_threshold
		self.indices = None
		self.length = None
        
	def __getitem__(self, index):
		if self.indices is not None and self.length is not None:  # mark
			entry = json.loads(self._get(self.indices[index]))  # mark
		else:
			entry = json.loads(self._get(index))
		seq = entry['seq']

		# Mask structure tokens
		if self.mask_struc_ratio is not None:
			tokens = self.tokenizer.tokenize(seq)
			mask_candi = [i for i, t in enumerate(tokens) if t[-1] != "#"]

			# Randomly shuffle the mask candidates and set seed to ensure mask is consistent
			setup_seed(20000812)
			random.shuffle(mask_candi)

			# Mask first n structure tokens
			mask_num = int(len(mask_candi) * self.mask_struc_ratio)
			for i in range(mask_num):
				idx = mask_candi[i]
				tokens[idx] = tokens[idx][:-1] + "#"

			seq = "".join(tokens)
		
		# Mask structure tokens with pLDDT < threshold
		if self.plddt_threshold is not None:
			plddt = entry["plddt"]
			tokens = self.tokenizer.tokenize(seq)
			seq = ""
			for token, score in zip(tokens, plddt):
				if score < self.plddt_threshold:
					seq += token[:-1] + "#"
				else:
					seq += token

		tokens = self.tokenizer.tokenize(seq)[:self.max_length]
		seq = " ".join(tokens)
	
		if self.min_clip is not None:
			given_min, clip_value = self.min_clip
			if entry['label'] < given_min:
				entry['label'] = clip_value
		
		if self.mix_max_norm is not None:
			min_norm, max_norm = self.mix_max_norm
			entry['label'] = (entry['label'] - min_norm) / (max_norm - min_norm)
				
		label = entry['label']
		return seq, label
	
	def __len__(self):
		return self.length if self.indices is not None and self.length is not None else int(self._get("length"))
	
	def collate_fn(self, batch):
		seqs, labels = tuple(zip(*batch))
		labels = torch.tensor(labels)
		labels = {"labels": labels}
		
		encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
		inputs = {"inputs": encoder_info}
		
		return inputs, labels

	def _meta_dataloader(self, stage):
		self.dataloader_kwargs["shuffle"] = True if stage == "train" else False
		lmdb_path = getattr(self, f"{stage}_lmdb")
		dataset = copy.copy(self)
		dataset._init_lmdb(lmdb_path)
		setattr(dataset, "stage", stage)
        
		dataset_length = int(dataset._get("length"))
 
		adapt_dataset = copy.copy(dataset)
		eval_dataset = copy.copy(dataset)
        
		meta_dataset = MetaSaprotRegressionDataset(adapt_dataset, eval_dataset, dataset_length, self.dataloader_kwargs)
		meta_dataloader = DataLoader(meta_dataset, batch_size=self.dataloader_kwargs['meta_batch_size'], collate_fn=meta_dataset.collate)
        
		return meta_dataloader

	def _dataloader(self, stage):
		self.dataloader_kwargs["shuffle"] = True if stage == "train" else False
		lmdb_path = getattr(self, f"{stage}_lmdb")
		dataset = copy.copy(self)
		dataset._init_lmdb(lmdb_path)
		setattr(dataset, "stage", stage)

		return DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=self.dataloader_kwargs['batch_size'], num_workers=self.dataloader_kwargs['num_workers'])

	def train_dataloader(self):
		return self._meta_dataloader("train")

	def test_dataloader(self):
		return self._dataloader("test")
    
	def val_dataloader(self):
		return self._dataloader("valid")



class MetaSaprotRegressionDataset(Dataset):
    def __init__(self,
                 adapt_dataset,
                 eval_dataset,
                 dataset_length,
                 dataloader_kwargs: dict = None):
        
        self.adapt_dataset = adapt_dataset
        self.eval_dataset = eval_dataset
        self.indices = list(range(dataset_length))
        self.weights = np.ones(len(self.indices))
        self.dataloader_kwargs = dataloader_kwargs
        self.iters_count = 0
        self.init_indices()
        ###################################### adapt iters #############################################
        # self.dataloader_kwargs['iters'] = len(self.indices) // (2 * self.dataloader_kwargs['adapt_batch_size'] * self.dataloader_kwargs['adapt_steps'])
        ###############################################################################################
        self.support_iters = []
        self.query_iters = []
        for _ in range(self.dataloader_kwargs['iters']):
            support_iter = DataLoader(self.adapt_dataset, collate_fn=self.adapt_dataset.collate_fn, batch_size=self.dataloader_kwargs['adapt_batch_size'], num_workers=self.dataloader_kwargs['num_workers'])
            query_iter = DataLoader(self.eval_dataset, collate_fn=self.eval_dataset.collate_fn, batch_size=self.dataloader_kwargs['eval_batch_size'], num_workers=self.dataloader_kwargs['num_workers'])

            self.support_iters.append(support_iter)
            self.query_iters.append(query_iter)

    def init_indices(self):
        if self.iters_count == 0:
            print('Initialize Indices...')
            random.shuffle(self.indices)
            self.weights = np.ones(len(self.indices))
        split_idx = len(self.indices) // 2
        part1, part2 = self.indices[:split_idx], self.indices[split_idx:]
        weights1, weights2 = self.weights[:split_idx], self.weights[split_idx:]
        
        adapt_size = self.dataloader_kwargs['adapt_batch_size'] * self.dataloader_kwargs['adapt_steps']
        eval_size = self.dataloader_kwargs['eval_batch_size']

        #################### iter adapt dataset and eval dataset mark ###########################################
        self.adapt_dataset.indices = part1[self.iters_count * adapt_size: (self.iters_count + 1) * adapt_size]
        self.eval_dataset.indices = part2[self.iters_count * eval_size: (self.iters_count + 1) * eval_size]
        ########################################origin mark #####################################################
        # self.adapt_dataset.indices = np.random.choice(part1, size=adapt_size, replace=False, p=weights1 / weights1.sum())
        # self.eval_dataset.indices = np.random.choice(part2, size=eval_size, replace=False, p=weights2 / weights2.sum())
        #########################################################################################################
        self.adapt_dataset.length = adapt_size
        self.eval_dataset.length = eval_size
        
    def __len__(self):
        return self.dataloader_kwargs['iters']
    
    def __getitem__(self, idx):
        self.init_indices()
        self.iters_count += 1
        if self.iters_count % self.dataloader_kwargs['iters'] == 0:
            self.iters_count = 0
        
        adapt_batch = [batch for batch in self.support_iters[idx]]
        eval_batch = next(iter(self.query_iters[idx]))
        return adapt_batch, eval_batch

    def collate(self, raw_batch):
        adapt_batches, eval_batches = zip(*raw_batch)
        return dict(adapt_batches=adapt_batches,
                    eval_batches=eval_batches)