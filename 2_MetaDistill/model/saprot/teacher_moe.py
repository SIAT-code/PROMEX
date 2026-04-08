import os
import time
import numpy as np
from datetime import datetime
from typing import List
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from transformers import EsmConfig, EsmTokenizer, EsmForMaskedLM, EsmForSequenceClassification, EsmForTokenClassification

import json
import lmdb
import pandas as pd
from tqdm import tqdm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.attend = nn.MultiheadAttention(dim, heads, dropout=dropout)

    def forward(self, x):
        q = x.permute((1,0,2))
        k = x.permute((1,0,2))
        v = x.permute((1,0,2))
        out, _ = self.attend(q, k, v)
        out = out.permute((1,0,2))
        return out


@dataclass
class TransformerConfig:
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    dropout: float


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class Expert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        return self.fc(x)
    
class MixtureFFNDown(nn.Module):
    def __init__(self, group_num_experts:List[int], transformer_config:TransformerConfig, in_features:int, out_features:int, num_experts_per_token, original_ffndown_weight, original_ffndown_bias):
        super().__init__()
        self.group_num_experts = group_num_experts
        self.top_k = num_experts_per_token # experts number involved in forward reasoning, in mixtral, that is 2
        self.original_weight = original_ffndown_weight
        self.original_bias = original_ffndown_bias
        
        self.gate_groups = nn.ModuleList([
            nn.Linear(in_features, num_experts, bias=False) for num_experts in self.group_num_experts
        ])
        self.expert_groups = nn.ModuleList([
            nn.ModuleList([Expert(in_features, out_features) for _ in range(num_experts)]) for num_experts in self.group_num_experts  # n experts, in mixtral, that is 8
        ])
        ##########################################################################
        #------------------------------ Transformer -----------------------------#
        # self.experts_aggregator = Transformer(**transformer_config)
        ##########################################################################
        #-------------------------------- Gating --------------------------------#
        self.experts_aggregator = nn.Linear(out_features, len(group_num_experts))

    def forward(self, x):
        batch_size, sequence_length, hidden_dim = x.shape
        result_dim = int(hidden_dim / 4)
        hidden_states = x.view(-1, hidden_dim)
        group_final_hidden_states = []
        for group_idx in range(len(self.group_num_experts)):
            router_logits = self.gate_groups[group_idx](hidden_states)
            routing_weights = torch.softmax(router_logits, dim=-1)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # change the weights back to hidden states dtype
            routing_weights = routing_weights.to(hidden_states.dtype)
            final_hidden_states = torch.zeros((batch_size * sequence_length, result_dim), dtype=x.dtype, device=x.device)
            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.group_num_experts[group_idx]).permute(2, 1, 0)
            # Loop over all available experts in the model and perform the computation on each expert
            for expert_idx in range(self.group_num_experts[group_idx]):
                expert_layer = self.expert_groups[group_idx][expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])
                if top_x.shape[0] == 0:
                    continue
                # in torch it is faster to index using lists than torch tensors
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()
                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, result_dim)
            group_final_hidden_states.append(final_hidden_states)

        final_hidden_states = torch.stack(group_final_hidden_states, 2)

        stack_final_hidden_states = torch.stack(group_final_hidden_states, dim=-1)
        stack_final_hidden_states = stack_final_hidden_states.permute(0, 1, 3, 2) 
        stack_final_hidden_states = stack_final_hidden_states.view(batch_size*sequence_length, len(self.group_num_experts), -1)
        final_hidden_states = self.experts_aggregator(stack_final_hidden_states) 
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, len(self.group_num_experts), -1)
        final_hidden_states = final_hidden_states.mean(dim=2)

        original_result = torch.nn.functional.linear(x, self.original_weight, self.original_bias)
        return original_result + final_hidden_states
    
class ExpertModel(nn.Module):
    config_path = "/data1/zhen/RA/LLMs/weights/esm2_t33_650M_UR50D"
    def __init__(self, multi_clusters, transformer_config):
        super(ExpertModel, self).__init__()
        start_time = datetime.now()
        self.model = EsmForMaskedLM.from_pretrained(self.config_path)
        config = EsmConfig.from_pretrained(self.config_path)
        self.multi_cluster_models = []
        for cluster in multi_clusters:
            cluster_models = []
            for _, from_checkpoint in cluster:
                cluster_model = EsmForMaskedLM(config)
                self.load_checkpoint(cluster_model, from_checkpoint)
                cluster_models.append(cluster_model)
            self.multi_cluster_models.append(cluster_models)
        self.initial_model(transformer_config)
        end_time = datetime.now()
        for name, param in self.model.named_parameters():
            if "encoder.layer.32." in name and "experts_aggregator" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elapsed_time = end_time - start_time
        print(f"Initialize expert model elapsed time: {elapsed_time.total_seconds()} seconds")

    @staticmethod
    def load_checkpoint(model, from_checkpoint):
        state_dict = torch.load(from_checkpoint)
        weights = state_dict["model"]
        model_dict = model.state_dict()

        unused_params = []
        missed_params = list(model_dict.keys())

        for k, v in weights.items():
            if k in model_dict.keys():
                model_dict[k] = v
                missed_params.remove(k)

            else:
                unused_params.append(k)

        if len(missed_params) > 0:
            print(f"\033[31mSome weights of {type(model).__name__} were not "
                  f"initialized from the model checkpoint: {missed_params}\033[0m")

        if len(unused_params) > 0:
            print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

        model.load_state_dict(model_dict)

    @staticmethod
    def frozen_paramters(model, entry):
        for name, param in model.named_parameters():
            if entry in name:
                param.requires_grad = False

    def initial_model(self, transformer_config):
        layer_idx_list = list(range(len(self.model.esm.encoder.layer)))
        for i in layer_idx_list:
            if i % 2 != 0:
                continue
            original_ffndown_weight = self.model.esm.encoder.layer[i].output.dense.weight
            original_ffndown_bias = self.model.esm.encoder.layer[i].output.dense.bias
            moeffndown = MixtureFFNDown(group_num_experts=[len(cluster_models) for cluster_models in self.multi_cluster_models], 
                                        transformer_config=transformer_config,
                                        in_features=5120,
                                        out_features=1280,
                                        num_experts_per_token=2,
                                        original_ffndown_weight=original_ffndown_weight,
                                        original_ffndown_bias=original_ffndown_bias)
            with torch.no_grad():
                for group_idx in range(len(self.multi_cluster_models)):
                    for k in range(len(self.multi_cluster_models[group_idx])):
                        moeffndown.expert_groups[group_idx][k].fc.weight.copy_(self.multi_cluster_models[group_idx][k].esm.encoder.layer[i].output.dense.weight)
                        moeffndown.expert_groups[group_idx][k].fc.bias.copy_(self.multi_cluster_models[group_idx][k].esm.encoder.layer[i].output.dense.bias)
            self.model.esm.encoder.layer[i].output.dense = moeffndown
            
    def forward(self, inputs):
        inputs['output_hidden_states'] = True
        outputs = self.model(**inputs)
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state

    
class TeacherModel(nn.Module):
    def __init__(self, num_labels, dim, depth, heads, mlp_dim, dropout, pool='mean'):
        super(TeacherModel, self).__init__()
        self.pool = pool
        start_time = datetime.now()
        
        # Cluster
        cluster0 = [
            ('aav', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/AAV/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster0-output.dense/esm2_t33_650M_UR50D.pt'),
            ('ec', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/EC/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster0-output.dense/esm2_t33_650M_UR50D.pt'),
            ('gobp', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/GO/BP/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster0-output.dense/esm2_t33_650M_UR50D.pt'),
            ('humanppi', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/HumanPPI/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster0-output.dense/esm2_t33_650M_UR50D.pt'),
            ('mutant', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/MoEMutant/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cls2/cluster0-output.dense/esm2_t33_650M_UR50D.pt'),
        ]
        cluster1 = [
            ('ec', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/EC/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster1-output.dense/esm2_t33_650M_UR50D.pt'),
            ('gobp', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/GO/BP/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster1-output.dense/esm2_t33_650M_UR50D.pt'),
            ('humanppi', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/HumanPPI/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster1-output.dense/esm2_t33_650M_UR50D.pt'),
            ('mutant', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/MoEMutant/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cls2/cluster1-output.dense/esm2_t33_650M_UR50D.pt'),
        ]
        cluster2 = [
            ('fl', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/Fluorescence/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster2-output.dense/esm2_t33_650M_UR50D.pt'),
            ('ec', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/EC/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster2-output.dense/esm2_t33_650M_UR50D.pt'),
            ('gobp', '/data1/zhen/RA/Projects/Meta-MOE/2_Property/1_Cluster/weights/GO/BP/tsne_and_agg_cls3-n_neighbors=21-mutant_frac=percent70/cluster2-output.dense/esm2_t33_650M_UR50D.pt'),
        ]

        all_clusters = [cluster0, cluster1, cluster2]
        # Expert Model
        transformer_config = TransformerConfig(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        transformer_config = asdict(transformer_config)
        self.teacher = nn.ModuleDict({
                'experts': ExpertModel(multi_clusters=all_clusters, transformer_config=transformer_config),
                'logits': nn.ModuleDict({
                    'layernorm': nn.LayerNorm(dim), 
                    'linear': nn.Linear(dim, num_labels),
                })
        })

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        print(f"Initialize teacher model elapsed time: {elapsed_time.total_seconds()} seconds")
        
    def forward(self, inputs):
        bs, seq_length = inputs['input_ids'].size()
        teacher_hidden_state = self.teacher['experts'](inputs)

        if self.pool == 'mean':
            teacher_outputs = teacher_hidden_state.mean(dim=1)
        elif self.pool == 'max':
            teacher_outputs = teacher_hidden_state.max(dim=1)
        elif self.pool == 'token':
            teacher_outputs = teacher_hidden_state
        else:
            raise ValueError('Error Pool.')
        
        teacher_outputs = self.teacher['logits']['layernorm'](teacher_outputs)
        teacher_logits = self.teacher['logits']['linear'](teacher_outputs)
        
        return teacher_logits, teacher_hidden_state
    

def load_teacher_checkpoint(model, teacher_checkpoint):
    state_dict = torch.load(teacher_checkpoint)
    weights = state_dict["model"]
    model_dict = model.state_dict()

    unused_params = []
    missed_params = list(model_dict.keys())

    for k, v in weights.items():
        if k in model_dict.keys() and 'teacher.logits' not in k:
            model_dict[k] = v
            missed_params.remove(k)

        else:
            unused_params.append(k)

    if len(missed_params) > 0:
        print(f"\033[31mSome weights of {type(model).__name__} were not "
                f"initialized from the model checkpoint: {missed_params}\033[0m")

    if len(unused_params) > 0:
        print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

    model.load_state_dict(model_dict)

def get_data(path):
    _10TB = 10995116277760
    # min_norm, max_norm = 40, 67
    env = lmdb.open(path, lock=False, map_size=_10TB)
    operator = env.begin()
    length = int(operator.get("length".encode()).decode())
    data_list = []
    for i in range(length):
        entry = json.loads(operator.get(str(i).encode()).decode())
        data_list.append(entry)
    df = pd.DataFrame(data_list)
    env.close()
    return df



