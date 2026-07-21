from datetime import datetime
from typing import List
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from transformers import EsmConfig, EsmTokenizer, EsmForMaskedLM, EsmForSequenceClassification, EsmForTokenClassification



PRETRAINED_TEACHER_MODEL_PATH = "../weights/pretrained/moe_teacher_pretrained_model.pt"


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

        self.experts_aggregator = Transformer(**transformer_config)
   
        
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

        stack_final_hidden_states = torch.stack(group_final_hidden_states, dim=-1)  # (bs, seq_length, 1280, len(self.group_num_experts))
        stack_final_hidden_states = stack_final_hidden_states.permute(0, 1, 3, 2)  # (bs, seq_length, len(self.group_num_experts), 1280)
        stack_final_hidden_states = stack_final_hidden_states.view(batch_size*sequence_length, len(self.group_num_experts), -1)  # (bs*seq_length, len(self.group_num_experts), 1280)
        final_hidden_states = self.experts_aggregator(stack_final_hidden_states)  # (bs*seq_length, len(self.group_num_experts), 1280)
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, len(self.group_num_experts), -1)  # (bs, seq_length, len(self.group_num_experts), 1280)
        final_hidden_states = final_hidden_states.mean(dim=2)

        original_result = torch.nn.functional.linear(x, self.original_weight, self.original_bias)
        return original_result + final_hidden_states
    
class ExpertModel(nn.Module):
    config_path = "../weights/esm2_t33_650M_UR50D"

    def __init__(
        self,
        group_num_experts,
        transformer_config,
        init_backbone_from_pretrained=False,
    ):
        super(ExpertModel, self).__init__()
        start_time = datetime.now()

        # When a complete TeacherModel checkpoint will be loaded later, config-only initialization is sufficient.
        config = EsmConfig.from_pretrained(self.config_path)
        if init_backbone_from_pretrained:
            self.model = EsmForMaskedLM.from_pretrained(self.config_path)
        else:
            self.model = EsmForMaskedLM(config)

        # Build the same MoE structure using the number of experts in each group.
        self.initial_model(
            group_num_experts=group_num_experts,
            transformer_config=transformer_config,
        )

        # Train only the replaced output.dense modules in the encoder.
        for name, param in self.model.named_parameters():
            if "encoder.layer.32." in name and "experts_aggregator" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        elapsed_time = datetime.now() - start_time
        print(
            "Initialize expert model architecture elapsed time: "
            f"{elapsed_time.total_seconds()} seconds"
        )

    def initial_model(self, group_num_experts, transformer_config):
        group_num_experts = list(group_num_experts)
        if not group_num_experts or any(num_experts <= 0 for num_experts in group_num_experts):
            raise ValueError(
                "group_num_experts must contain positive integers, "
                f"but got {group_num_experts}."
            )

        for layer_idx, layer in enumerate(self.model.esm.encoder.layer):
            # Match the original implementation by replacing FFN output.dense only in even-indexed layers.
            if layer_idx % 2 != 0:
                continue

            original_dense = layer.output.dense
            moeffndown = MixtureFFNDown(
                group_num_experts=group_num_experts,
                transformer_config=transformer_config,
                in_features=original_dense.in_features,
                out_features=original_dense.out_features,
                num_experts_per_token=1,
                original_ffndown_weight=original_dense.weight,
                original_ffndown_bias=original_dense.bias,
            )

            # Do not copy weights from cluster checkpoints here.
            # All expert, gate, and aggregator parameters will be overwritten by the complete teacher checkpoint.
            layer.output.dense = moeffndown

    def forward(self, inputs):
        inputs['output_hidden_states'] = True
        outputs = self.model(**inputs)
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state


class TeacherModel(nn.Module):
    def __init__(
        self,
        num_labels,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout,
        pool='mean',
        group_num_experts=(5, 2, 1, 1),
        init_backbone_from_pretrained=False,
        teacher_checkpoint=PRETRAINED_TEACHER_MODEL_PATH,
        load_teacher_logits=False,
    ):
        super(TeacherModel, self).__init__()
        self.pool = pool
        start_time = datetime.now()

        # Use only these counts to build the same structure without reading any cluster checkpoint.
        transformer_config = TransformerConfig(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        transformer_config = asdict(transformer_config)

        self.teacher = nn.ModuleDict({
            'experts': ExpertModel(
                group_num_experts=group_num_experts,
                transformer_config=transformer_config,
                init_backbone_from_pretrained=init_backbone_from_pretrained,
            ),
            'logits': nn.ModuleDict({
                'layernorm': nn.LayerNorm(dim),
                'linear': nn.Linear(dim, num_labels),
            }),
        })

        if teacher_checkpoint:
            load_teacher_checkpoint(
                self,
                teacher_checkpoint=teacher_checkpoint,
                load_logits=load_teacher_logits,
            )

        elapsed_time = datetime.now() - start_time
        print(
            "Initialize teacher model architecture elapsed time: "
            f"{elapsed_time.total_seconds()} seconds"
        )

    def forward(self, inputs):
        teacher_hidden_state = self.teacher['experts'](inputs)

        if self.pool == 'mean':
            teacher_outputs = teacher_hidden_state.mean(dim=1)
        elif self.pool == 'max':
            teacher_outputs = teacher_hidden_state.max(dim=1).values
        elif self.pool == 'token':
            teacher_outputs = teacher_hidden_state
        else:
            raise ValueError(f'Unsupported pool method: {self.pool}')

        teacher_outputs = self.teacher['logits']['layernorm'](teacher_outputs)
        teacher_logits = self.teacher['logits']['linear'](teacher_outputs)

        return teacher_logits, teacher_hidden_state


def load_teacher_checkpoint(model, teacher_checkpoint, load_logits=True):
        state_dict = torch.load(teacher_checkpoint, map_location='cpu')
        weights = state_dict["model"]
        model_dict = model.state_dict()

        unused_params = []
        skipped_params = []
        missed_params = list(model_dict.keys())
        skipped_logits_keys = set()

        checkpoint_logits_keys = [
            k for k in weights.keys()
            if k in model_dict.keys() and k.startswith('teacher.logits')
        ]
        logits_shape_mismatches = [
            k for k in checkpoint_logits_keys
            if tuple(model_dict[k].shape) != tuple(weights[k].shape)
        ]

        if checkpoint_logits_keys and (not load_logits or logits_shape_mismatches):
            skipped_logits_keys.update(checkpoint_logits_keys)
            reason = 'logits loading disabled'
            if logits_shape_mismatches:
                reason = 'logits head shape mismatch'
            for k in checkpoint_logits_keys:
                skipped_params.append((k, tuple(weights[k].shape), tuple(model_dict[k].shape), reason))

        for k, v in weights.items():
            if k not in model_dict.keys():
                unused_params.append(k)
                continue

            if k in skipped_logits_keys:
                continue

            if tuple(model_dict[k].shape) != tuple(v.shape):
                skipped_params.append((k, tuple(v.shape), tuple(model_dict[k].shape), 'shape mismatch'))
                continue

            model_dict[k] = v
            missed_params.remove(k)

        model.load_state_dict(model_dict)





