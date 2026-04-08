import os
import json
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
from .teacher_moe import TeacherModel as TeacherModelMoe

from peft import PeftModel
from learn2learn.algorithms import MAML
from .utils import pack_lora_layers, replace_modules, get_optimizer, Esm2LRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F

@register_model
class MetaDistillSaprotClassificationModel(SaprotBaseModel):
    def __init__(self, 
                 num_labels: int,
                 teacher_name: str = None,
                 teacher_checkpoint: str = None,
                 alpha=0.5, temperature=4.0, max_grad_norm=5, adapt_lr=1e-4, first_order=True, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher = TeacherModelMoe(num_labels=self.num_labels, dim=1280, depth=1, heads=20, mlp_dim=2560, dropout=0.1, pool='mean')
        self.load_teacher_checkpoint(self.teacher, self.teacher_checkpoint)
        self.init_optimizers()
        self.alpha = alpha
        self.temperature = temperature
        self.max_grad_norm = max_grad_norm
        
        if isinstance(self.model, PeftModel):
            self.adapter_name, adapter = pack_lora_layers(self.model)
            self.adapter = MAML(adapter, adapt_lr, first_order)
        else:
            self.model = MAML(self.model, adapt_lr, first_order, allow_nograd=True)
        
        self.train_state = {'valid': [], 'test': []}
        self.last_train_metrics = {}

    def compute_l2_loss(self, student_logits, teacher_logits):
        l2_loss = torch.square(teacher_logits - student_logits)
        l2_loss = torch.mean(l2_loss)
        return l2_loss
        
    def compute_distillation_loss(self, student_logits, teacher_logits):
        # Distillation Loss Multi Classification
        distillation_loss = torch.nn.functional.kl_div(
                            torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1),
                            torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1),
                            reduction='batchmean'
        ) * (self.temperature ** 2)
        return distillation_loss
 
    def fast_adapt(self, adapt_batch, eval_batch, training=True):
        # copy model for meta-training
        if isinstance(self.model, PeftModel): # replace the adapter with a cloned one
            cloned_adapter = self.adapter.clone()
            replace_modules(self.model, self.adapter_name, cloned_adapter.module)
            adapt = cloned_adapter.adapt
        else: # simply copy the full model
            backup = self.model
            self.model = self.model.clone()
            adapt = self.model.adapt

        self.teacher.eval()
        for batch in adapt_batch:
            inputs, labels = batch
            # s_logits = self.model(**inputs['inputs']).logits
            s__hidden_state = self.model(**inputs['inputs'], output_hidden_states=True).hidden_states[-1]
            t_logits, t_hidden_state = self.teacher(inputs['inputs'])
            # adapt_loss = self.compute_distillation_loss(s_logits, t_logits)
            # adapt_loss = self.compute_distillation_loss(s__hidden_state.mean(dim=1), t_hidden_state.mean(dim=1))
            adapt_loss = self.compute_l2_loss(s__hidden_state, t_hidden_state)
            adapt(adapt_loss)
        
        if training: # compute loss for training, eval_batch should be ranking dataset
            inputs, labels = eval_batch
            t_logits, t_hidden_state = self.teacher(inputs['inputs'])
            logits = self.model(**inputs['inputs']).logits
            output = 0.5 * cross_entropy(logits, labels['labels']) + 0.5 * cross_entropy(t_logits, labels['labels'])
            return_val = (output, logits, labels['labels'])
        else:
            with torch.no_grad():
                self.model.eval()
                inputs, labels = eval_batch
                output = self.model(**inputs['inputs']).logits
                self.model.train()
            return_val = output

        if isinstance(self.model, PeftModel):
            replace_modules(self.model, self.adapter_name, self.adapter.module)
        else:
            self.model = backup

        return return_val

    def training_step(self, batch, batch_idx):
    
        loss = []
        preds_list = []
        targets_list = []

        self.optimizer.zero_grad()
        for adapt_batch, eval_batch in zip(batch['adapt_batches'], batch['eval_batches']):
            eval_loss, logits, targets = self.fast_adapt(adapt_batch, eval_batch)
            if not eval_loss.isfinite():
                continue
            loss.append(eval_loss)

            preds_list.append(logits.detach())
            targets_list.append(targets.detach())

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(1. / len(loss))
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        if len(preds_list) > 0:
            all_preds = torch.cat(preds_list)
            all_targets = torch.cat(targets_list)
            

            for name, metric in self.metrics['train'].items():
                if name != 'train_loss':
                    metric.update(all_preds, all_targets)

 
            log_dict = self.get_log_dict("train")
            log_dict = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in log_dict.items()}
 
            avg_loss = sum(loss) / len(loss)
            log_dict['train_loss'] = avg_loss.item()


            self.last_train_metrics = log_dict.copy()

            self.log_info(log_dict)
            

            self.reset_metrics("train")

        return sum(loss) / len(loss)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states(inputs, reduction="mean"))
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)

        else:
            logits = self.model(**inputs).logits

        return logits

    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)
        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def test_epoch_end(self, outputs):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()

        print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

        self.train_state['test'].append({'epoch': self.epoch, 'test_loss': log_dict['test_loss'].item(), 'test_acc': log_dict['test_acc'].item()})
        if self.trainer.max_epochs > 0:
            with open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'w') as f:
                json.dump(self.train_state, f, indent=4)
        
    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()
        if hasattr(self, 'last_train_metrics') and self.last_train_metrics:
            train_log_dict = self.last_train_metrics
        else:
            train_log_dict = {'train_loss': 0.0, 'train_acc': 0.0}
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min") 
        self.train_state['valid'].append({'epoch': self.epoch + 1,
                                          'train_loss': train_log_dict.get('train_loss', 0.0), 'train_acc': train_log_dict.get('train_acc', 0.0), \
                                          'valid_loss': log_dict['valid_loss'].item(), 'valid_acc': log_dict['valid_acc'].item()
                                        })
        if self.trainer.max_epochs > 0:
            with open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'w') as f:
                json.dump(self.train_state, f, indent=4)
        

    def init_optimizers(self):
        # No decay for layer norm and bias
        no_decay = ['LayerNorm.weight', 'bias']
        
        if "weight_decay" in self.optimizer_kwargs:
            weight_decay = self.optimizer_kwargs.pop("weight_decay")
        else:
            weight_decay = 0.01
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)] + \
                        [p for n, p in self.teacher.named_parameters() if p.requires_grad],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                           lr=self.lr_scheduler_kwargs['init_lr'],
                                           **self.optimizer_kwargs)

        self.lr_scheduler = Esm2LRScheduler(self.optimizer, **self.lr_scheduler_kwargs)
    
    def configure_optimizers(self):
        return {"optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": self.lr_scheduler,
                                 "interval": "step",
                                 "frequency": 1}
                }

    @staticmethod
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

        

            
