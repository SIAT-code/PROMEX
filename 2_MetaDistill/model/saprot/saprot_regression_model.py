import os
import json
import torch.distributed as dist
import torchmetrics
import torch
import torch.nn as nn
from ..model_interface import register_model
from .base import SaprotBaseModel
from .teacher_moe import TeacherModel as TeacherModelMoe # mark

from peft import PeftModel
from learn2learn.algorithms import MAML
from .utils import pack_lora_layers, replace_modules, get_optimizer, Esm2LRScheduler


@register_model
class MetaDistillSaprotRegressionModel(SaprotBaseModel):
    def __init__(self, 
                 test_result_path: str = None,
                 teacher_name: str = None,
                 teacher_checkpoint: str = None,
                 alpha=0.5, temperature=4.0, max_grad_norm=5, adapt_lr=1e-4, first_order=True, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            **kwargs: other arguments for SaprotBaseModel
        """
        self.test_result_path = test_result_path
        super().__init__(task="regression", **kwargs)
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher = TeacherModelMoe(num_labels=1, dim=1280, depth=1, heads=20, mlp_dim=2560, dropout=0.1, pool='mean')
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
            t_logits = t_logits.squeeze(dim=-1)
            logits = self.model(**inputs['inputs']).logits.squeeze(dim=-1)
            output = 0.5 * torch.nn.functional.mse_loss(logits, labels['labels']) + 0.5 * torch.nn.functional.mse_loss(t_logits, labels['labels'])
            return_val = (output, logits, labels['labels'])

        else: # make predictions for testing, eval_batch should be regular dataset
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
            preds_list.append(logits.detach().float())
            targets_list.append(targets.detach().float())

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(1. / len(loss))
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        if len(preds_list) > 0:
            all_preds = torch.cat(preds_list)
            all_targets = torch.cat(targets_list)
            
            if all_targets.shape[0] > 1:
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
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}
    
    def forward(self, inputs, structure_info=None):
        if structure_info:
            # To be implemented
            raise NotImplementedError

        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states(inputs, reduction="mean"))
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x).squeeze(dim=-1)

        else:
            logits = self.model(**inputs).logits.squeeze(dim=-1)

        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)
        
        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.update(outputs.detach().float(), fitness.float())
        
        if stage == "train":
            # Skip calculating metrics if the batch size is 1
            if fitness.shape[0] > 1:
                log_dict = self.get_log_dict("train")
                self.log_info(log_dict)
            
            # Reset train metrics
            self.reset_metrics("train")
        
        return loss

    def test_epoch_end(self, outputs):
        if self.test_result_path is not None:
            from torchmetrics.utilities.distributed import gather_all_tensors
            
            preds = self.test_spearman.preds
            preds[-1] = preds[-1].unsqueeze(dim=0) if preds[-1].shape == () else preds[-1]
            preds = torch.cat(gather_all_tensors(torch.cat(preds, dim=0)))
            
            targets = self.test_spearman.target
            targets[-1] = targets[-1].unsqueeze(dim=0) if targets[-1].shape == () else targets[-1]
            targets = torch.cat(gather_all_tensors(torch.cat(targets, dim=0)))

            if dist.get_rank() == 0:
                with open(self.test_result_path, 'w') as w:
                    w.write("pred\ttarget\n")
                    for pred, target in zip(preds, targets):
                        w.write(f"{pred.item()}\t{target.item()}\n")
        
        log_dict = self.get_log_dict("test")
        
        print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")
        self.train_state['test'].append({'epoch': self.epoch,
                                         'test_loss': log_dict['test_loss'].item(), 'test_spearman': log_dict['test_spearman'].item(), \
                                            'test_R2': log_dict['test_R2'].item(), 'test_pearson': log_dict['test_pearson'].item()})
        if self.trainer.max_epochs > 0:
            with open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'w') as f:
                json.dump(self.train_state, f, indent=4)
        
    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")
        if hasattr(self, 'last_train_metrics') and self.last_train_metrics:
            train_log_dict = self.last_train_metrics
        else:
            # 默认空值，避免 KeyError
            train_log_dict = {
                'train_loss': 0.0, 'train_spearman': 0.0, 
                'train_R2': 0.0, 'train_pearson': 0.0
            }
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
        self.train_state['valid'].append({'epoch': self.epoch + 1,
                                          'train_loss': train_log_dict.get('train_loss', 0.0), 'train_spearman': train_log_dict.get('train_spearman', 0.0), \
                                            'train_R2': train_log_dict.get('train_R2', 0.0), 'train_pearson': train_log_dict.get('train_pearson', 0.0), \
                                          'valid_loss': log_dict['valid_loss'].item(), 'valid_spearman': log_dict['valid_spearman'].item(), \
                                            'valid_R2': log_dict['valid_R2'].item(), 'valid_pearson': log_dict['valid_pearson'].item()
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