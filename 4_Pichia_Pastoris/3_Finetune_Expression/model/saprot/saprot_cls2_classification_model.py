import os
import json
import pandas as pd
import torchmetrics
import torch

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)
        #----------------------------- create save_metrics mark ----------------------------------#
        self.train_state = {'valid': [], 'test': []}
        #-----------------------------------------------------------------------------------------# 
        self.predictions = {'label': [], 'pred': []}
        #-----------------------------------------------------------------------------------------#
    def initialize_metrics(self, stage):
        
        class AUC(torchmetrics.AUC):
            def update(self, preds: torch.Tensor, target: torch.Tensor):
                preds = torch.softmax(preds, dim=1)[:, 1]
                super().update(preds, target)
                
        class AUROC(torchmetrics.AUROC):
            def update(self, preds: torch.Tensor, target: torch.Tensor):
                preds = torch.softmax(preds, dim=1)[:, 1]
                super().update(preds, target)
                
        return {f"{stage}_acc": torchmetrics.Accuracy(),
                f"{stage}_auc": AUC(task='binary', reorder=True),
                f"{stage}_aucroc": AUROC(pos_label=1)}

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
            # metric.update(torch.softmax(logits.detach(), dim=1)[:, 1], label)  # mark
            metric.update(logits.detach(), label)
        #------------------------------------ mark ---------------------------------------#
        if stage == 'test':
            self.predictions['label'] += label.tolist()
            self.predictions['pred'] += torch.softmax(logits.detach(), dim=1)[:, 1].tolist()
        #---------------------------------------------------------------------------------#
        
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
        #------------------------------ save test metrics mark -----------------------------------#
        if self.trainer.max_epochs > 0:
            if self.use_lora and os.path.isfile(os.path.join(os.path.dirname(self.save_path), 'train_state.json')):
                self.train_state = json.load(open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'r'))
            self.train_state['test'].append({'epoch': self.epoch, 'test_loss': log_dict['test_loss'].item(), 'test_auc': log_dict['test_auc'].item(), \
                                    'test_aucroc': log_dict['test_aucroc'].item(), 'test_acc': log_dict['test_acc'].item()})
            with open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'w') as f:
                json.dump(self.train_state, f, indent=4)
        pd.DataFrame(self.predictions).to_csv(os.path.join(os.path.dirname(self.save_path), 'predictions.txt'), index=0)
        #------------------------------------------------------------------------------------------#
        
    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_aucroc"], mode="max")
        #------------------------------ save valid metrics mark -----------------------------------#
        self.train_state['valid'].append({'epoch': self.epoch + 1, 'valid_loss': log_dict['valid_loss'].item(), 'valid_auc': log_dict['valid_auc'].item(), \
                                            'valid_aucroc': log_dict['valid_aucroc'].item(), 'valid_acc': log_dict['valid_acc'].item()})
        if self.trainer.max_epochs > 0:
            with open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'w') as f:
                json.dump(self.train_state, f, indent=4)
        #------------------------------------------------------------------------------------------#