import os
import json
import torch.distributed as dist
import torchmetrics
import torch
import torch.nn.functional as F

from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotRegressionModel(SaprotBaseModel):
    def __init__(self,
                 test_result_path: str = None,
                 loss_type: str = "mse",
                 mse_loss_weight: float = 1.0,
                 rank_loss_weight: float = 1.0,
                 list_size: int = None,
                 save_metric: str = "valid_loss",
                 save_mode: str = "min",
                 **kwargs):
        """
        Args:
            test_result_path: path to save test result
            loss_type: one of ["mse", "listwise", "mse_listwise"]
            mse_loss_weight: weight of mse loss when loss_type is "mse_listwise"
            rank_loss_weight: weight of listwise ranking loss when loss_type is "mse_listwise"
            list_size: ranking list size. None means using the whole batch as one list.
            save_metric: validation metric used to save the best checkpoint
            save_mode: "min" or "max"
            **kwargs: other arguments for SaprotBaseModel
        """
        self.test_result_path = test_result_path
        self.loss_type = loss_type
        self.mse_loss_weight = mse_loss_weight
        self.rank_loss_weight = rank_loss_weight
        self.list_size = None if list_size is None else int(list_size)
        self.save_metric = save_metric
        self.save_mode = save_mode
        self._validate_loss_config()
        super().__init__(task="regression", **kwargs)
        #----------------------------- create save_metrics mark ----------------------------------#
        self.train_state = {'valid': [], 'test': []}
        self.train_log_dict = {}  # mark add
        #-----------------------------------------------------------------------------------------#  

    def _validate_loss_config(self):
        supported_loss_types = {"mse", "listwise", "mse_listwise"}
        if self.loss_type not in supported_loss_types:
            raise ValueError(f"loss_type should be one of {supported_loss_types}, got {self.loss_type}")

        if self.save_mode not in {"min", "max"}:
            raise ValueError(f"save_mode should be 'min' or 'max', got {self.save_mode}")

        if self.mse_loss_weight < 0 or self.rank_loss_weight < 0:
            raise ValueError("mse_loss_weight and rank_loss_weight should be non-negative")

        if self.list_size is not None and self.list_size <= 1:
            raise ValueError("list_size should be greater than 1, or None to use the whole batch")

    @staticmethod
    def listwise_ranking_loss(predicts, targets):
        """
        Listwise ranking loss from VenusFSFP, computed with logcumsumexp for fp16 stability.
        """
        if predicts.dim() == 1:
            predicts = predicts.unsqueeze(0)
            targets = targets.unsqueeze(0)

        if predicts.shape[1] <= 1:
            return predicts.sum() * 0.0

        indices = targets.sort(descending=True, dim=-1).indices
        predicts = torch.gather(predicts.float(), dim=1, index=indices)
        log_cumsums = torch.logcumsumexp(predicts.flip(dims=[1]), dim=1).flip(dims=[1])
        loss = log_cumsums - predicts
        return loss.sum(dim=1).mean()

    def compute_listwise_ranking_loss(self, outputs, fitness):
        outputs = outputs.reshape(-1)
        fitness = fitness.reshape(-1)

        if self.list_size is None or outputs.shape[0] <= self.list_size:
            return self.listwise_ranking_loss(outputs, fitness)

        usable_size = outputs.shape[0] // self.list_size * self.list_size
        if usable_size <= 1:
            return outputs.sum() * 0.0

        outputs = outputs[:usable_size].reshape(-1, self.list_size)
        fitness = fitness[:usable_size].reshape(-1, self.list_size)
        return self.listwise_ranking_loss(outputs, fitness)

    def compute_regression_loss(self, outputs, fitness):
        mse_loss = F.mse_loss(outputs, fitness)

        if self.loss_type == "mse":
            return mse_loss

        rank_loss = self.compute_listwise_ranking_loss(outputs, fitness)
        if self.loss_type == "listwise":
            return rank_loss

        return self.mse_loss_weight * mse_loss + self.rank_loss_weight * rank_loss
        
    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
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
        loss = self.compute_regression_loss(outputs, fitness)
        
        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.update(outputs.detach().float(), fitness.float())
        
        if stage == "train":
            # Skip calculating metrics if the batch size is 1
            if fitness.shape[0] > 1:
                log_dict = self.get_log_dict("train")
                self.train_log_dict = log_dict  # mark add
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
        #------------------------------ save test metrics mark -----------------------------------#
        self.train_state['test'].append({'epoch': self.epoch,
                                         'test_loss': log_dict['test_loss'].item(), 'test_spearman': log_dict['test_spearman'].item(), \
                                            'test_pearson': log_dict['test_pearson'].item()})
        if self.trainer.max_epochs > 0:
            with open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'w') as f:
                json.dump(self.train_state, f, indent=4)
        #------------------------------------------------------------------------------------------#
        
    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")
        train_log_dict = self.train_log_dict if self.train_log_dict else {'train_loss': 0.0, 'train_spearman': 0.0, 'train_pearson': 0.0}  # mark add
        self.log_info(log_dict)
        self.reset_metrics("valid")
        if self.save_metric not in log_dict:
            raise KeyError(f"save_metric {self.save_metric} is not in validation metrics: {list(log_dict.keys())}")
        self.check_save_condition(log_dict[self.save_metric], mode=self.save_mode)  # origin: valid_spearman, max; valid_loss, min.  mark
        #------------------------------ save valid metrics mark -----------------------------------#
        self.train_state['valid'].append({'epoch': self.epoch + 1, 
                                        'train_loss': train_log_dict['train_loss'].item(), 'train_spearman': train_log_dict['train_spearman'].item(), \
                                        'train_pearson': train_log_dict['train_pearson'].item(), \
                                        'valid_loss': log_dict['valid_loss'].item(), 'valid_spearman': log_dict['valid_spearman'].item(), \
                                        'valid_pearson': log_dict['valid_pearson'].item()
                                        })
        if self.trainer.max_epochs > 0:
            with open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'w') as f:
                json.dump(self.train_state, f, indent=4)
        #------------------------------------------------------------------------------------------#
        self.train_log_dict = {}