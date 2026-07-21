import os
import json
import torch.distributed as dist
import torchmetrics
import torch

from ..model_interface import register_model
from .base import SaprotBaseModel
from .teacher_moe_model import TeacherModel


@register_model
class SaprotRegressionModel(SaprotBaseModel):
    def __init__(self, teacher_name: str = None, test_result_path: str = None, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            **kwargs: other arguments for SaprotBaseModel
        """
        self.teacher_name = teacher_name
        self.test_result_path = test_result_path
        super().__init__(task="regression", **kwargs)
        #----------------------------- create save_metrics mark ----------------------------------#
        self.train_state = {'valid': [], 'test': []}
        #-----------------------------------------------------------------------------------------#  

        
    def initialize_model(self):
        teacher_kwargs = {}
        if self.teacher_name is not None:
            teacher_kwargs["teacher_checkpoint"] = self.teacher_name

        self.model = TeacherModel(
            num_labels=1,
            dim=1280,
            depth=1,
            heads=20,
            mlp_dim=2560,
            dropout=0.1,
            pool='mean',
            load_teacher_logits=False,
            **teacher_kwargs,
        )
        
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
            logits, _ = self.model(inputs)
            logits = logits.squeeze(dim=-1)

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
        #------------------------------ save test metrics mark -----------------------------------#
        self.train_state['test'].append({'epoch': self.epoch,
                                         'test_loss': log_dict['test_loss'].item(), 'test_spearman': log_dict['test_spearman'].item(), 'test_pearson': log_dict['test_pearson'].item()})
        if self.trainer.max_epochs > 0:
            with open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'w') as f:
                json.dump(self.train_state, f, indent=4)
        #------------------------------------------------------------------------------------------#
        
    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
        #------------------------------ save valid metrics mark -----------------------------------#
        self.train_state['valid'].append({'epoch': self.epoch + 1, 
                                          'valid_loss': log_dict['valid_loss'].item(), 'valid_spearman': log_dict['valid_spearman'].item(), 'valid_pearson': log_dict['valid_pearson'].item()})
        if self.trainer.max_epochs > 0:
            with open(os.path.join(os.path.dirname(self.save_path), 'train_state.json'), 'w') as f:
                json.dump(self.train_state, f, indent=4)
        #------------------------------------------------------------------------------------------#
