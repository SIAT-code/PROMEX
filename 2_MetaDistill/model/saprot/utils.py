import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from peft.tuners.lora import LoraLayer

lora_modules = ['lora_A',
                'lora_B',
                'lora_embedding_A',
                'lora_embedding_B',
                'lora_dropout']

def pack_lora_layers(model):
    '''
    Find Lora modules in a PeftModel and return them as a single ModuleList.
    Return their corresponding names (same as the ones from model.named_modules()) as well.
    '''
    packed_layers = torch.nn.ModuleList([])
    names = []
    key_list = [key for key, _ in model.named_modules() if 'lora' not in key]
    for key in key_list:
        target = model.get_submodule(key)
        if isinstance(target, LoraLayer):
            for attr in lora_modules:
                if model.active_adapter in getattr(target, attr):
                    names.append('.'.join([key, attr]))
                    packed_layers.append(getattr(target, attr))
    return names, packed_layers

def replace_modules(model, module_names, new_modules):
    '''
    Replace specific modules in the model.
    Names of the new modules should be given by model.named_modules()
    '''
    for name, module in zip(module_names, new_modules):
        name = name.split('.')
        parent = model.get_submodule('.'.join(name[:-1]))
        target_name = name[-1]
        setattr(parent, target_name, module)

def get_optimizer(optimizer, lr, params):
    params = filter(lambda p: p.requires_grad, params)
    if optimizer == 'sgd':
        return optim.SGD(params, lr=lr)
    elif optimizer == 'nag':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    elif optimizer == 'adadelta':
        return optim.Adadelta(params, lr=lr)
    elif optimizer == 'adam':
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError('Unknown optimizer: ' + optimizer)
    
    

class Esm2LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
        This is an implementation of ESM2's learning rate scheduler.
    """

    def __init__(self,
                 optimizer,
                 last_epoch: int = -1,
                 verbose: bool = False,
                 init_lr: float = 0.,
                 max_lr: float = 4e-4,
                 final_lr: float = 4e-5,
                 warmup_steps: int = 2000,
                 start_decay_after_n_steps: int = 500000,
                 end_decay_after_n_steps: int = 5000000,
                 on_use: bool = True,
                 ):
        
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.start_decay_after_n_steps = start_decay_after_n_steps
        self.end_decay_after_n_steps = end_decay_after_n_steps
        self.on_use = on_use
        super(Esm2LRScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def state_dict(self):
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            raise RuntimeError(
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )

        step_no = self.last_epoch
        if not self.on_use:
            return [base_lr for base_lr in self.base_lrs]

        if step_no <= self.warmup_steps:
            lr = self.init_lr + step_no / self.warmup_steps * (self.max_lr - self.init_lr)
        
        elif step_no <= self.start_decay_after_n_steps:
            lr = self.max_lr
        
        elif step_no <= self.end_decay_after_n_steps:
            portion = (step_no - self.start_decay_after_n_steps) / (self.end_decay_after_n_steps - self.start_decay_after_n_steps)
            lr = self.max_lr - portion * (self.max_lr - self.final_lr)
           
        else:
            lr = self.final_lr
    
        return [lr for group in self.optimizer.param_groups]