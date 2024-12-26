import torch
# from .torch_utils import *

class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9, nesterov=False):
        super().__init__(params, lr, weight_decay, nesterov=nesterov)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        
        self.__initial_lr = [group['lr'] for group in self.param_groups]
    
    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

def get_parameter_groups(model, print_fn=print):
        groups = ([], [], [], [])

        for name, value in model.named_parameters():
            # pretrained weights
            if 'fc' in name:
                if 'weight' in name:
                    # print_fn(f'pretrained weights : {name}')
                    groups[2].append(value)
                else:
                    # print_fn(f'pretrained bias : {name}')
                    groups[3].append(value)
                    
            # scracthed weights
            else:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'scratched weights : {name}')
                    groups[0].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'scratched bias : {name}')
                    groups[1].append(value)
        return groups
