import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def linear_warmup_cosine_decay(optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float=0.1):
    """
    Stable linear warmup to the base LR, then a cosine decay down to the base_lr * min_lr_ratio

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        warmup_steps (int): The number of steps to linearly warm up the learning rate.
        total_steps (int): The total number of training steps.
        min_lr_ratio (float, optional): The minimum learning rate ratio. Defaults to 0.1.
    """

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    
    return LambdaLR(optimizer, lr_lambda)

def linear_warmup_polynomial_decay(optimizer: Optimizer, warmup_steps: int, total_steps: int,
                                   end_lr_ratio: float=0.0, power: float=1.0):
    """
    Linear warmup then polynomial decay to end_lr_ratio * base_lr.
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return (1 - progress) ** power * (1 - end_lr_ratio) + end_lr_ratio
    return LambdaLR(optimizer, lr_lambda)

def inverse_sqrt_warmup(optimizer: Optimizer, warmup_steps: int):
    """
    Classic transformer schedule: lr * min(step^{-0.5}, step * warmup^{-1.5})
    Implemented through LambdaLR.
    """
    def lr_lambda(step: int):
        step = max(1, step)
        if step < warmup_steps:
            return step / (warmup_steps ** 1.5)
        return (step ** -0.5)
    return LambdaLR(optimizer, lr_lambda)