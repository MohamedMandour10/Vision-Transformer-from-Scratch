import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Union, List

class ViTOptimizer:
    """
    Optimizer and learning rate scheduler for Vision Transformer.
    Implements AdamW with weight decay and cosine learning rate schedule with warmup.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.05,
        warmup_steps: int = 10000,
        max_steps: int = 100000,
        min_lr: float = 1e-5,
        no_weight_decay_params: Optional[List[str]] = None
    ):
        """
        Initialize optimizer and scheduler.
        
        Args:
            model: Vision Transformer model
            lr: Peak learning rate after warmup
            betas: Adam beta parameters
            weight_decay: Weight decay coefficient
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            min_lr: Minimum learning rate at the end of training
            no_weight_decay_params: List of parameter names to exclude from weight decay
        """
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        
        # Separate parameters that should and shouldn't have weight decay applied
        if no_weight_decay_params is None:
            no_weight_decay_params = [
                'bias', 'LayerNorm.weight', 'layernorm.weight',
                'layer_norm.weight', 'norm.weight'
            ]
            
        decay_parameters = []
        no_decay_parameters = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in no_weight_decay_params):
                    no_decay_parameters.append(param)
                else:
                    decay_parameters.append(param)
        
        # Create optimizer with different parameter groups
        self.optimizer = AdamW(
            [
                {
                    'params': decay_parameters,
                    'weight_decay': weight_decay
                },
                {
                    'params': no_decay_parameters,
                    'weight_decay': 0.0
                }
            ],
            lr=lr,
            betas=betas
        )
        
        # Create learning rate scheduler
        self.scheduler = self.create_scheduler()
        
    def create_scheduler(self) -> LambdaLR:
        """
        Creates a cosine learning rate scheduler with linear warmup.
        """
        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            
            # Cosine decay
            progress = float(current_step - self.warmup_steps) / \
                      float(max(1, self.max_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Ensure learning rate doesn't go below min_lr
            return max(self.min_lr / self.lr, cosine_decay)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def step(self, loss: torch.Tensor) -> None:
        """
        Performs a single optimization step.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def get_current_lr(self) -> float:
        """
        Returns the current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> dict:
        """
        Returns the state dict of optimizer and scheduler for checkpointing.
        """
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads optimizer and scheduler state from checkpoint.
        
        Args:
            state_dict: Dictionary containing optimizer and scheduler states
        """
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

