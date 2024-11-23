import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple
from enum import Enum

class TaskType(Enum):
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'
    MULTILABEL = 'multilabel'

class FocalLoss(nn.Module):
    def __init__(
        self,
        task_type: TaskType,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Generalized Focal Loss for various classification tasks
        
        Args:
            task_type (TaskType): Type of task (binary, multiclass, multilabel)
            alpha (float): Weighting factor for rare classes
            gamma (float): Focusing parameter for hard examples
            reduction (str): 'mean', 'sum', or 'none'
            class_weights (torch.Tensor, optional): Manual class weights
        """
        super().__init__()
        self.task_type = task_type
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) or (B, ) depending on task_type
            targets: (B, C) or (B, ) depending on task_type
        """
        if self.task_type == TaskType.BINARY:
            return self._binary_focal_loss(inputs, targets)
        elif self.task_type == TaskType.MULTICLASS:
            return self._multiclass_focal_loss(inputs, targets)
        elif self.task_type == TaskType.MULTILABEL:
            return self._multilabel_focal_loss(inputs, targets)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _binary_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure proper shapes
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        
        # Focal term
        pt = torch.exp(-bce_loss)
        focal_term = self.alpha * (1 - pt) ** self.gamma
        
        # Combine
        loss = focal_term * bce_loss
        
        return self._reduce(loss)
    
    def _multiclass_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert one-hot to class indices if necessary
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)
        
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,
            reduction='none'
        )
        
        pt = torch.exp(-ce_loss)
        focal_term = self.alpha * (1 - pt) ** self.gamma
        
        loss = focal_term * ce_loss
        return self._reduce(loss)
    
    def _multilabel_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE loss for each label
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(),
            weight=self.class_weights,
            reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        focal_term = self.alpha * (1 - pt) ** self.gamma
        
        loss = focal_term * bce_loss
        # Sum across labels, then reduce across batch
        loss = loss.sum(dim=1)
        return self._reduce(loss)
    
    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class GeneralizedLoss(nn.Module):
    def __init__(
        self,
        task_configs: Union[Dict[str, Dict], TaskType],
        learn_weights: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Generalized loss function for single/multi-task learning
        
        Args:
            task_configs: For single-task, pass TaskType enum
                        For multi-task, pass dict of format:
                        {
                            'task_name': {
                                'type': TaskType,
                                'alpha': float,
                                'gamma': float,
                                'weight': float,  # Optional task weight
                                'class_weights': torch.Tensor  # Optional class weights
                            }
                        }
            learn_weights: Whether to learn task weights for MTL
            device: Device to place the loss function on
        """
        super().__init__()
        self.device = device
        
        # Handle single-task case
        if isinstance(task_configs, TaskType):
            self.single_task = True
            self.task_configs = {'main': {'type': task_configs}}
        else:
            self.single_task = False
            self.task_configs = task_configs
        
        # Initialize loss functions
        self.loss_functions = {}
        for task, config in self.task_configs.items():
            if config['type'] == TaskType.REGRESSION:
                self.loss_functions[task] = nn.MSELoss()
            else:
                self.loss_functions[task] = FocalLoss(
                    task_type=config['type'],
                    alpha=config.get('alpha', 1.0),
                    gamma=config.get('gamma', 2.0),
                    class_weights=config.get('class_weights')
                )
        
        # Initialize learnable task weights for MTL
        if not self.single_task and learn_weights:
            self.log_vars = nn.Parameter(torch.zeros(len(self.task_configs)))
        else:
            self.log_vars = None
        
        self.to(device)
    
    def forward(
        self,
        predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Forward pass
        
        Args:
            predictions: For single-task, tensor of predictions
                       For multi-task, dict of predictions per task
            targets: For single-task, tensor of targets
                    For multi-task, dict of targets per task
        
        Returns:
            For single-task: loss tensor
            For multi-task: (total_loss, dict of individual losses)
        """
        if self.single_task:
            return self.loss_functions['main'](predictions, targets)
        
        total_loss = 0
        task_losses = {}
        
        for i, (task, loss_fn) in enumerate(self.loss_functions.items()):
            pred = predictions[task]
            target = targets[task]
            
            loss = loss_fn(pred, target)
            
            # Apply task weights if using learned weights
            if self.log_vars is not None:
                precision = torch.exp(-self.log_vars[i])
                loss = precision * loss + 0.5 * self.log_vars[i]
            else:
                # Apply manual task weights if specified
                weight = self.task_configs[task].get('weight', 1.0)
                loss = weight * loss
            
            total_loss += loss
            task_losses[task] = loss.item()
        
        return total_loss, task_losses

# Example usage:
if __name__ == "__main__":
    # Single-task binary classification
    binary_loss = GeneralizedLoss(TaskType.BINARY)
    
    # Single-task multiclass classification
    multiclass_loss = GeneralizedLoss(TaskType.MULTICLASS)
    
    # Multi-task learning
    mtl_configs = {
        'gender': {
            'type': TaskType.BINARY,
            'alpha': 0.25,
            'gamma': 2.0
        },
        'category': {
            'type': TaskType.MULTICLASS,
            'alpha': 0.25,
            'gamma': 2.0,
            'class_weights': torch.ones(10)  # for 10 classes
        },
        'color': {
            'type': TaskType.REGRESSION
        }
    }
    mtl_loss = GeneralizedLoss(mtl_configs, learn_weights=True)


