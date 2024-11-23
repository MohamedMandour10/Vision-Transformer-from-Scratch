import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, defaultdict
import logging
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Any,  Union
from loss import TaskType, GeneralizedLoss
from optimizer import ViTOptimizer
from models.vit import VisionTransformer

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_configs: Union[Dict[str, Dict], TaskType],
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Trainer for Vision Transformer with support for single/multi-task learning
        
        Args:
            model: Vision Transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            task_configs: Task configuration for GeneralizedLoss
            config: Training configuration dictionary containing:
                - num_epochs: Number of training epochs
                - learning_rate: Peak learning rate
                - weight_decay: Weight decay coefficient
                - warmup_steps: Number of warmup steps
                - checkpoint_dir: Directory to save checkpoints
                - log_interval: Steps between logging
                - eval_interval: Steps between evaluations
                - use_wandb: Whether to use Weights & Biases logging
            device: Device to train on (default: auto-detect)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize loss function
        self.criterion = GeneralizedLoss(
            task_configs=task_configs,
            learn_weights=config.get('learn_task_weights', True),
            device=self.device
        )
        
        # Calculate total steps for optimizer
        steps_per_epoch = len(train_loader)
        total_steps = config['num_epochs'] * steps_per_epoch
        
        # Initialize optimizer
        self.optimizer = ViTOptimizer(
            model=model,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            warmup_steps=config['warmup_steps'],
            max_steps=total_steps,
            min_lr=config.get('min_lr', 1e-5)
        )
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Initialize W&B if requested
        if config.get('use_wandb', False):
            wandb.init(project=config.get('wandb_project', 'vit-training'),
                      config=config)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dict containing training metrics
        """
        self.model.train()
        epoch_metrics = {
            'train_loss': 0.0,
            'train_task_losses': defaultdict(float)
        }
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            if self.criterion.single_task:
                loss = self.criterion(outputs, targets)
                task_losses = {'main': loss.item()}
            else:
                loss, task_losses = self.criterion(outputs, targets)
            
            # Optimization step
            self.optimizer.step(loss)
            
            # Update metrics
            epoch_metrics['train_loss'] += loss.item()
            for task, task_loss in task_losses.items():
                epoch_metrics['train_task_losses'][task] += task_loss
            
            # Logging
            if batch_idx % self.config['log_interval'] == 0:
                self._log_step(epoch_metrics, batch_idx)
            
            # Evaluation
            if batch_idx % self.config['eval_interval'] == 0:
                val_metrics = self.evaluate()
                self._save_checkpoint(val_metrics['val_loss'])
                self.model.train()
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.get_current_lr()
            })
        
        # Average metrics over epoch
        num_batches = len(self.train_loader)
        epoch_metrics['train_loss'] /= num_batches
        for task in epoch_metrics['train_task_losses']:
            epoch_metrics['train_task_losses'][task] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set
        
        Returns:
            Dict containing validation metrics
        """
        self.model.eval()
        val_metrics = {
            'val_loss': 0.0,
            'val_task_losses': defaultdict(float)
        }
        
        for images, targets in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.device)
            
            outputs = self.model(images)
            
            if self.criterion.single_task:
                loss = self.criterion(outputs, targets)
                task_losses = {'main': loss.item()}
            else:
                loss, task_losses = self.criterion(outputs, targets)
            
            val_metrics['val_loss'] += loss.item()
            for task, task_loss in task_losses.items():
                val_metrics['val_task_losses'][task] += task_loss
        
        # Average metrics
        num_batches = len(self.val_loader)
        val_metrics['val_loss'] /= num_batches
        for task in val_metrics['val_task_losses']:
            val_metrics['val_task_losses'][task] /= num_batches
        
        return val_metrics
    
    def train(self) -> None:
        """
        Main training loop
        """
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.evaluate()
            
            # Log epoch metrics
            self._log_epoch(epoch, train_metrics, val_metrics)
            
            # Save checkpoint if best validation loss
            self._save_checkpoint(val_metrics['val_loss'])
        
        self.logger.info("Training completed!")
    
    def _log_step(self, metrics: Dict[str, float], batch_idx: int) -> None:
        """Log metrics for current step"""
        if self.config.get('use_wandb', False):
            log_dict = {
                'train/loss': metrics['train_loss'] / (batch_idx + 1),
                'train/learning_rate': self.optimizer.get_current_lr(),
                'train/step': self.global_step
            }
            
            for task, loss in metrics['train_task_losses'].items():
                log_dict[f'train/{task}_loss'] = loss / (batch_idx + 1)
            
            wandb.log(log_dict)
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log metrics for current epoch"""
        # Console logging
        self.logger.info(
            f"Epoch {epoch+1} - Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}"
        )
        
        # W&B logging
        if self.config.get('use_wandb', False):
            log_dict = {
                'epoch': epoch + 1,
                'train/epoch_loss': train_metrics['train_loss'],
                'val/epoch_loss': val_metrics['val_loss']
            }
            
            # Log task-specific losses
            for task, loss in train_metrics['train_task_losses'].items():
                log_dict[f'train/epoch_{task}_loss'] = loss
            for task, loss in val_metrics['val_task_losses'].items():
                log_dict[f'val/epoch_{task}_loss'] = loss
            
            wandb.log(log_dict)
    
    def _save_checkpoint(self, val_loss: float) -> None:
        """Save model checkpoint if validation loss improved"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint = {
                'epoch': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'config': self.config
            }
            torch.save(
                checkpoint,
                self.checkpoint_dir / 'best_model.pth'
            )
            self.logger.info(f"Saved new best model with val_loss: {val_loss:.4f}")

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 0.05,
        'warmup_steps': 10000,
        'checkpoint_dir': 'checkpoints',
        'log_interval': 100,
        'eval_interval': 1000,
        'use_wandb': True
    }

    # Multi-task configuration
    task_configs = {
        'classification': {
            'type': TaskType.MULTICLASS,
            'alpha': 1.0,
            'gamma': 2.0,
            'weight': 1.0
        },
        'multilabel': {
            'type': TaskType.MULTILABEL,
            'alpha': 0.5,
            'gamma': 2.0,
            'weight': 0.5
        }
    }

    model = VisionTransformer(num_classes=1000)
    train_loader = ...  # Your training DataLoader
    val_loader = ...    # Your validation DataLoader 

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_configs=task_configs,
        config=config
    )

    # Start training
    trainer.train()