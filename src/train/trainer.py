"""Training script for X-ray analysis models."""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config, load_config
from utils.device import get_device, set_seed, move_to_device
from data.dataset import create_data_loaders
from models.xray_classifier import create_model, count_parameters
from losses.losses import create_loss_function
from metrics.evaluation import MedicalMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for X-ray analysis models."""
    
    def __init__(self, config: Config):
        """Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Set up device and seeding
        self.device = get_device(config.training.device)
        set_seed(config.seed)
        
        # Create directories
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.log_dir, exist_ok=True)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()
        self._setup_metrics()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
    def _setup_data(self) -> None:
        """Set up data loaders."""
        logger.info("Setting up data loaders...")
        self.data_loaders = create_data_loaders(self.config)
        
        # Calculate class weights for imbalanced datasets
        if hasattr(self.config.training, 'use_class_weights') and self.config.training.use_class_weights:
            from data.dataset import get_class_weights
            self.class_weights = get_class_weights(self.data_loaders['train'].dataset)
            logger.info(f"Class weights: {self.class_weights}")
        else:
            self.class_weights = None
    
    def _setup_model(self) -> None:
        """Set up model."""
        logger.info("Setting up model...")
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        
        # Print model info
        param_counts = count_parameters(self.model)
        logger.info(f"Model parameters: {param_counts}")
        
        # Mixed precision training
        if self.config.training.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _setup_optimizer(self) -> None:
        """Set up optimizer and scheduler."""
        logger.info("Setting up optimizer...")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        
        # Scheduler
        if self.config.training.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.1
            )
        elif self.config.training.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )
        else:
            self.scheduler = None
    
    def _setup_loss(self) -> None:
        """Set up loss function."""
        logger.info("Setting up loss function...")
        self.criterion = create_loss_function(self.config, self.class_weights)
    
    def _setup_metrics(self) -> None:
        """Set up metrics."""
        logger.info("Setting up metrics...")
        self.metrics = MedicalMetrics()
    
    def _setup_logging(self) -> None:
        """Set up logging and tensorboard."""
        # Tensorboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(self.config.training.log_dir, f"run_{timestamp}")
        self.writer = SummaryWriter(log_dir)
        
        # Log configuration
        self.writer.add_text("config", str(self.config.__dict__))
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.metrics.reset()
        
        train_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
        pbar = tqdm(self.data_loaders['train'], desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = move_to_device(batch['image'], self.device)
            labels = move_to_device(batch['label'], self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update metrics
            probabilities = torch.softmax(outputs, dim=1)
            self.metrics.update(
                outputs.argmax(dim=1), 
                labels, 
                probabilities
            )
            
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Compute epoch metrics
        epoch_metrics = self.metrics.compute_metrics()
        epoch_metrics['loss'] = train_loss / num_batches
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        val_loss = 0.0
        num_batches = len(self.data_loaders['val'])
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc="Validation"):
                # Move data to device
                images = move_to_device(batch['image'], self.device)
                labels = move_to_device(batch['label'], self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                probabilities = torch.softmax(outputs, dim=1)
                self.metrics.update(
                    outputs.argmax(dim=1), 
                    labels, 
                    probabilities
                )
                
                val_loss += loss.item()
        
        # Compute epoch metrics
        epoch_metrics = self.metrics.compute_metrics()
        epoch_metrics['loss'] = val_loss / num_batches
        
        return epoch_metrics
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.__dict__
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir, 
            'checkpoint_latest.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.training.checkpoint_dir, 
                'checkpoint_best.pth'
            )
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with metric: {self.best_metric:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('auroc', val_metrics.get('accuracy', 0)))
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Check for best model
            current_metric = val_metrics.get('auroc', val_metrics.get('accuracy', 0))
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if self.config.training.save_best and is_best:
                self.save_checkpoint(is_best=True)
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUROC: {val_metrics.get('auroc', 0):.4f}, "
                f"Val Accuracy: {val_metrics.get('accuracy', 0):.4f}"
            )
        
        logger.info("Training completed!")
        self.writer.close()
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log metrics to tensorboard.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Loss
        self.writer.add_scalar('Loss/Train', train_metrics['loss'], self.current_epoch)
        self.writer.add_scalar('Loss/Val', val_metrics['loss'], self.current_epoch)
        
        # Accuracy
        if 'accuracy' in train_metrics:
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], self.current_epoch)
        if 'accuracy' in val_metrics:
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], self.current_epoch)
        
        # AUROC
        if 'auroc' in train_metrics:
            self.writer.add_scalar('AUROC/Train', train_metrics['auroc'], self.current_epoch)
        if 'auroc' in val_metrics:
            self.writer.add_scalar('AUROC/Val', val_metrics['auroc'], self.current_epoch)
        
        # Learning rate
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.current_epoch)


def main():
    """Main training function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_path = "configs/default.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    config = load_config(config_path)
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
