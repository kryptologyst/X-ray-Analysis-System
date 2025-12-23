"""Loss functions for medical imaging tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss = -α(1-p_t)^γ * log(p_t)
    where p_t is the predicted probability for the true class.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for each class
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits of shape (N, C)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            ce_loss = ce_loss * at
        
        # Apply focal weighting
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            inputs: Predicted probabilities of shape (N, C, H, W)
            targets: Ground truth labels of shape (N, H, W)
            
        Returns:
            Dice loss value
        """
        # Convert targets to one-hot encoding
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to inputs
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Compute Dice coefficient for each class
        dice_scores = []
        for c in range(num_classes):
            input_c = inputs_soft[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (input_c * target_c).sum()
            union = input_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Return average Dice loss
        dice_loss = 1.0 - torch.stack(dice_scores).mean()
        return dice_loss


class TverskyLoss(nn.Module):
    """Tversky Loss for handling class imbalance in segmentation."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        """Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Tversky loss.
        
        Args:
            inputs: Predicted probabilities of shape (N, C, H, W)
            targets: Ground truth labels of shape (N, H, W)
            
        Returns:
            Tversky loss value
        """
        # Convert targets to one-hot encoding
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to inputs
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Compute Tversky coefficient for each class
        tversky_scores = []
        for c in range(num_classes):
            input_c = inputs_soft[:, c]
            target_c = targets_one_hot[:, c]
            
            true_positives = (input_c * target_c).sum()
            false_positives = (input_c * (1 - target_c)).sum()
            false_negatives = ((1 - input_c) * target_c).sum()
            
            tversky = (true_positives + self.smooth) / (
                true_positives + self.alpha * false_positives + 
                self.beta * false_negatives + self.smooth
            )
            tversky_scores.append(tversky)
        
        # Return average Tversky loss
        tversky_loss = 1.0 - torch.stack(tversky_scores).mean()
        return tversky_loss


class CombinedLoss(nn.Module):
    """Combined loss function for medical imaging tasks."""
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        focal_weight: float = 0.0,
        dice_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[torch.Tensor] = None
    ):
        """Initialize combined loss.
        
        Args:
            ce_weight: Weight for cross-entropy loss
            focal_weight: Weight for focal loss
            dice_weight: Weight for dice loss
            focal_gamma: Gamma parameter for focal loss
            focal_alpha: Alpha parameter for focal loss
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        
        if focal_weight > 0:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        if dice_weight > 0:
            self.dice_loss = DiceLoss()
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
            return_components: Whether to return individual loss components
            
        Returns:
            Combined loss value or dictionary with components
        """
        total_loss = 0.0
        components = {}
        
        # Cross-entropy loss
        if self.ce_weight > 0:
            ce_loss = self.ce_loss(inputs, targets)
            total_loss += self.ce_weight * ce_loss
            components["ce_loss"] = ce_loss
        
        # Focal loss
        if self.focal_weight > 0:
            focal_loss = self.focal_loss(inputs, targets)
            total_loss += self.focal_weight * focal_loss
            components["focal_loss"] = focal_loss
        
        # Dice loss (for segmentation tasks)
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(inputs, targets)
            total_loss += self.dice_weight * dice_loss
            components["dice_loss"] = dice_loss
        
        if return_components:
            components["total_loss"] = total_loss
            return components
        else:
            return total_loss


def create_loss_function(config: Any, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Create loss function from configuration.
    
    Args:
        config: Configuration object
        class_weights: Optional class weights for imbalanced datasets
        
    Returns:
        Loss function
    """
    loss_type = getattr(config.training, 'loss_type', 'cross_entropy')
    
    if loss_type == "cross_entropy":
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()
    
    elif loss_type == "focal":
        return FocalLoss(
            alpha=class_weights,
            gamma=getattr(config.training, 'focal_gamma', 2.0)
        )
    
    elif loss_type == "combined":
        return CombinedLoss(
            ce_weight=getattr(config.training, 'ce_weight', 1.0),
            focal_weight=getattr(config.training, 'focal_weight', 0.0),
            dice_weight=getattr(config.training, 'dice_weight', 0.0),
            focal_gamma=getattr(config.training, 'focal_gamma', 2.0),
            focal_alpha=class_weights
        )
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
