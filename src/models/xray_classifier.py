"""Advanced models for X-ray analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class XrayClassifier(nn.Module):
    """Base X-ray classifier with configurable backbone.
    
    Supports ResNet, EfficientNet, and Vision Transformer architectures.
    """
    
    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        """Initialize X-ray classifier.
        
        Args:
            architecture: Model architecture ('resnet18', 'efficientnet_b0', 'vit_tiny')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout = dropout
        self.freeze_backbone = freeze_backbone
        
        # Build backbone
        self.backbone = self._build_backbone()
        
        # Build classifier head
        self.classifier = self._build_classifier()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _build_backbone(self) -> nn.Module:
        """Build the backbone network."""
        if self.architecture.startswith("resnet"):
            return self._build_resnet()
        elif self.architecture.startswith("efficientnet"):
            return self._build_efficientnet()
        elif self.architecture.startswith("vit"):
            return self._build_vit()
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
    
    def _build_resnet(self) -> nn.Module:
        """Build ResNet backbone."""
        if self.architecture == "resnet18":
            backbone = models.resnet18(pretrained=self.pretrained)
            feature_dim = backbone.fc.in_features
        elif self.architecture == "resnet34":
            backbone = models.resnet34(pretrained=self.pretrained)
            feature_dim = backbone.fc.in_features
        elif self.architecture == "resnet50":
            backbone = models.resnet50(pretrained=self.pretrained)
            feature_dim = backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported ResNet variant: {self.architecture}")
        
        # Remove the original classifier
        backbone.fc = nn.Identity()
        
        # Store feature dimension for classifier
        self.feature_dim = feature_dim
        
        return backbone
    
    def _build_efficientnet(self) -> nn.Module:
        """Build EfficientNet backbone."""
        try:
            import torchvision.models as models
            if self.architecture == "efficientnet_b0":
                backbone = models.efficientnet_b0(pretrained=self.pretrained)
                feature_dim = backbone.classifier[1].in_features
            elif self.architecture == "efficientnet_b1":
                backbone = models.efficientnet_b1(pretrained=self.pretrained)
                feature_dim = backbone.classifier[1].in_features
            else:
                raise ValueError(f"Unsupported EfficientNet variant: {self.architecture}")
            
            # Remove the original classifier
            backbone.classifier = nn.Identity()
            
            # Store feature dimension for classifier
            self.feature_dim = feature_dim
            
            return backbone
        except AttributeError:
            logger.warning("EfficientNet not available, falling back to ResNet18")
            return self._build_resnet()
    
    def _build_vit(self) -> nn.Module:
        """Build Vision Transformer backbone."""
        try:
            import torchvision.models as models
            if self.architecture == "vit_tiny":
                backbone = models.vit_b_16(pretrained=self.pretrained)
                feature_dim = backbone.heads.head.in_features
            elif self.architecture == "vit_small":
                backbone = models.vit_b_16(pretrained=self.pretrained)
                feature_dim = backbone.heads.head.in_features
            else:
                raise ValueError(f"Unsupported ViT variant: {self.architecture}")
            
            # Remove the original classifier
            backbone.heads = nn.Identity()
            
            # Store feature dimension for classifier
            self.feature_dim = feature_dim
            
            return backbone
        except AttributeError:
            logger.warning("Vision Transformer not available, falling back to ResNet18")
            return self._build_resnet()
    
    def _build_classifier(self) -> nn.Module:
        """Build classifier head."""
        return nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_classes)
        )
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        return self.backbone(x)


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple X-ray classifiers."""
    
    def __init__(
        self,
        architectures: list,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        """Initialize ensemble classifier.
        
        Args:
            architectures: List of architecture names
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.models = nn.ModuleList([
            XrayClassifier(
                architecture=arch,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                freeze_backbone=freeze_backbone
            )
            for arch in architectures
        ])
        
        self.num_models = len(self.models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ensemble averaging.
        
        Args:
            x: Input tensor
            
        Returns:
            Averaged logits
        """
        logits = []
        for model in self.models:
            logits.append(model(x))
        
        # Average logits
        avg_logits = torch.stack(logits, dim=0).mean(dim=0)
        return avg_logits
    
    def forward_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        logits = []
        for model in self.models:
            logits.append(model(x))
        
        logits_tensor = torch.stack(logits, dim=0)  # (num_models, batch_size, num_classes)
        
        # Mean prediction
        mean_logits = logits_tensor.mean(dim=0)
        
        # Uncertainty (standard deviation)
        uncertainty = logits_tensor.std(dim=0)
        
        return {
            "logits": mean_logits,
            "uncertainty": uncertainty,
            "individual_logits": logits_tensor
        }


def create_model(config: Any) -> nn.Module:
    """Create model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    if isinstance(config.model.architecture, list):
        # Ensemble model
        model = EnsembleClassifier(
            architectures=config.model.architecture,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
            freeze_backbone=config.model.freeze_backbone
        )
    else:
        # Single model
        model = XrayClassifier(
            architecture=config.model.architecture,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
            freeze_backbone=config.model.freeze_backbone
        )
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }
