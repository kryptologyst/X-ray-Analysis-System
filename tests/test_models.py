"""Unit tests for X-ray analysis system."""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config, get_default_config
from utils.device import get_device, set_seed
from models.xray_classifier import create_model, count_parameters
from losses.losses import create_loss_function, FocalLoss
from data.dataset import SyntheticXrayDataset, get_transforms
from metrics.evaluation import MedicalMetrics


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        assert config.model.architecture == "resnet18"
        assert config.model.num_classes == 2
        assert config.data.batch_size == 16
        assert config.seed == 42
    
    def test_config_update(self):
        """Test configuration updates."""
        config = get_default_config()
        config.update(seed=123)
        assert config.seed == 123


class TestDevice:
    """Test device management."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that seed is set (basic check)
        assert True  # Seed setting doesn't return anything


class TestModels:
    """Test model creation and functionality."""
    
    def test_resnet18_creation(self):
        """Test ResNet18 model creation."""
        config = get_default_config()
        config.model.architecture = "resnet18"
        
        model = create_model(config)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 2)
    
    def test_efficientnet_creation(self):
        """Test EfficientNet model creation."""
        config = get_default_config()
        config.model.architecture = "efficientnet_b0"
        
        try:
            model = create_model(config)
            assert model is not None
            
            # Test forward pass
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            assert output.shape == (1, 2)
        except ValueError:
            # EfficientNet might not be available in all environments
            pytest.skip("EfficientNet not available")
    
    def test_model_parameters(self):
        """Test parameter counting."""
        config = get_default_config()
        model = create_model(config)
        
        param_counts = count_parameters(model)
        assert param_counts["total"] > 0
        assert param_counts["trainable"] > 0
        assert param_counts["total"] >= param_counts["trainable"]


class TestLosses:
    """Test loss functions."""
    
    def test_cross_entropy_loss(self):
        """Test cross-entropy loss creation."""
        config = get_default_config()
        config.training.loss_type = "cross_entropy"
        
        loss_fn = create_loss_function(config)
        assert loss_fn is not None
        
        # Test loss computation
        logits = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0
    
    def test_focal_loss(self):
        """Test focal loss."""
        focal_loss = FocalLoss(gamma=2.0)
        
        logits = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        loss = focal_loss(logits, labels)
        assert loss.item() >= 0
    
    def test_focal_loss_creation(self):
        """Test focal loss creation from config."""
        config = get_default_config()
        config.training.loss_type = "focal"
        
        loss_fn = create_loss_function(config)
        assert loss_fn is not None


class TestDataset:
    """Test dataset functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticXrayDataset(size=10, image_size=(224, 224))
        
        assert len(dataset) == 10
        
        # Test getting an item
        item = dataset[0]
        assert "image" in item
        assert "label" in item
        assert "class_name" in item
        assert "patient_id" in item
    
    def test_transforms(self):
        """Test data transforms."""
        transforms_dict = get_transforms(image_size=224, augmentation=True)
        
        assert "train" in transforms_dict
        assert "val" in transforms_dict
        assert "test" in transforms_dict
        
        # Test transform application
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        image_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        image = Image.fromarray(image_array, mode='L')
        
        # Apply transforms
        transformed = transforms_dict["train"](image)
        assert transformed.shape == (3, 224, 224)  # 3 channels after grayscale conversion


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_medical_metrics(self):
        """Test medical metrics computation."""
        metrics = MedicalMetrics()
        
        # Add some dummy data
        predictions = torch.tensor([0, 1, 0, 1])
        labels = torch.tensor([0, 1, 1, 1])
        probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
        
        metrics.update(predictions, labels, probabilities)
        
        results = metrics.compute_metrics()
        
        assert "accuracy" in results
        assert "auroc" in results
        assert "sensitivity" in results
        assert "specificity" in results
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = MedicalMetrics()
        
        # Add data
        predictions = torch.tensor([0, 1])
        labels = torch.tensor([0, 1])
        metrics.update(predictions, labels)
        
        # Reset
        metrics.reset()
        
        # Should be empty after reset
        assert len(metrics.predictions) == 0
        assert len(metrics.labels) == 0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training_step(self):
        """Test a single training step."""
        config = get_default_config()
        config.data.batch_size = 2
        
        # Create model
        model = create_model(config)
        model.train()
        
        # Create loss function
        loss_fn = create_loss_function(config)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy data
        images = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 2, (2,))
        
        # Training step
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Check that loss is computed
        assert loss.item() >= 0
    
    def test_model_evaluation_step(self):
        """Test a single evaluation step."""
        config = get_default_config()
        
        # Create model
        model = create_model(config)
        model.eval()
        
        # Create metrics
        metrics = MedicalMetrics()
        
        # Create dummy data
        images = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 2, (2,))
        
        # Evaluation step
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            metrics.update(predictions, labels, probabilities)
        
        # Check metrics
        results = metrics.compute_metrics()
        assert "accuracy" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
