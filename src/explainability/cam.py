"""Explainability methods for X-ray analysis models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM)."""
    
    def __init__(self, model: nn.Module, target_layer: str):
        """Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer for activation maps
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found")
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Class index for which to generate CAM
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()


class ScoreCAM:
    """Score-weighted Class Activation Mapping (Score-CAM)."""
    
    def __init__(self, model: nn.Module, target_layer: str):
        """Initialize Score-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer for activation maps
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # Register forward hook
        self._register_hook()
    
    def _register_hook(self) -> None:
        """Register forward hook."""
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found")
        
        target_module.register_forward_hook(forward_hook)
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        class_idx: Optional[int] = None,
        device: torch.device = torch.device('cpu')
    ) -> np.ndarray:
        """Generate Score-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Class index for which to generate CAM
            device: Device to run computations on
            
        Returns:
            Score-CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass to get activations
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                class_idx = output.argmax(dim=1).item()
        
        activations = self.activations[0]  # Remove batch dimension
        input_shape = input_tensor.shape[2:]  # Height, width
        
        # Upsample activations to input size
        upsampled_activations = F.interpolate(
            activations.unsqueeze(0),
            size=input_shape,
            mode='bilinear',
            align_corners=False
        )[0]
        
        # Normalize activations to [0, 1]
        upsampled_activations = F.relu(upsampled_activations)
        upsampled_activations = upsampled_activations - upsampled_activations.min()
        upsampled_activations = upsampled_activations / upsampled_activations.max()
        
        # Generate masks and compute scores
        scores = []
        for i in range(activations.shape[0]):
            # Create mask from activation
            mask = upsampled_activations[i].cpu().numpy()
            
            # Apply mask to input
            masked_input = input_tensor.clone()
            masked_input[0, 0] = masked_input[0, 0] * torch.from_numpy(mask).to(device)
            masked_input[0, 1] = masked_input[0, 1] * torch.from_numpy(mask).to(device)
            masked_input[0, 2] = masked_input[0, 2] * torch.from_numpy(mask).to(device)
            
            # Get score for target class
            with torch.no_grad():
                masked_output = self.model(masked_input)
                score = F.softmax(masked_output, dim=1)[0, class_idx].item()
            
            scores.append(score)
        
        # Weighted combination of activation maps
        scores = torch.tensor(scores)
        cam = torch.zeros(input_shape, dtype=torch.float32)
        
        for i, score in enumerate(scores):
            cam += score * upsampled_activations[i].cpu()
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.numpy()


class IntegratedGradients:
    """Integrated Gradients for model interpretability."""
    
    def __init__(self, model: nn.Module):
        """Initialize Integrated Gradients.
        
        Args:
            model: PyTorch model
        """
        self.model = model
    
    def generate_attribution(
        self, 
        input_tensor: torch.Tensor, 
        class_idx: Optional[int] = None,
        steps: int = 50
    ) -> np.ndarray:
        """Generate integrated gradients attribution.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Class index for attribution
            steps: Number of integration steps
            
        Returns:
            Attribution map as numpy array
        """
        self.model.eval()
        
        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                class_idx = output.argmax(dim=1).item()
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps + 1).to(input_tensor.device)
        
        # Compute gradients
        gradients = []
        for alpha in alphas[1:]:  # Skip alpha=0
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            output = self.model(interpolated_input)
            score = output[0, class_idx]
            
            gradient = torch.autograd.grad(score, interpolated_input)[0]
            gradients.append(gradient)
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Compute integrated gradients
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        # Sum across channels for visualization
        attribution = integrated_gradients.sum(dim=1).squeeze()
        
        # Normalize
        attribution = attribution - attribution.min()
        attribution = attribution / attribution.max()
        
        return attribution.detach().cpu().numpy()


class ExplainabilityAnalyzer:
    """Comprehensive explainability analysis for X-ray models."""
    
    def __init__(self, model: nn.Module, target_layer: str = "backbone.layer4"):
        """Initialize explainability analyzer.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for CAM methods
        """
        self.model = model
        self.target_layer = target_layer
        
        # Initialize explainability methods
        self.gradcam = GradCAM(model, target_layer)
        self.scorecam = ScoreCAM(model, target_layer)
        self.integrated_gradients = IntegratedGradients(model)
    
    def analyze_sample(
        self, 
        input_tensor: torch.Tensor, 
        class_idx: Optional[int] = None,
        methods: List[str] = ["gradcam", "scorecam", "integrated_gradients"]
    ) -> Dict[str, np.ndarray]:
        """Analyze a single sample with multiple explainability methods.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Class index for analysis
            methods: List of methods to use
            
        Returns:
            Dictionary containing explanation maps
        """
        explanations = {}
        
        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                class_idx = output.argmax(dim=1).item()
        
        if "gradcam" in methods:
            explanations["gradcam"] = self.gradcam.generate_cam(input_tensor, class_idx)
        
        if "scorecam" in methods:
            explanations["scorecam"] = self.scorecam.generate_cam(input_tensor, class_idx)
        
        if "integrated_gradients" in methods:
            explanations["integrated_gradients"] = self.integrated_gradients.generate_attribution(
                input_tensor, class_idx
            )
        
        return explanations
    
    def visualize_explanations(
        self, 
        input_tensor: torch.Tensor,
        explanations: Dict[str, np.ndarray],
        class_names: List[str] = ["Normal", "Pneumonia"],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize explanation maps.
        
        Args:
            input_tensor: Original input image
            explanations: Dictionary of explanation maps
            class_names: Names of classes
            save_path: Path to save visualization
        """
        # Convert input tensor to numpy
        input_image = input_tensor[0].permute(1, 2, 0).cpu().numpy()
        if input_image.shape[2] == 3:
            # Convert from normalized ImageNet format
            input_image = input_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            input_image = np.clip(input_image, 0, 1)
        
        # Create subplot
        n_methods = len(explanations)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
        
        if n_methods == 0:
            axes = axes.reshape(2, 1)
        
        # Original image
        axes[0, 0].imshow(input_image, cmap='gray' if input_image.shape[2] == 1 else None)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(input_image, cmap='gray' if input_image.shape[2] == 1 else None)
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis('off')
        
        # Explanation maps
        for i, (method, explanation) in enumerate(explanations.items()):
            # Heatmap
            im1 = axes[0, i + 1].imshow(explanation, cmap='jet', alpha=0.8)
            axes[0, i + 1].set_title(f"{method.upper()} Heatmap")
            axes[0, i + 1].axis('off')
            plt.colorbar(im1, ax=axes[0, i + 1])
            
            # Overlay
            axes[1, i + 1].imshow(input_image, cmap='gray' if input_image.shape[2] == 1 else None)
            axes[1, i + 1].imshow(explanation, cmap='jet', alpha=0.4)
            axes[1, i + 1].set_title(f"{method.upper()} Overlay")
            axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compute_explanation_metrics(
        self, 
        explanations: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for explanation maps.
        
        Args:
            explanations: Dictionary of explanation maps
            
        Returns:
            Dictionary containing metrics for each method
        """
        metrics = {}
        
        for method, explanation in explanations.items():
            method_metrics = {
                "mean_activation": float(explanation.mean()),
                "max_activation": float(explanation.max()),
                "std_activation": float(explanation.std()),
                "sparsity": float((explanation < 0.1).sum() / explanation.size),
                "energy_concentration": float((explanation > 0.8).sum() / explanation.size)
            }
            metrics[method] = method_metrics
        
        return metrics


def create_explainability_analyzer(
    model: nn.Module, 
    architecture: str = "resnet18"
) -> ExplainabilityAnalyzer:
    """Create explainability analyzer for different architectures.
    
    Args:
        model: PyTorch model
        architecture: Model architecture name
        
    Returns:
        Configured explainability analyzer
    """
    # Define target layers for different architectures
    target_layers = {
        "resnet18": "backbone.layer4",
        "resnet34": "backbone.layer4", 
        "resnet50": "backbone.layer4",
        "efficientnet_b0": "backbone.features",
        "efficientnet_b1": "backbone.features",
        "vit_tiny": "backbone.encoder.layers.11",
        "vit_small": "backbone.encoder.layers.11"
    }
    
    target_layer = target_layers.get(architecture, "backbone.layer4")
    
    return ExplainabilityAnalyzer(model, target_layer)
