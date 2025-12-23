#!/usr/bin/env python3
"""
Modernized X-ray Analysis System
A research-ready AI system for chest X-ray analysis

This is a complete refactor of the original 0444.py file with:
- Modern PyTorch 2.x compatibility
- Comprehensive evaluation metrics
- Explainability features
- Production-ready structure
- Compliance and privacy scaffolding

IMPORTANT: This is for research and educational purposes only.
NOT approved for clinical use.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config, get_default_config
from utils.device import get_device, set_seed, move_to_device
from data.dataset import create_data_loaders, SyntheticXrayDataset
from models.xray_classifier import create_model, count_parameters
from losses.losses import create_loss_function
from metrics.evaluation import MedicalMetrics
from explainability.cam import create_explainability_analyzer
from utils.compliance import create_compliance_suite

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModernXrayAnalysisSystem:
    """Modern X-ray analysis system with comprehensive features."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the analysis system.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or get_default_config()
        
        # Set up device and seeding
        self.device = get_device(self.config.training.device)
        set_seed(self.config.seed)
        
        # Initialize components
        self.model = None
        self.data_loaders = None
        self.metrics = None
        self.compliance_suite = None
        
        logger.info(f"Initialized X-ray Analysis System on {self.device}")
        logger.info(f"Configuration: {self.config.model.architecture}")
    
    def setup_data(self) -> None:
        """Set up data loaders."""
        logger.info("Setting up data loaders...")
        self.data_loaders = create_data_loaders(self.config)
        
        # Log dataset information
        for split, loader in self.data_loaders.items():
            logger.info(f"{split.capitalize()} dataset: {len(loader.dataset)} samples")
    
    def setup_model(self) -> None:
        """Set up model."""
        logger.info("Setting up model...")
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        
        # Log model information
        param_counts = count_parameters(self.model)
        logger.info(f"Model parameters: {param_counts}")
        
        # Set up compliance hooks if enabled
        if hasattr(self.config, 'compliance') and self.config.compliance.enabled:
            self.compliance_suite = create_compliance_suite()
            logger.info("Compliance hooks enabled")
    
    def setup_metrics(self) -> None:
        """Set up evaluation metrics."""
        logger.info("Setting up metrics...")
        self.metrics = MedicalMetrics()
    
    def train_model(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            epochs: Number of epochs to train (uses config default if None)
            
        Returns:
            Training results
        """
        if self.model is None or self.data_loaders is None:
            raise ValueError("Model and data loaders must be set up first")
        
        epochs = epochs or self.config.training.epochs
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Set up training components
        criterion = create_loss_function(self.config)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        
        # Training loop
        self.model.train()
        training_results = {
            'epochs': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(criterion)
            
            # Store results
            training_results['epochs'].append(epoch + 1)
            training_results['train_loss'].append(train_loss)
            training_results['train_accuracy'].append(train_acc)
            training_results['val_loss'].append(val_loss)
            training_results['val_accuracy'].append(val_acc)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        
        logger.info("Training completed!")
        return training_results
    
    def _train_epoch(self, optimizer, criterion) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.data_loaders['train']:
            images = move_to_device(batch['image'], self.device)
            labels = move_to_device(batch['label'], self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(self.data_loaders['train']), correct / total
    
    def _validate_epoch(self, criterion) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.data_loaders['val']:
                images = move_to_device(batch['image'], self.device)
                labels = move_to_device(batch['label'], self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(self.data_loaders['val']), correct / total
    
    def evaluate_model(self, split: str = 'test') -> Dict[str, Any]:
        """Evaluate the model.
        
        Args:
            split: Data split to evaluate on
            
        Returns:
            Evaluation results
        """
        if self.model is None or self.data_loaders is None:
            raise ValueError("Model and data loaders must be set up first")
        
        logger.info(f"Evaluating model on {split} split...")
        
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in self.data_loaders[split]:
                images = move_to_device(batch['image'], self.device)
                labels = move_to_device(batch['label'], self.device)
                
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                
                self.metrics.update(predictions, labels, probabilities)
        
        # Compute comprehensive metrics
        results = self.metrics.compute_metrics()
        
        # Log key metrics
        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy: {results.get('accuracy', 0):.4f}")
        logger.info(f"  AUROC: {results.get('auroc', 0):.4f}")
        logger.info(f"  AUPRC: {results.get('auprc', 0):.4f}")
        logger.info(f"  Sensitivity: {results.get('sensitivity', 0):.4f}")
        logger.info(f"  Specificity: {results.get('specificity', 0):.4f}")
        
        return results
    
    def generate_explanations(self, num_samples: int = 5) -> Dict[str, Any]:
        """Generate explanations for sample images.
        
        Args:
            num_samples: Number of samples to explain
            
        Returns:
            Explanation results
        """
        if self.model is None:
            raise ValueError("Model must be set up first")
        
        logger.info(f"Generating explanations for {num_samples} samples...")
        
        # Create explainability analyzer
        analyzer = create_explainability_analyzer(
            self.model, 
            self.config.model.architecture
        )
        
        explanations = []
        dataset = self.data_loaders['test'].dataset
        
        # Get random samples
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        for idx in indices:
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            label = sample['label'].item()
            
            # Get prediction
            with torch.no_grad():
                output = self.model(image)
                prediction = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, prediction].item()
            
            # Generate explanations
            explanation_maps = analyzer.analyze_sample(
                image, 
                prediction,
                methods=["gradcam", "scorecam"]
            )
            
            explanations.append({
                'index': idx,
                'true_label': label,
                'predicted_label': prediction,
                'confidence': confidence,
                'explanation_maps': explanation_maps
            })
        
        return {'explanations': explanations}
    
    def plot_training_results(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot training results.
        
        Args:
            results: Training results dictionary
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(results['epochs'], results['train_loss'], label='Train Loss')
        ax1.plot(results['epochs'], results['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(results['epochs'], results['train_accuracy'], label='Train Accuracy')
        ax2.plot(results['epochs'], results['val_accuracy'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training plot saved to {save_path}")
        
        plt.show()
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline.
        
        Returns:
            Complete analysis results
        """
        logger.info("Starting complete analysis pipeline...")
        
        # Setup
        self.setup_data()
        self.setup_model()
        self.setup_metrics()
        
        # Training
        training_results = self.train_model()
        
        # Evaluation
        evaluation_results = self.evaluate_model()
        
        # Explanations
        explanation_results = self.generate_explanations()
        
        # Compile results
        complete_results = {
            'training': training_results,
            'evaluation': evaluation_results,
            'explanations': explanation_results,
            'model_info': {
                'architecture': self.config.model.architecture,
                'parameters': count_parameters(self.model),
                'device': str(self.device)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Complete analysis pipeline finished!")
        return complete_results


def main():
    """Main function for running the analysis system."""
    parser = argparse.ArgumentParser(description='Modern X-ray Analysis System')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='complete',
                       choices=['train', 'eval', 'explain', 'complete'],
                       help='Mode to run')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of explanation samples')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using default")
        config = get_default_config()
    
    # Create analysis system
    system = ModernXrayAnalysisSystem(config)
    
    # Run based on mode
    if args.mode == 'complete':
        results = system.run_complete_analysis()
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")
    
    elif args.mode == 'train':
        system.setup_data()
        system.setup_model()
        results = system.train_model(args.epochs)
        system.plot_training_results(results)
    
    elif args.mode == 'eval':
        system.setup_data()
        system.setup_model()
        results = system.evaluate_model()
        print(f"Evaluation Results: {results}")
    
    elif args.mode == 'explain':
        system.setup_data()
        system.setup_model()
        results = system.generate_explanations(args.samples)
        print(f"Explanation Results: {results}")


if __name__ == "__main__":
    # Print disclaimer
    print("="*60)
    print("X-RAY ANALYSIS SYSTEM - RESEARCH DEMO")
    print("="*60)
    print("IMPORTANT: This is for research and educational purposes only.")
    print("NOT approved for clinical use.")
    print("NOT intended for diagnosis or treatment decisions.")
    print("="*60)
    print()
    
    main()
