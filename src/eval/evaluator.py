"""Evaluation script for X-ray analysis models."""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional, List
import json
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config, load_config
from utils.device import get_device, move_to_device
from data.dataset import create_data_loaders
from models.xray_classifier import create_model
from metrics.evaluation import MedicalMetrics
from explainability.cam import create_explainability_analyzer

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for X-ray analysis models."""
    
    def __init__(self, config: Config, checkpoint_path: str):
        """Initialize evaluator.
        
        Args:
            config: Configuration object
            checkpoint_path: Path to model checkpoint
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        # Set up device
        self.device = get_device(config.training.device)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._load_checkpoint()
        self._setup_metrics()
        self._setup_explainability()
        
        # Create output directory
        os.makedirs(config.evaluation.output_dir, exist_ok=True)
    
    def _setup_data(self) -> None:
        """Set up data loaders."""
        logger.info("Setting up data loaders...")
        self.data_loaders = create_data_loaders(self.config)
    
    def _setup_model(self) -> None:
        """Set up model."""
        logger.info("Setting up model...")
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
    
    def _load_checkpoint(self) -> None:
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def _setup_metrics(self) -> None:
        """Set up metrics."""
        logger.info("Setting up metrics...")
        self.metrics = MedicalMetrics()
    
    def _setup_explainability(self) -> None:
        """Set up explainability analyzer."""
        logger.info("Setting up explainability analyzer...")
        self.explainability_analyzer = create_explainability_analyzer(
            self.model, 
            self.config.model.architecture
        )
    
    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        """Evaluate model on specified split.
        
        Args:
            split: Data split to evaluate on ('train', 'val', 'test')
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating on {split} split...")
        
        self.model.eval()
        self.metrics.reset()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders[split], desc=f"Evaluating {split}"):
                # Move data to device
                images = move_to_device(batch['image'], self.device)
                labels = move_to_device(batch['label'], self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if 'patient_id' in batch:
                    all_patient_ids.extend(batch['patient_id'])
                
                # Update metrics
                self.metrics.update(predictions, labels, probabilities)
        
        # Compute metrics
        results = self.metrics.compute_metrics()
        
        # Add additional information
        results.update({
            'split': split,
            'num_samples': len(all_predictions),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'patient_ids': all_patient_ids
        })
        
        return results
    
    def generate_explanations(
        self, 
        num_samples: int = 10, 
        split: str = "test"
    ) -> List[Dict[str, Any]]:
        """Generate explanations for sample images.
        
        Args:
            num_samples: Number of samples to explain
            split: Data split to use
            
        Returns:
            List of explanation results
        """
        logger.info(f"Generating explanations for {num_samples} samples from {split} split...")
        
        self.model.eval()
        explanations = []
        
        # Get random samples
        dataset = self.data_loaders[split].dataset
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        for idx in tqdm(indices, desc="Generating explanations"):
            sample = dataset[idx]
            
            # Prepare input
            image = sample['image'].unsqueeze(0).to(self.device)
            label = sample['label'].item()
            
            # Get prediction
            with torch.no_grad():
                output = self.model(image)
                prediction = output.argmax(dim=1).item()
                probability = torch.softmax(output, dim=1)[0, prediction].item()
            
            # Generate explanations
            explanation_maps = self.explainability_analyzer.analyze_sample(
                image, 
                prediction,
                methods=["gradcam", "scorecam"]
            )
            
            # Compute explanation metrics
            explanation_metrics = self.explainability_analyzer.compute_explanation_metrics(
                explanation_maps
            )
            
            explanations.append({
                'index': idx,
                'patient_id': sample.get('patient_id', f'sample_{idx}'),
                'true_label': label,
                'predicted_label': prediction,
                'confidence': probability,
                'explanation_maps': explanation_maps,
                'explanation_metrics': explanation_metrics
            })
        
        return explanations
    
    def create_evaluation_report(self, results: Dict[str, Any]) -> None:
        """Create comprehensive evaluation report.
        
        Args:
            results: Evaluation results
        """
        logger.info("Creating evaluation report...")
        
        # Create plots
        self._plot_roc_curve(results)
        self._plot_pr_curve(results)
        self._plot_confusion_matrix(results)
        self._plot_calibration_curve(results)
        
        # Save detailed results
        results_path = os.path.join(
            self.config.evaluation.output_dir, 
            'evaluation_results.json'
        )
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_json_results(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation report saved to {self.config.evaluation.output_dir}")
    
    def _plot_roc_curve(self, results: Dict[str, Any]) -> None:
        """Plot ROC curve."""
        if 'roc_curve' not in results:
            return
        
        roc_data = results['roc_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'ROC Curve (AUC = {results["auroc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.config.evaluation.output_dir, 'roc_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self, results: Dict[str, Any]) -> None:
        """Plot Precision-Recall curve."""
        if 'pr_curve' not in results:
            return
        
        pr_data = results['pr_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(pr_data['recall'], pr_data['precision'], 
                label=f'PR Curve (AUC = {results["auprc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.config.evaluation.output_dir, 'pr_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, results: Dict[str, Any]) -> None:
        """Plot confusion matrix."""
        if 'confusion_matrix' not in results:
            return
        
        cm = results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        save_path = os.path.join(self.config.evaluation.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, results: Dict[str, Any]) -> None:
        """Plot calibration curve."""
        if 'probabilities' not in results:
            return
        
        probabilities = np.array(results['probabilities'])
        labels = np.array(results['labels'])
        
        if probabilities.shape[1] == 2:
            pos_probs = probabilities[:, 1]
        else:
            pos_probs = probabilities.flatten()
        
        # Compute calibration curve
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (pos_probs > bin_lower) & (pos_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(labels[in_bin].mean())
                bin_confidences.append(pos_probs[in_bin].mean())
        
        plt.figure(figsize=(8, 6))
        plt.plot(bin_confidences, bin_accuracies, 'o-', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.config.evaluation.output_dir, 'calibration_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _prepare_json_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization."""
        json_results = {}
        
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = self._prepare_json_results(value)
            else:
                json_results[key] = value
        
        return json_results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print evaluation summary.
        
        Args:
            results: Evaluation results
        """
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Split: {results['split']}")
        print(f"Number of samples: {results['num_samples']}")
        print(f"Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"AUROC: {results.get('auroc', 0):.4f}")
        print(f"AUPRC: {results.get('auprc', 0):.4f}")
        print(f"Sensitivity: {results.get('sensitivity', 0):.4f}")
        print(f"Specificity: {results.get('specificity', 0):.4f}")
        print(f"PPV: {results.get('positive_predictive_value', 0):.4f}")
        print(f"NPV: {results.get('negative_predictive_value', 0):.4f}")
        print(f"F1 Score: {results.get('f1_score', 0):.4f}")
        print(f"Expected Calibration Error: {results.get('expected_calibration_error', 0):.4f}")
        print(f"Brier Score: {results.get('brier_score', 0):.4f}")
        print("="*50)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate X-ray analysis model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--explanations', action='store_true',
                       help='Generate explanations')
    parser.add_argument('--num_explanations', type=int, default=10,
                       help='Number of explanations to generate')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config(args.config)
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint)
    
    # Evaluate model
    results = evaluator.evaluate(args.split)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Create evaluation report
    evaluator.create_evaluation_report(results)
    
    # Generate explanations if requested
    if args.explanations:
        explanations = evaluator.generate_explanations(
            num_samples=args.num_explanations,
            split=args.split
        )
        
        # Save explanations
        explanations_path = os.path.join(
            config.evaluation.output_dir,
            'explanations.json'
        )
        
        with open(explanations_path, 'w') as f:
            json.dump(explanations, f, indent=2, default=str)
        
        logger.info(f"Explanations saved to {explanations_path}")


if __name__ == "__main__":
    main()
