"""Comprehensive evaluation metrics for medical imaging tasks."""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class MedicalMetrics:
    """Comprehensive medical imaging evaluation metrics."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize metrics calculator.
        
        Args:
            class_names: Names of classes for reporting
        """
        self.class_names = class_names or ["Normal", "Abnormal"]
        self.reset()
    
    def reset(self) -> None:
        """Reset all stored predictions and labels."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, probabilities: Optional[torch.Tensor] = None) -> None:
        """Update stored predictions and labels.
        
        Args:
            predictions: Predicted class labels
            labels: Ground truth labels
            probabilities: Predicted class probabilities
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if not self.predictions:
            logger.warning("No predictions available for metric computation")
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._compute_basic_metrics(predictions, labels))
        
        # ROC and PR metrics
        if len(self.probabilities) > 0:
            probabilities = np.array(self.probabilities)
            metrics.update(self._compute_roc_pr_metrics(probabilities, labels))
        
        # Confusion matrix metrics
        metrics.update(self._compute_confusion_matrix_metrics(predictions, labels))
        
        # Calibration metrics
        if len(self.probabilities) > 0:
            metrics.update(self._compute_calibration_metrics(probabilities, labels))
        
        return metrics
    
    def _compute_basic_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics."""
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
            "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(labels, predictions, average="weighted", zero_division=0),
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            class_precision = precision_score(labels, predictions, labels=[i], average="micro", zero_division=0)
            class_recall = recall_score(labels, predictions, labels=[i], average="micro", zero_division=0)
            class_f1 = f1_score(labels, predictions, labels=[i], average="micro", zero_division=0)
            
            metrics.update({
                f"{class_name.lower()}_precision": class_precision,
                f"{class_name.lower()}_recall": class_recall,
                f"{class_name.lower()}_f1": class_f1,
            })
        
        return metrics
    
    def _compute_roc_pr_metrics(self, probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute ROC and PR curve metrics."""
        metrics = {}
        
        # For binary classification, use positive class probabilities
        if probabilities.shape[1] == 2:
            pos_probs = probabilities[:, 1]
        else:
            pos_probs = probabilities.flatten()
        
        # ROC metrics
        try:
            auroc = roc_auc_score(labels, pos_probs)
            metrics["auroc"] = auroc
            
            # ROC curve for plotting
            fpr, tpr, roc_thresholds = roc_curve(labels, pos_probs)
            metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds}
            
        except ValueError as e:
            logger.warning(f"Could not compute AUROC: {e}")
            metrics["auroc"] = 0.0
        
        # PR metrics
        try:
            auprc = average_precision_score(labels, pos_probs)
            metrics["auprc"] = auprc
            
            # PR curve for plotting
            precision, recall, pr_thresholds = precision_recall_curve(labels, pos_probs)
            metrics["pr_curve"] = {"precision": precision, "recall": recall, "thresholds": pr_thresholds}
            
        except ValueError as e:
            logger.warning(f"Could not compute AUPRC: {e}")
            metrics["auprc"] = 0.0
        
        return metrics
    
    def _compute_confusion_matrix_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute confusion matrix and related metrics."""
        cm = confusion_matrix(labels, predictions)
        
        # Basic confusion matrix metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "confusion_matrix": cm,
        }
        
        # Sensitivity (Recall) and Specificity
        if tp + fn > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0.0
        
        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0
        
        metrics.update({
            "sensitivity": sensitivity,
            "specificity": specificity,
        })
        
        # Positive and Negative Predictive Values
        if tp + fp > 0:
            ppv = tp / (tp + fp)
        else:
            ppv = 0.0
        
        if tn + fn > 0:
            npv = tn / (tn + fn)
        else:
            npv = 0.0
        
        metrics.update({
            "positive_predictive_value": ppv,
            "negative_predictive_value": npv,
        })
        
        return metrics
    
    def _compute_calibration_metrics(self, probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute calibration metrics."""
        metrics = {}
        
        # For binary classification, use positive class probabilities
        if probabilities.shape[1] == 2:
            pos_probs = probabilities[:, 1]
        else:
            pos_probs = probabilities.flatten()
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(pos_probs, labels)
        metrics["expected_calibration_error"] = ece
        
        # Brier Score
        brier_score = np.mean((pos_probs - labels) ** 2)
        metrics["brier_score"] = brier_score
        
        return metrics
    
    def _compute_ece(self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """Plot ROC curve."""
        if "roc_curve" not in self.compute_metrics():
            logger.warning("No ROC curve data available")
            return
        
        metrics = self.compute_metrics()
        roc_data = metrics["roc_curve"]
        
        plt.figure(figsize=(8, 6))
        plt.plot(roc_data["fpr"], roc_data["tpr"], 
                label=f'ROC Curve (AUC = {metrics["auroc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pr_curve(self, save_path: Optional[str] = None) -> None:
        """Plot Precision-Recall curve."""
        if "pr_curve" not in self.compute_metrics():
            logger.warning("No PR curve data available")
            return
        
        metrics = self.compute_metrics()
        pr_data = metrics["pr_curve"]
        
        plt.figure(figsize=(8, 6))
        plt.plot(pr_data["recall"], pr_data["precision"], 
                label=f'PR Curve (AUC = {metrics["auprc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        metrics = self.compute_metrics()
        cm = metrics["confusion_matrix"]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, save_path: Optional[str] = None) -> None:
        """Plot calibration curve."""
        if not self.probabilities:
            logger.warning("No probability data available for calibration plot")
            return
        
        probabilities = np.array(self.probabilities)
        labels = np.array(self.labels)
        
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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def compute_confidence_intervals(
    metric_values: List[float], 
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence intervals for metrics.
    
    Args:
        metric_values: List of metric values (e.g., from cross-validation)
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    mean_val = np.mean(metric_values)
    std_val = np.std(metric_values)
    
    # Use t-distribution for small samples
    if len(metric_values) < 30:
        t_val = stats.t.ppf((1 + confidence_level) / 2, len(metric_values) - 1)
        margin_error = t_val * (std_val / np.sqrt(len(metric_values)))
    else:
        # Use normal distribution for large samples
        z_val = stats.norm.ppf((1 + confidence_level) / 2)
        margin_error = z_val * (std_val / np.sqrt(len(metric_values)))
    
    return mean_val - margin_error, mean_val + margin_error
