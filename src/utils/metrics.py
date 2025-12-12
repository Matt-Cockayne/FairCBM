"""
Simplified metrics for FairCBM.
Provides basic classification metrics without heavy dependencies.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import Dict, Union
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        labels: True binary labels
        predictions: Predicted probabilities (0-1 range)
        threshold: Classification threshold
        
    Returns:
        Dictionary with accuracy, precision, recall, F1, AUC
    """
    # Ensure numpy arrays
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Flatten if needed
    labels = labels.flatten()
    predictions = predictions.flatten()
    
    # Binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    metrics = {}
    
    try:
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, binary_preds)
        metrics['precision'] = precision_score(labels, binary_preds, zero_division=0)
        metrics['recall'] = recall_score(labels, binary_preds, zero_division=0)
        metrics['f1'] = f1_score(labels, binary_preds, zero_division=0)
        
        # AUC (using probabilities)
        metrics['auc'] = roc_auc_score(labels, predictions)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Sensitivity and specificity
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
    except Exception as e:
        # Fallback for edge cases
        print(f"Warning: Error computing metrics: {e}")
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.5,
            'sensitivity': 0.0,
            'specificity': 0.0
        }
    
    return metrics


def compute_group_metrics(labels: np.ndarray, 
                         predictions: np.ndarray, 
                         groups: np.ndarray,
                         threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics stratified by group (e.g., Fitzpatrick types).
    
    Args:
        labels: True binary labels
        predictions: Predicted probabilities
        groups: Group identifiers
        threshold: Classification threshold
        
    Returns:
        Dictionary mapping group_id -> metrics
    """
    unique_groups = np.unique(groups)
    group_metrics = {}
    
    for group_id in unique_groups:
        mask = (groups == group_id)
        if mask.sum() > 0:
            group_labels = labels[mask]
            group_preds = predictions[mask]
            group_metrics[f'group_{int(group_id)}'] = compute_metrics(
                group_labels, group_preds, threshold
            )
    
    return group_metrics


def aggregate_metrics(metrics_list: list) -> Dict[str, float]:
    """
    Aggregate metrics across multiple batches or runs.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary with mean values for each metric
    """
    if not metrics_list:
        return {}
    
    # Get all metric keys
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    # Compute means
    aggregated = {}
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    return aggregated
