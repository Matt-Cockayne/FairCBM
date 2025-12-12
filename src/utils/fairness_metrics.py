#!/usr/bin/env python3
"""
Fairness Metrics for Concept Bottleneck Models

Implements comprehensive fairness evaluation metrics including:
- Demographic Parity
- Equalized Odds
- Calibration by Group
- Worst-Group Performance
- Aequitas Integration

Author: Matt Cockayne
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def compute_demographic_parity(predictions: np.ndarray, 
                              groups: np.ndarray,
                              threshold: float = 0.5) -> Dict[str, Any]:
    """
    Demographic Parity: P(Ŷ=1 | A=a) should be equal across groups.
    
    Measures whether positive prediction rates are equal across groups,
    regardless of ground truth labels. Ensures equal treatment.
    
    Args:
        predictions: Model probability predictions [n_samples]
        groups: Group membership labels (e.g., Fitzpatrick types 0-5) [n_samples]
        threshold: Binary classification threshold (default: 0.5)
    
    Returns:
        {
            'group_positive_rates': {group_id: positive_rate},
            'max_disparity': max_rate - min_rate,
            'disparate_impact_ratio': min_rate / max_rate (80% rule),
            'statistical_parity_difference': max_disparity,
            'passes_80_percent_rule': bool (disparate_impact >= 0.8)
        }
    
    Reference:
        Feldman et al. (2015). "Certifying and removing disparate impact"
    """
    binary_preds = (predictions >= threshold).astype(int)
    
    group_rates = {}
    group_counts = {}
    
    for group in np.unique(groups):
        group_mask = (groups == group)
        group_counts[int(group)] = group_mask.sum()
        if group_mask.sum() > 0:
            group_rates[int(group)] = binary_preds[group_mask].mean()
        else:
            group_rates[int(group)] = 0.0
    
    rates = list(group_rates.values())
    if len(rates) == 0:
        return {
            'group_positive_rates': {},
            'group_counts': {},
            'max_disparity': 0.0,
            'disparate_impact_ratio': 1.0,
            'statistical_parity_difference': 0.0,
            'passes_80_percent_rule': True
        }
    
    max_rate = max(rates)
    min_rate = min(rates)
    
    # Disparate impact ratio (80% rule)
    disparate_impact = min_rate / max_rate if max_rate > 0 else 0.0
    passes_80_rule = disparate_impact >= 0.8
    
    return {
        'group_positive_rates': group_rates,
        'group_counts': group_counts,
        'max_disparity': max_rate - min_rate,
        'disparate_impact_ratio': disparate_impact,
        'statistical_parity_difference': max_rate - min_rate,
        'passes_80_percent_rule': passes_80_rule
    }


def compute_equalized_odds(predictions: np.ndarray,
                          labels: np.ndarray,
                          groups: np.ndarray,
                          threshold: float = 0.5) -> Dict[str, Any]:
    """
    Equalized Odds: P(Ŷ=1 | Y=y, A=a) should be equal across groups for y ∈ {0, 1}.
    
    Ensures equal true positive rates (TPR) and false positive rates (FPR)
    across groups. Stronger than demographic parity as it conditions on true labels.
    
    Args:
        predictions: Model probability predictions [n_samples]
        labels: True binary labels [n_samples]
        groups: Group membership labels [n_samples]
        threshold: Binary classification threshold (default: 0.5)
    
    Returns:
        {
            'group_tpr': {group_id: true_positive_rate},
            'group_fpr': {group_id: false_positive_rate},
            'group_tnr': {group_id: true_negative_rate},
            'group_fnr': {group_id: false_negative_rate},
            'tpr_disparity': max_tpr - min_tpr,
            'fpr_disparity': max_fpr - min_fpr,
            'equalized_odds_difference': (tpr_disparity + fpr_disparity) / 2,
            'max_disparity': max(tpr_disparity, fpr_disparity)
        }
    
    Reference:
        Hardt et al. (2016). "Equality of Opportunity in Supervised Learning"
    """
    binary_preds = (predictions >= threshold).astype(int)
    
    group_tpr = {}
    group_fpr = {}
    group_tnr = {}
    group_fnr = {}
    
    for group in np.unique(groups):
        group_mask = (groups == group)
        group_preds = binary_preds[group_mask]
        group_labels = labels[group_mask]
        
        if len(group_labels) == 0:
            group_tpr[int(group)] = 0.0
            group_fpr[int(group)] = 0.0
            group_tnr[int(group)] = 0.0
            group_fnr[int(group)] = 0.0
            continue
        
        # True Positive Rate (Sensitivity, Recall)
        positives = (group_labels == 1)
        if positives.sum() > 0:
            group_tpr[int(group)] = group_preds[positives].mean()
            group_fnr[int(group)] = 1.0 - group_tpr[int(group)]
        else:
            group_tpr[int(group)] = 0.0
            group_fnr[int(group)] = 0.0
        
        # False Positive Rate
        negatives = (group_labels == 0)
        if negatives.sum() > 0:
            group_fpr[int(group)] = group_preds[negatives].mean()
            group_tnr[int(group)] = 1.0 - group_fpr[int(group)]
        else:
            group_fpr[int(group)] = 0.0
            group_tnr[int(group)] = 0.0
    
    tpr_values = list(group_tpr.values())
    fpr_values = list(group_fpr.values())
    
    if len(tpr_values) == 0:
        tpr_disparity = 0.0
        fpr_disparity = 0.0
    else:
        tpr_disparity = max(tpr_values) - min(tpr_values)
        fpr_disparity = max(fpr_values) - min(fpr_values)
    
    return {
        'group_tpr': group_tpr,
        'group_fpr': group_fpr,
        'group_tnr': group_tnr,
        'group_fnr': group_fnr,
        'tpr_disparity': tpr_disparity,
        'fpr_disparity': fpr_disparity,
        'equalized_odds_difference': (tpr_disparity + fpr_disparity) / 2,
        'max_disparity': max(tpr_disparity, fpr_disparity) if len(tpr_values) > 0 else 0.0
    }


def compute_equal_opportunity(predictions: np.ndarray,
                             labels: np.ndarray,
                             groups: np.ndarray,
                             threshold: float = 0.5) -> Dict[str, Any]:
    """
    Equal Opportunity: P(Ŷ=1 | Y=1, A=a) should be equal across groups.
    
    Relaxed version of equalized odds - only requires equal TPR across groups.
    Ensures qualified individuals have equal chance of positive prediction.
    
    Args:
        predictions: Model probability predictions [n_samples]
        labels: True binary labels [n_samples]
        groups: Group membership labels [n_samples]
        threshold: Binary classification threshold
    
    Returns:
        {
            'group_tpr': {group_id: true_positive_rate},
            'tpr_disparity': max_tpr - min_tpr,
            'equal_opportunity_difference': tpr_disparity
        }
    """
    eq_odds = compute_equalized_odds(predictions, labels, groups, threshold)
    
    return {
        'group_tpr': eq_odds['group_tpr'],
        'tpr_disparity': eq_odds['tpr_disparity'],
        'equal_opportunity_difference': eq_odds['tpr_disparity']
    }


def compute_calibration_by_group(predictions: np.ndarray,
                                 labels: np.ndarray,
                                 groups: np.ndarray,
                                 n_bins: int = 10) -> Dict[str, Any]:
    """
    Calibration quality per group via Expected Calibration Error (ECE).
    
    A model is calibrated if predictions match empirical probabilities.
    Group-wise calibration ensures reliability across demographic groups.
    
    Args:
        predictions: Model probability predictions [n_samples]
        labels: True binary labels [n_samples]
        groups: Group membership labels [n_samples]
        n_bins: Number of bins for calibration (default: 10)
    
    Returns:
        {
            'group_ece': {group_id: expected_calibration_error},
            'group_mce': {group_id: max_calibration_error},
            'calibration_disparity': max_ece - min_ece,
            'mean_ece': mean(group_ece)
        }
    
    Reference:
        Naeini et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
    """
    group_ece = {}
    group_mce = {}
    group_calibration_curves = {}
    
    for group in np.unique(groups):
        group_mask = (groups == group)
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        
        if len(group_labels) == 0:
            group_ece[int(group)] = 0.0
            group_mce[int(group)] = 0.0
            continue
        
        # Compute ECE
        ece, mce, calibration_curve = _compute_ece(group_preds, group_labels, n_bins)
        group_ece[int(group)] = ece
        group_mce[int(group)] = mce
        group_calibration_curves[int(group)] = calibration_curve
    
    ece_values = list(group_ece.values())
    if len(ece_values) == 0:
        calibration_disparity = 0.0
        mean_ece = 0.0
    else:
        calibration_disparity = max(ece_values) - min(ece_values)
        mean_ece = np.mean(ece_values)
    
    return {
        'group_ece': group_ece,
        'group_mce': group_mce,
        'group_calibration_curves': group_calibration_curves,
        'calibration_disparity': calibration_disparity,
        'mean_ece': mean_ece
    }


def _compute_ece(predictions: np.ndarray, 
                 labels: np.ndarray,
                 n_bins: int = 10) -> Tuple[float, float, Dict[str, List]]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        predictions: Probability predictions [n_samples]
        labels: True binary labels [n_samples]
        n_bins: Number of bins
    
    Returns:
        ece: Expected calibration error
        mce: Maximum calibration error
        calibration_curve: {bin_edges, bin_accs, bin_confs, bin_counts}
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
    
    calibration_curve = {
        'bin_edges': bin_boundaries.tolist(),
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs,
        'bin_counts': bin_counts
    }
    
    return ece, mce, calibration_curve


def compute_worst_group_performance(predictions: np.ndarray,
                                   labels: np.ndarray,
                                   groups: np.ndarray,
                                   threshold: float = 0.5) -> Dict[str, Any]:
    """
    Identify worst-performing group and compute performance gap.
    
    Min-max fairness criterion: maximize worst-group performance.
    
    Args:
        predictions: Model probability predictions [n_samples]
        labels: True binary labels [n_samples]
        groups: Group membership labels [n_samples]
        threshold: Binary classification threshold
    
    Returns:
        {
            'group_f1': {group_id: f1_score},
            'group_precision': {group_id: precision},
            'group_recall': {group_id: recall},
            'group_accuracy': {group_id: accuracy},
            'worst_group': group_id with minimum F1,
            'worst_group_f1': min(group_f1),
            'best_group_f1': max(group_f1),
            'performance_gap': max_f1 - min_f1,
            'performance_gap_ratio': min_f1 / max_f1
        }
    """
    binary_preds = (predictions >= threshold).astype(int)
    
    group_f1 = {}
    group_precision = {}
    group_recall = {}
    group_accuracy = {}
    
    for group in np.unique(groups):
        group_mask = (groups == group)
        group_preds_i = binary_preds[group_mask]
        group_labels_i = labels[group_mask]
        
        if len(group_labels_i) == 0:
            group_f1[int(group)] = 0.0
            group_precision[int(group)] = 0.0
            group_recall[int(group)] = 0.0
            group_accuracy[int(group)] = 0.0
            continue
        
        group_f1[int(group)] = f1_score(group_labels_i, group_preds_i, zero_division=0)
        group_precision[int(group)] = precision_score(group_labels_i, group_preds_i, zero_division=0)
        group_recall[int(group)] = recall_score(group_labels_i, group_preds_i, zero_division=0)
        group_accuracy[int(group)] = (group_preds_i == group_labels_i).mean()
    
    f1_values = list(group_f1.values())
    if len(f1_values) == 0:
        return {
            'group_f1': {},
            'group_precision': {},
            'group_recall': {},
            'group_accuracy': {},
            'worst_group': None,
            'worst_group_f1': 0.0,
            'best_group_f1': 0.0,
            'performance_gap': 0.0,
            'performance_gap_ratio': 0.0
        }
    
    worst_group = min(group_f1, key=group_f1.get)
    best_group = max(group_f1, key=group_f1.get)
    
    min_f1 = group_f1[worst_group]
    max_f1 = group_f1[best_group]
    
    return {
        'group_f1': group_f1,
        'group_precision': group_precision,
        'group_recall': group_recall,
        'group_accuracy': group_accuracy,
        'worst_group': worst_group,
        'best_group': best_group,
        'worst_group_f1': min_f1,
        'best_group_f1': max_f1,
        'performance_gap': max_f1 - min_f1,
        'performance_gap_ratio': min_f1 / max_f1 if max_f1 > 0 else 0.0
    }


def generate_aequitas_report(predictions: np.ndarray,
                            labels: np.ndarray,
                            groups: np.ndarray,
                            threshold: float = 0.5,
                            output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive fairness audit using Aequitas toolkit.
    
    Aequitas provides industry-standard fairness metrics and bias analysis.
    
    Args:
        predictions: Model probability predictions [n_samples]
        labels: True binary labels [n_samples]
        groups: Group membership labels [n_samples]
        threshold: Binary classification threshold
        output_path: Optional path to save report CSV
    
    Returns:
        {
            'aequitas_crosstabs': Group-level confusion matrix metrics,
            'aequitas_bias': Disparity ratios relative to reference group,
            'aequitas_fairness': Binary fairness flags per metric
        }
    
    Reference:
        Saleiro et al. (2019). "Aequitas: A Bias and Fairness Audit Toolkit"
    """
    try:
        from aequitas.group import Group
        from aequitas.bias import Bias
        from aequitas.fairness import Fairness
    except ImportError:
        print("Warning: Aequitas not installed. Install with: pip install aequitas")
        return {
            'aequitas_crosstabs': None,
            'aequitas_bias': None,
            'aequitas_fairness': None,
            'error': 'Aequitas not installed'
        }
    
    binary_preds = (predictions >= threshold).astype(int)
    
    # Create Aequitas-compatible DataFrame
    df = pd.DataFrame({
        'entity_id': np.arange(len(predictions)),
        'score': predictions,
        'label_value': labels.astype(int),
        'pred_value': binary_preds,
        'group': groups.astype(int)
    })
    
    try:
        # Compute group metrics
        g = Group()
        xtab, _ = g.get_crosstabs(df, attr_cols=['group'])
        
        # Compute bias metrics (disparity ratios)
        b = Bias()
        # Use most common group as reference (can be customized)
        ref_group = int(pd.Series(groups).value_counts().idxmax())
        bdf = b.get_disparity(xtab, original_df=df, ref_groups_dict={'group': ref_group})
        
        # Compute fairness flags
        f = Fairness()
        fdf = f.get_group_value_fairness(bdf)
        
        if output_path:
            fdf.to_csv(output_path, index=False)
            print(f"Aequitas report saved to {output_path}")
        
        return {
            'aequitas_crosstabs': xtab.to_dict(),
            'aequitas_bias': bdf.to_dict(),
            'aequitas_fairness': fdf.to_dict(),
            'reference_group': ref_group
        }
    except Exception as e:
        print(f"Error generating Aequitas report: {e}")
        return {
            'aequitas_crosstabs': None,
            'aequitas_bias': None,
            'aequitas_fairness': None,
            'error': str(e)
        }


def compute_all_fairness_metrics(predictions: np.ndarray,
                                 labels: np.ndarray,
                                 groups: np.ndarray,
                                 threshold: float = 0.5,
                                 n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute all fairness metrics in one call for convenience.
    
    Args:
        predictions: Model probability predictions [n_samples]
        labels: True binary labels [n_samples]
        groups: Group membership labels [n_samples]
        threshold: Binary classification threshold
        n_bins: Number of bins for calibration
    
    Returns:
        Dictionary with all fairness metrics
    """
    return {
        'demographic_parity': compute_demographic_parity(predictions, groups, threshold),
        'equalized_odds': compute_equalized_odds(predictions, labels, groups, threshold),
        'equal_opportunity': compute_equal_opportunity(predictions, labels, groups, threshold),
        'calibration': compute_calibration_by_group(predictions, labels, groups, n_bins),
        'worst_group': compute_worst_group_performance(predictions, labels, groups, threshold),
        'threshold_used': threshold
    }


def compute_fairness_summary(fairness_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key fairness summary statistics from full metrics.
    
    Args:
        fairness_metrics: Output from compute_all_fairness_metrics
    
    Returns:
        Summary with key fairness indicators
    """
    summary = {
        # Demographic Parity
        'demographic_parity_difference': fairness_metrics['demographic_parity']['statistical_parity_difference'],
        'disparate_impact_ratio': fairness_metrics['demographic_parity']['disparate_impact_ratio'],
        'passes_80_percent_rule': fairness_metrics['demographic_parity']['passes_80_percent_rule'],
        
        # Equalized Odds
        'equalized_odds_difference': fairness_metrics['equalized_odds']['equalized_odds_difference'],
        'tpr_disparity': fairness_metrics['equalized_odds']['tpr_disparity'],
        'fpr_disparity': fairness_metrics['equalized_odds']['fpr_disparity'],
        
        # Performance Parity
        'performance_gap': fairness_metrics['worst_group']['performance_gap'],
        'worst_group_f1': fairness_metrics['worst_group']['worst_group_f1'],
        'best_group_f1': fairness_metrics['worst_group']['best_group_f1'],
        'worst_group_id': fairness_metrics['worst_group']['worst_group'],
        
        # Calibration
        'calibration_disparity': fairness_metrics['calibration']['calibration_disparity'],
        'mean_ece': fairness_metrics['calibration']['mean_ece'],
        
        # Overall assessment
        'num_groups': len(fairness_metrics['worst_group']['group_f1']),
        'threshold': fairness_metrics['threshold_used']
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate data
    n_samples = 1000
    predictions = np.random.rand(n_samples)
    labels = (np.random.rand(n_samples) > 0.5).astype(int)
    groups = np.random.randint(0, 6, n_samples)  # 6 Fitzpatrick types
    
    # Compute all metrics
    fairness_metrics = compute_all_fairness_metrics(predictions, labels, groups)
    
    # Print summary
    summary = compute_fairness_summary(fairness_metrics)
    print("Fairness Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
