"""
Clinical evaluation metrics for lung cancer detection.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, brier_score_loss
)
from scipy import stats
from typing import Dict, Tuple
from loguru import logger

from src.config import config


class ClinicalMetrics:
    """
    Compute clinical evaluation metrics with focus on sensitivity.
    """
    
    def __init__(self, target_sensitivity: float = None):
        """
        Initialize metrics calculator.
        
        Args:
            target_sensitivity: Target sensitivity for threshold optimization
        """
        self.target_sensitivity = target_sensitivity if target_sensitivity is not None else config.inference.target_sensitivity
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """
        Compute all clinical metrics.
        
        Args:
            y_true: Ground truth labels (N,)
            y_prob: Predicted probabilities (N,)
            
        Returns:
            Dictionary with all metrics
        """
        # AUC-ROC
        auc = roc_auc_score(y_true, y_prob)
        
        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find optimal threshold at target sensitivity
        target_idx = np.argmin(np.abs(tpr - self.target_sensitivity))
        opt_threshold = thresholds[target_idx]
        
        logger.info(f"Optimal threshold for {self.target_sensitivity:.1%} sensitivity: {opt_threshold:.3f}")
        
        # Predictions at optimal threshold
        y_pred = (y_prob >= opt_threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Metrics
        sensitivity = recall_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        brier = brier_score_loss(y_true, y_prob)
        
        metrics = {
            'auc': float(auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1': float(f1),
            'brier_score': float(brier),
            'optimal_threshold': float(opt_threshold),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
        
        return metrics
    
    def delong_test(
        self,
        y_true: np.ndarray,
        probs_a: np.ndarray,
        probs_b: np.ndarray,
        n_bootstrap: int = 1000
    ) -> Dict:
        """
        DeLong test for comparing two ROC curves.
        Uses bootstrap approximation.
        
        Args:
            y_true: Ground truth labels
            probs_a: Predictions from model A
            probs_b: Predictions from model B
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary with AUCs and p-value
        """
        auc_a = roc_auc_score(y_true, probs_a)
        auc_b = roc_auc_score(y_true, probs_b)
        
        # Bootstrap
        n_samples = len(y_true)
        auc_diffs = []
        
        np.random.seed(config.training.seed)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            try:
                auc_a_boot = roc_auc_score(y_true[indices], probs_a[indices])
                auc_b_boot = roc_auc_score(y_true[indices], probs_b[indices])
                auc_diffs.append(auc_a_boot - auc_b_boot)
            except:
                continue
        
        auc_diffs = np.array(auc_diffs)
        
        # Two-tailed test
        p_value = 2 * min(
            np.mean(auc_diffs >= 0),
            np.mean(auc_diffs <= 0)
        )
        
        result = {
            'auc_a': float(auc_a),
            'auc_b': float(auc_b),
            'delta_auc': float(auc_a - auc_b),
            'p_value': float(p_value)
        }
        
        logger.info(f"DeLong Test: AUC_A={auc_a:.3f}, AUC_B={auc_b:.3f}, p={p_value:.4f}")
        
        return result
    
    @staticmethod
    def compute_iou(box_pred: np.ndarray, box_gt: np.ndarray) -> float:
        """
        Compute Intersection over Union for bounding boxes.
        
        Args:
            box_pred: Predicted box [x1, y1, x2, y2] normalized 0-1
            box_gt: Ground truth box [x1, y1, x2, y2] normalized 0-1
            
        Returns:
            IoU score
        """
        # Intersection coordinates
        x1_inter = max(box_pred[0], box_gt[0])
        y1_inter = max(box_pred[1], box_gt[1])
        x2_inter = min(box_pred[2], box_gt[2])
        y2_inter = min(box_pred[3], box_gt[3])
        
        # Intersection area
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height
        
        # Union area
        pred_area = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
        gt_area = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
        union_area = pred_area + gt_area - inter_area
        
        # IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        return float(iou)
    
    @staticmethod
    def compute_map_at_iou(
        predictions: list,
        groundtruths: list,
        iou_threshold: float = 0.5
    ) -> float:
        """
        Compute mean Average Precision at IoU threshold.
        
        Args:
            predictions: List of (bbox, confidence) tuples
            groundtruths: List of ground truth bboxes
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            mAP score
        """
        if len(predictions) == 0 or len(groundtruths) == 0:
            return 0.0
        
        # Sort predictions by confidence descending
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        matched_gt = set()
        
        for i, (pred_box, conf) in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(groundtruths):
                if gt_idx in matched_gt:
                    continue
                
                iou = ClinicalMetrics.compute_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Precision and recall
        recalls = tp_cumsum / len(groundtruths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using trapezoidal rule
        ap = np.trapz(precisions, recalls)
        
        return float(ap)
