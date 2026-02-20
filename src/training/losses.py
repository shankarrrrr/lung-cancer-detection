"""
Custom loss functions for lung cancer detection.
"""
import tensorflow as tf
from tensorflow.keras import losses
from typing import Dict

from src.config import config


class FocalLoss(losses.Loss):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.
    """
    
    def __init__(
        self,
        gamma: float = None,
        alpha: float = None,
        name: str = 'focal_loss'
    ):
        super().__init__(name=name)
        self.gamma = gamma if gamma is not None else config.training.focal_loss_gamma
        self.alpha = alpha if alpha is not None else config.training.focal_loss_alpha
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss.
        
        Args:
            y_true: Ground truth labels (B, 1)
            y_pred: Predicted probabilities (B, 1)
            
        Returns:
            Focal loss value
        """
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Binary cross entropy
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Compute p_t
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Focal weight
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Alpha weight
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        return tf.reduce_mean(focal_loss)


class SmoothL1Loss(losses.Loss):
    """
    Smooth L1 Loss for bounding box regression.
    Only computed where objects exist (objectness mask).
    """
    
    def __init__(self, delta: float = 1.0, name: str = 'smooth_l1_loss'):
        super().__init__(name=name)
        self.huber = losses.Huber(delta=delta)
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute smooth L1 loss with objectness masking.
        
        Args:
            y_true: Ground truth with objectness mask (B, 5) [obj, x1, y1, x2, y2]
            y_pred: Predicted bbox coordinates (B, 4) [x1, y1, x2, y2]
            
        Returns:
            Masked smooth L1 loss
        """
        # Extract objectness mask (first column)
        obj_mask = y_true[:, 0:1]
        
        # Extract bbox coordinates
        bbox_true = y_true[:, 1:]
        
        # Compute Huber loss
        huber_loss = self.huber(bbox_true, y_pred)
        
        # Apply objectness mask - only compute loss where objects exist
        masked_loss = obj_mask * tf.expand_dims(huber_loss, axis=-1)
        
        # Average over samples with objects
        n_objects = tf.maximum(tf.reduce_sum(obj_mask), 1.0)
        
        return tf.reduce_sum(masked_loss) / n_objects


class CombinedLoss:
    """
    Combined loss for multi-task learning.
    Combines classification, detection, and objectness losses.
    """
    
    def __init__(self):
        self.focal_loss = FocalLoss()
        self.smooth_l1_loss = SmoothL1Loss()
        self.bce_loss = losses.BinaryCrossentropy()
        self.detection_weight = config.training.detection_loss_weight
    
    def compute(
        self,
        y_true_dict: Dict[str, tf.Tensor],
        y_pred_dict: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """
        Compute combined loss.
        
        Args:
            y_true_dict: Dictionary of ground truth tensors
            y_pred_dict: Dictionary of predicted tensors
            
        Returns:
            Dictionary with 'total', 'cls', 'det' losses
        """
        # Classification loss (malignancy)
        l_cls = self.focal_loss(
            y_true_dict['malignancy'],
            y_pred_dict['malignancy']
        )
        
        # Detection loss (bbox + objectness)
        # Combine objectness with bbox for masking
        bbox_with_obj = tf.concat([
            y_true_dict['objectness'],
            y_true_dict['bbox']
        ], axis=-1)
        
        l_bbox = self.smooth_l1_loss(bbox_with_obj, y_pred_dict['bbox'])
        l_obj = self.bce_loss(y_true_dict['objectness'], y_pred_dict['objectness'])
        
        l_det = l_bbox + l_obj
        
        # Total loss
        l_total = l_cls + self.detection_weight * l_det
        
        return {
            'total': l_total,
            'cls': l_cls,
            'det': l_det
        }
