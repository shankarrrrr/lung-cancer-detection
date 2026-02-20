"""
Detection head for bounding box regression and objectness prediction.
"""
import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple


def build_detection_head(features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Build detection head for bbox regression and objectness.
    
    Args:
        features: Feature tensor from backbone (B, H, W, C)
        
    Returns:
        Tuple of (bbox_output, objectness_output)
        - bbox_output: (B, 4) normalized coordinates [x1, y1, x2, y2]
        - objectness_output: (B, 1) confidence that lesion exists
    """
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(features)
    
    # Shared dense layer
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # BBox regression branch
    bbox_branch = layers.Dense(128, activation='relu')(x)
    bbox_output = layers.Dense(
        4,
        activation='sigmoid',
        name='bbox_output'
    )(bbox_branch)
    
    # Objectness branch
    obj_branch = layers.Dense(64, activation='relu')(x)
    obj_output = layers.Dense(
        1,
        activation='sigmoid',
        name='obj_output'
    )(obj_branch)
    
    return bbox_output, obj_output
