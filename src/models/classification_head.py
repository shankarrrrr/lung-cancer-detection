"""
Classification head for malignancy prediction.
"""
import tensorflow as tf
from tensorflow.keras import layers

from src.config import config


def build_classification_head(features: tf.Tensor) -> tf.Tensor:
    """
    Build classification head for malignancy probability.
    
    Args:
        features: Feature tensor from backbone (B, H, W, C)
        
    Returns:
        Classification output tensor (B, 1) with sigmoid activation
    """
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(features)
    
    # Dense layers
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Dropout with specific name for MC Dropout at inference
    x = layers.Dropout(config.model.dropout_rate, name='cls_dropout')(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='cls_output')(x)
    
    return output


def build_classification_head_intermediate(features: tf.Tensor) -> tf.Tensor:
    """
    Build classification head without final sigmoid (for metadata fusion).
    
    Args:
        features: Feature tensor from backbone (B, H, W, C)
        
    Returns:
        Intermediate features before final classification
    """
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(features)
    
    # Dense layers
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Dropout
    x = layers.Dropout(config.model.dropout_rate, name='cls_dropout')(x)
    
    return x
