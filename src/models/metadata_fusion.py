"""
Late fusion module for integrating clinical metadata with CNN features.
"""
import tensorflow as tf
from tensorflow.keras import layers

from src.config import config


def build_metadata_fusion(
    cnn_features: tf.Tensor,
    metadata_input: tf.Tensor
) -> tf.Tensor:
    """
    Build late fusion module combining CNN features with clinical metadata.
    
    Args:
        cnn_features: Intermediate CNN features (B, 512)
        metadata_input: Clinical metadata (B, n_features)
        
    Returns:
        Fused output tensor (B, 1) with sigmoid activation
    """
    # Process metadata independently
    meta_branch = layers.Dense(64, activation='relu')(metadata_input)
    meta_branch = layers.BatchNormalization()(meta_branch)
    meta_branch = layers.Dense(32, activation='relu')(meta_branch)
    
    # Process CNN features
    cnn_branch = layers.Dense(256, activation='relu')(cnn_features)
    
    # Concatenate both branches
    fused = layers.Concatenate()([cnn_branch, meta_branch])
    
    # Final fusion layers
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.Dropout(0.3)(fused)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='fused_output')(fused)
    
    return output
