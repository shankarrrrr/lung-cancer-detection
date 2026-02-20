"""
CNN backbone factory for feature extraction.
"""
import tensorflow as tf
from typing import Tuple

from src.config import config


def build_backbone(
    name: str = None,
    input_shape: Tuple[int, int, int] = None
) -> tf.keras.Model:
    """
    Build CNN backbone for feature extraction.
    
    Args:
        name: Backbone architecture name ('densenet121' or 'efficientnetb3')
        input_shape: Input image shape (H, W, C)
        
    Returns:
        Keras Model with last_conv_layer_name attribute for Grad-CAM
    """
    if name is None:
        name = config.model.backbone
    
    if input_shape is None:
        input_shape = (*config.data.image_size, config.data.image_channels)
    
    if name == 'densenet121':
        base = tf.keras.applications.DenseNet121(
            weights=config.model.pretrained_weights,
            include_top=False,
            input_shape=input_shape
        )
        base.last_conv_layer_name = 'conv5_block16_concat'
        
    elif name == 'efficientnetb3':
        base = tf.keras.applications.EfficientNetB3(
            weights=config.model.pretrained_weights,
            include_top=False,
            input_shape=input_shape
        )
        base.last_conv_layer_name = 'top_conv'
        
    else:
        raise ValueError(f"Unknown backbone: {name}. Choose 'densenet121' or 'efficientnetb3'")
    
    return base
