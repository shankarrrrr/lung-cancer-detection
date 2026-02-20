"""
Full model assembly combining all components.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Optional

from src.config import config
from src.models.backbone import build_backbone
from src.models.classification_head import build_classification_head, build_classification_head_intermediate
from src.models.detection_head import build_detection_head
from src.models.metadata_fusion import build_metadata_fusion


def build_full_model(use_metadata: Optional[bool] = None) -> tf.keras.Model:
    """
    Build complete lung cancer detection model.
    
    Args:
        use_metadata: Whether to use metadata fusion. If None, uses config value.
        
    Returns:
        Keras Model with outputs: {'malignancy', 'bbox', 'objectness'}
    """
    if use_metadata is None:
        use_metadata = config.model.use_metadata_fusion
    
    # Input layers
    image_input = layers.Input(
        shape=(*config.data.image_size, config.data.image_channels),
        name='image_input'
    )
    
    inputs = {'image': image_input}
    
    if use_metadata:
        n_meta_features = len(config.model.metadata_features)
        meta_input = layers.Input(
            shape=(n_meta_features,),
            name='metadata_input'
        )
        inputs['metadata'] = meta_input
    
    # Backbone
    backbone = build_backbone()
    feature_map = backbone(image_input)
    
    # Detection head
    bbox_output, obj_output = build_detection_head(feature_map)
    
    # Classification head
    if use_metadata:
        # Get intermediate features for fusion
        cnn_features = build_classification_head_intermediate(feature_map)
        malignancy_output = build_metadata_fusion(cnn_features, meta_input)
    else:
        # Direct classification
        malignancy_output = build_classification_head(feature_map)
    
    # Build model
    outputs = {
        'malignancy': malignancy_output,
        'bbox': bbox_output,
        'objectness': obj_output
    }
    
    model = Model(inputs=inputs, outputs=outputs, name='lung_cancer_detector')
    
    # Attach backbone reference and last conv layer name for Grad-CAM
    model.backbone = backbone
    model.last_conv_layer_name = backbone.last_conv_layer_name
    
    return model
