"""
Test model architecture and output shapes.
"""
import pytest
import numpy as np
import tensorflow as tf

from src.models.full_model import build_full_model
from src.config import config


def test_model_without_metadata():
    """Test model output shapes without metadata"""
    # Build model
    model = build_full_model(use_metadata=False)
    
    # Create dummy input
    batch_size = 2
    dummy_input = {
        'image': np.random.randn(batch_size, 512, 512, 3).astype(np.float32)
    }
    
    # Get predictions
    predictions = model(dummy_input)
    
    # Check output shapes
    assert predictions['malignancy'].shape == (batch_size, 1), \
        f"Expected malignancy shape (2, 1), got {predictions['malignancy'].shape}"
    
    assert predictions['bbox'].shape == (batch_size, 4), \
        f"Expected bbox shape (2, 4), got {predictions['bbox'].shape}"
    
    assert predictions['objectness'].shape == (batch_size, 1), \
        f"Expected objectness shape (2, 1), got {predictions['objectness'].shape}"
    
    # Check output ranges
    assert tf.reduce_all(predictions['malignancy'] >= 0) and tf.reduce_all(predictions['malignancy'] <= 1), \
        "Malignancy predictions must be in [0, 1]"
    
    assert tf.reduce_all(predictions['bbox'] >= 0) and tf.reduce_all(predictions['bbox'] <= 1), \
        "BBox coordinates must be in [0, 1]"
    
    assert tf.reduce_all(predictions['objectness'] >= 0) and tf.reduce_all(predictions['objectness'] <= 1), \
        "Objectness predictions must be in [0, 1]"
    
    print("✓ Model without metadata test passed")


def test_model_with_metadata():
    """Test model output shapes with metadata"""
    # Build model
    model = build_full_model(use_metadata=True)
    
    # Create dummy input
    batch_size = 2
    n_meta_features = len(config.model.metadata_features)
    
    dummy_input = {
        'image': np.random.randn(batch_size, 512, 512, 3).astype(np.float32),
        'metadata': np.random.randn(batch_size, n_meta_features).astype(np.float32)
    }
    
    # Get predictions
    predictions = model(dummy_input)
    
    # Check output shapes
    assert predictions['malignancy'].shape == (batch_size, 1), \
        f"Expected malignancy shape (2, 1), got {predictions['malignancy'].shape}"
    
    assert predictions['bbox'].shape == (batch_size, 4), \
        f"Expected bbox shape (2, 4), got {predictions['bbox'].shape}"
    
    assert predictions['objectness'].shape == (batch_size, 1), \
        f"Expected objectness shape (2, 1), got {predictions['objectness'].shape}"
    
    # Check output ranges
    assert tf.reduce_all(predictions['malignancy'] >= 0) and tf.reduce_all(predictions['malignancy'] <= 1), \
        "Malignancy predictions must be in [0, 1]"
    
    print("✓ Model with metadata test passed")


def test_model_summary():
    """Test that model can be built and summarized"""
    model = build_full_model(use_metadata=False)
    
    # Check that model has expected attributes
    assert hasattr(model, 'backbone'), "Model should have backbone attribute"
    assert hasattr(model, 'last_conv_layer_name'), "Model should have last_conv_layer_name attribute"
    
    # Print summary
    print("\nModel Summary:")
    model.summary()
    
    print("✓ Model summary test passed")


if __name__ == "__main__":
    test_model_without_metadata()
    test_model_with_metadata()
    test_model_summary()
    print("\n✓ All model shape tests passed!")
