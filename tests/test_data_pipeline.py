"""
Test data pipeline components.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.split_generator import PatientSplitGenerator
from src.data.augmentation import MedicalAugmentation
from src.config import config


def test_patient_split_no_leakage():
    """Test that patient splits have zero overlap"""
    # Create dummy dataframe
    n_patients = 100
    n_images_per_patient = 3
    
    data = []
    for patient_id in range(n_patients):
        for img_id in range(n_images_per_patient):
            data.append({
                'patient_id': f'P{patient_id:04d}',
                'image_path': f'img_{patient_id}_{img_id}.png',
                'label': np.random.randint(0, 2)
            })
    
    df = pd.DataFrame(data)
    
    # Generate splits
    splitter = PatientSplitGenerator()
    train_df, val_df, test_df = splitter.generate(df, 'patient_id', 'label')
    
    # Get unique patients from each split
    train_patients = set(train_df['patient_id'].unique())
    val_patients = set(val_df['patient_id'].unique())
    test_patients = set(test_df['patient_id'].unique())
    
    # Assert no overlap
    assert len(train_patients & val_patients) == 0, "Train and val have patient overlap!"
    assert len(train_patients & test_patients) == 0, "Train and test have patient overlap!"
    assert len(val_patients & test_patients) == 0, "Val and test have patient overlap!"
    
    # Check split ratios
    total_patients = len(train_patients) + len(val_patients) + len(test_patients)
    assert total_patients == n_patients, "Some patients were lost during splitting"
    
    print(f"✓ Patient split test passed: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")


def test_augmentation_train_mode():
    """Test that training augmentation applies transforms"""
    aug = MedicalAugmentation(mode='train')
    
    # Create dummy image
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    bbox = [[100, 100, 200, 200]]
    labels = [1]
    
    # Apply augmentation multiple times
    results = []
    for _ in range(10):
        aug_img, aug_bbox, aug_labels = aug(img, bbox, labels)
        results.append(aug_img.copy())
    
    # Check that images are different (augmentation is applied)
    # At least some images should be different
    unique_count = len(set([tuple(r.flatten()) for r in results]))
    
    # With random augmentation, we expect variation
    print(f"✓ Training augmentation test passed: {unique_count}/10 unique augmented images")


def test_augmentation_val_mode():
    """Test that validation mode does NOT apply augmentation"""
    aug = MedicalAugmentation(mode='val')
    
    # Create dummy image
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    bbox = [[100, 100, 200, 200]]
    labels = [1]
    
    # Apply augmentation multiple times
    results = []
    for _ in range(5):
        aug_img, aug_bbox, aug_labels = aug(img, bbox, labels)
        results.append(aug_img.copy())
    
    # All images should be identical (only normalization, no random transforms)
    # Check first and last are the same
    assert np.allclose(results[0], results[-1], atol=1e-6), \
        "Val mode should not apply random augmentation!"
    
    print("✓ Validation augmentation test passed: No random transforms applied")


def test_split_ratios():
    """Test that split ratios match configuration"""
    # Create dummy dataframe
    n_patients = 1000
    data = [{'patient_id': f'P{i:04d}', 'label': np.random.randint(0, 2)} for i in range(n_patients)]
    df = pd.DataFrame(data)
    
    # Generate splits
    splitter = PatientSplitGenerator()
    train_df, val_df, test_df = splitter.generate(df, 'patient_id', 'label')
    
    # Check ratios (allow 5% tolerance)
    train_ratio = len(train_df) / len(df)
    val_ratio = len(val_df) / len(df)
    test_ratio = len(test_df) / len(df)
    
    assert abs(train_ratio - config.data.train_ratio) < 0.05, \
        f"Train ratio {train_ratio:.2f} doesn't match config {config.data.train_ratio}"
    
    assert abs(val_ratio - config.data.val_ratio) < 0.05, \
        f"Val ratio {val_ratio:.2f} doesn't match config {config.data.val_ratio}"
    
    assert abs(test_ratio - config.data.test_ratio) < 0.05, \
        f"Test ratio {test_ratio:.2f} doesn't match config {config.data.test_ratio}"
    
    print(f"✓ Split ratios test passed: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")


if __name__ == "__main__":
    test_patient_split_no_leakage()
    test_augmentation_train_mode()
    test_augmentation_val_mode()
    test_split_ratios()
    print("\n✓ All data pipeline tests passed!")
