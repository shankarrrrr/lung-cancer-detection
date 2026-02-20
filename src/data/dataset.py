"""
TensorFlow data pipeline with batching and augmentation.
"""
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

from src.config import config
from src.data.augmentation import MedicalAugmentation


class LungCancerDataset:
    """TensorFlow dataset pipeline for lung cancer detection"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = 'train',
        use_metadata: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with columns: image_path, label, patient_id, bbox (optional), metadata (optional)
            mode: 'train', 'val', or 'test'
            use_metadata: Whether to include clinical metadata
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.use_metadata = use_metadata
        self.augmentation = MedicalAugmentation(mode=mode)
        self.batch_size = config.training.batch_size
        
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _parse_function(self, idx: int) -> Dict:
        """Parse single sample"""
        row = self.df.iloc[idx]
        
        # Load image
        image = self._load_image(row['image_path'])
        
        # Get label
        label = float(row['label'])
        
        # Get bounding box if available
        if 'bbox' in row and pd.notna(row['bbox']):
            bbox = eval(row['bbox']) if isinstance(row['bbox'], str) else row['bbox']
        else:
            bbox = []
        
        # Apply augmentation
        image, bbox, _ = self.augmentation(image, bboxes=[bbox] if bbox else [], labels=[1] if bbox else [])
        
        # Prepare bbox output (normalized coordinates + objectness)
        if bbox and len(bbox) > 0:
            bbox_output = list(bbox[0]) + [1.0]  # [x1, y1, x2, y2, objectness]
        else:
            bbox_output = [0.0, 0.0, 0.0, 0.0, 0.0]  # No object
        
        output = {
            'image': image.astype(np.float32),
            'malignancy': np.array([label], dtype=np.float32),
            'bbox': np.array(bbox_output[:4], dtype=np.float32),
            'objectness': np.array([bbox_output[4]], dtype=np.float32)
        }
        
        # Add metadata if requested
        if self.use_metadata:
            metadata_values = []
            for feat in config.model.metadata_features:
                if feat in row:
                    metadata_values.append(float(row[feat]))
                else:
                    metadata_values.append(0.0)
            output['metadata'] = np.array(metadata_values, dtype=np.float32)
        
        return output
    
    def build(self) -> tf.data.Dataset:
        """Build TensorFlow dataset"""
        
        def generator():
            indices = np.arange(len(self.df))
            if self.mode == 'train':
                np.random.shuffle(indices)
            
            for idx in indices:
                yield self._parse_function(idx)
        
        # Define output signature
        output_signature = {
            'image': tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
            'malignancy': tf.TensorSpec(shape=(1,), dtype=tf.float32),
            'bbox': tf.TensorSpec(shape=(4,), dtype=tf.float32),
            'objectness': tf.TensorSpec(shape=(1,), dtype=tf.float32)
        }
        
        if self.use_metadata:
            output_signature['metadata'] = tf.TensorSpec(
                shape=(len(config.model.metadata_features),),
                dtype=tf.float32
            )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        # Batch and prefetch
        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=100)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
