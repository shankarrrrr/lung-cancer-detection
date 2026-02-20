"""
Medical image augmentation pipeline using Albumentations.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Optional, List, Tuple

from src.config import config


class MedicalAugmentation:
    """
    Augmentation pipeline for medical chest X-rays.
    Training mode applies augmentations, val/test mode only normalizes.
    """
    
    def __init__(self, mode: str = 'train'):
        """
        Initialize augmentation pipeline.
        
        Args:
            mode: 'train' or 'val' - determines augmentation strategy
        """
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'"
        self.mode = mode
        
        if mode == 'train':
            self.transform = self._build_train_pipeline()
        else:
            self.transform = self._build_val_pipeline()
    
    def _build_train_pipeline(self) -> A.Compose:
        """
        Build training augmentation pipeline.
        Includes geometric and intensity transforms suitable for medical imaging.
        """
        return A.Compose([
            A.Rotate(limit=10, p=0.5, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.GaussNoise(var_limit=(5, 20), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.3,
                border_mode=0
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def _build_val_pipeline(self) -> A.Compose:
        """
        Build validation/test pipeline.
        Only normalization - NO augmentation on val/test data.
        """
        return A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        labels: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, Optional[List[List[float]]], Optional[List[int]]]:
        """
        Apply augmentation pipeline.
        
        Args:
            image: Input image as numpy array (H, W, C)
            bboxes: Optional list of bounding boxes in pascal_voc format [x_min, y_min, x_max, y_max]
            labels: Optional list of class labels for each bbox
            
        Returns:
            Tuple of (augmented_image, augmented_bboxes, class_labels)
        """
        if bboxes is None:
            bboxes = []
        if labels is None:
            labels = []
        
        # Apply transforms
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )
        
        return (
            transformed['image'],
            transformed.get('bboxes', []),
            transformed.get('class_labels', [])
        )
