"""
3-phase training loop with progressive unfreezing.
"""
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from loguru import logger
from typing import Optional

from src.config import config
from src.models.full_model import build_full_model
from src.training.losses import FocalLoss, SmoothL1Loss
from src.training.callbacks import get_callbacks


class Trainer:
    """
    Three-phase trainer for lung cancer detection model.
    Phase 1: Frozen backbone, train heads only
    Phase 2: Unfreeze last N layers, fine-tune
    Phase 3: Full fine-tuning with low learning rate
    """
    
    def __init__(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        use_metadata: bool = None
    ):
        """
        Initialize trainer.
        
        Args:
            train_dataset: Training tf.data.Dataset
            val_dataset: Validation tf.data.Dataset
            use_metadata: Whether to use metadata fusion
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.use_metadata = use_metadata if use_metadata is not None else config.model.use_metadata_fusion
        
        # Enable mixed precision if configured
        if config.training.mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision training enabled")
        
        # Build model
        self.model = build_full_model(use_metadata=self.use_metadata)
        logger.info(f"Model built with metadata fusion: {self.use_metadata}")
        
    def _compile(self, learning_rate: float) -> None:
        """
        Compile model with optimizer and losses.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        # Optimizer
        optimizer = AdamW(
            learning_rate=learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Losses
        losses = {
            'malignancy': FocalLoss(),
            'bbox': SmoothL1Loss(),
            'objectness': tf.keras.losses.BinaryCrossentropy()
        }
        
        # Loss weights
        loss_weights = {
            'malignancy': 1.0,
            'bbox': config.training.detection_loss_weight,
            'objectness': config.training.detection_loss_weight
        }
        
        # Metrics
        metrics = {
            'malignancy': [
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='sensitivity'),
                tf.keras.metrics.Precision(name='precision')
            ]
        }
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with learning rate: {learning_rate}")
    
    def _freeze_backbone(self) -> None:
        """Freeze all backbone layers"""
        self.model.backbone.trainable = False
        logger.info("Backbone frozen")
    
    def _unfreeze_backbone_partial(self) -> None:
        """Unfreeze last N layers of backbone"""
        self.model.backbone.trainable = True
        
        # Freeze all layers first
        for layer in self.model.backbone.layers:
            layer.trainable = False
        
        # Unfreeze last N layers
        unfreeze_from = config.model.unfreeze_from_layer
        for layer in self.model.backbone.layers[unfreeze_from:]:
            layer.trainable = True
        
        trainable_count = sum([1 for layer in self.model.backbone.layers if layer.trainable])
        logger.info(f"Backbone partially unfrozen: {trainable_count} layers trainable")
    
    def _unfreeze_all(self) -> None:
        """Unfreeze all layers"""
        self.model.backbone.trainable = True
        for layer in self.model.backbone.layers:
            layer.trainable = True
        logger.info("All layers unfrozen")
    
    def run(self) -> tf.keras.Model:
        """
        Execute 3-phase training.
        
        Returns:
            Trained Keras model
        """
        logger.info("=" * 60)
        logger.info("STARTING 3-PHASE TRAINING")
        logger.info("=" * 60)
        
        # PHASE 1: Frozen backbone
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Training heads with frozen backbone")
        logger.info("=" * 60)
        
        self._freeze_backbone()
        self._compile(config.training.lr_phase1)
        
        history_phase1 = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=config.training.epochs_phase1,
            callbacks=get_callbacks(phase=1),
            verbose=1
        )
        
        # PHASE 2: Partial unfreezing
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Fine-tuning with partial backbone unfreezing")
        logger.info("=" * 60)
        
        self._unfreeze_backbone_partial()
        self._compile(config.training.lr_phase2)
        
        history_phase2 = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=config.training.epochs_phase2,
            callbacks=get_callbacks(phase=2),
            verbose=1
        )
        
        # PHASE 3: Full fine-tuning
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Full fine-tuning with low learning rate")
        logger.info("=" * 60)
        
        self._unfreeze_all()
        self._compile(config.training.lr_phase3)
        
        history_phase3 = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=config.training.epochs_phase3,
            callbacks=get_callbacks(phase=3),
            verbose=1
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        return self.model
