"""
Training callbacks for model checkpointing, learning rate scheduling, and monitoring.
"""
import tensorflow as tf
from tensorflow.keras import callbacks
from pathlib import Path
from typing import List

from src.config import config


def get_callbacks(phase: int, monitor: str = 'val_auc') -> List[callbacks.Callback]:
    """
    Get training callbacks for a specific phase.
    
    Args:
        phase: Training phase number (1, 2, or 3)
        monitor: Metric to monitor for callbacks
        
    Returns:
        List of Keras callbacks
    """
    # Ensure checkpoint directory exists
    checkpoint_dir = config.training.checkpoint_dir / f'phase{phase}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Model checkpoint - save best model
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / 'best_model.h5'),
        monitor=monitor,
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
        mode='max'
    )
    
    # Early stopping
    early_stop_cb = callbacks.EarlyStopping(
        monitor=monitor,
        patience=config.training.early_stop_patience,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    
    # TensorBoard logging
    log_dir = Path('logs') / f'phase{phase}'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_cb = callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # CSV logger
    csv_logger_cb = callbacks.CSVLogger(
        filename=str(checkpoint_dir / 'training_log.csv'),
        append=True
    )
    
    return [
        checkpoint_cb,
        reduce_lr_cb,
        early_stop_cb,
        tensorboard_cb,
        csv_logger_cb
    ]
