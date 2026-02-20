"""
Central configuration for Lung Cancer Detection System.
SINGLE SOURCE OF TRUTH for all hyperparameters.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List


@dataclass
class DataConfig:
    """Data pipeline configuration"""
    raw_dir: Path = Path('data/raw')
    processed_dir: Path = Path('data/processed')
    splits_dir: Path = Path('data/splits')
    annotations_dir: Path = Path('data/annotations')
    image_size: Tuple[int, int] = (512, 512)
    image_channels: int = 3
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    backbone: str = 'densenet121'  # Options: 'densenet121' or 'efficientnetb3'
    pretrained_weights: str = 'imagenet'
    freeze_backbone_epochs: int = 5
    unfreeze_from_layer: int = -50
    use_metadata_fusion: bool = True
    metadata_features: List[str] = field(default_factory=lambda: ['age', 'smoking_pack_years', 'symptom_score'])
    dropout_rate: float = 0.4
    mc_dropout_samples: int = 20
    num_bbox_anchors: int = 5


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 16
    epochs_phase1: int = 10
    epochs_phase2: int = 20
    epochs_phase3: int = 30
    lr_phase1: float = 1e-3
    lr_phase2: float = 1e-4
    lr_phase3: float = 5e-5
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    detection_loss_weight: float = 0.5
    early_stop_patience: int = 10
    checkpoint_dir: Path = Path('checkpoints')
    mixed_precision: bool = True
    seed: int = 42


@dataclass
class InferenceConfig:
    """Inference and deployment configuration"""
    model_path: Path = Path('artifacts/lung_ai_model')
    high_risk_threshold: float = 0.70
    moderate_risk_threshold: float = 0.40
    nms_iou_threshold: float = 0.45
    min_bbox_confidence: float = 0.30
    target_sensitivity: float = 0.90


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# Singleton instance
config = Config()
