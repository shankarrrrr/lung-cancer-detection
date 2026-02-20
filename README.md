# Lung Cancer Detection System - rises.io

Production-grade AI system for detecting lung cancer-related findings in Chest X-Rays (CXR) with clinical validation and explainability.

## 🎯 Overview

This system provides:
- **Malignancy Detection**: Binary classification with probability scores
- **Lesion Localization**: Bounding box detection for suspicious regions
- **Explainability**: Grad-CAM heatmaps for visual interpretation
- **Uncertainty Quantification**: Monte Carlo Dropout for confidence estimation
- **Clinical Metadata Fusion**: Integration of patient demographics and symptoms
- **Production API**: FastAPI deployment with Docker support

## 🏗️ Architecture

```
Input CXR → Preprocessing → DenseNet121/EfficientNetB3 Backbone
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Classification Head              Detection Head
            (Malignancy Prob)               (BBox + Objectness)
                    ↓                               ↓
            Metadata Fusion (Optional)
                    ↓
            Clinical Output + Grad-CAM
```

## 📋 Requirements

- Python 3.11+
- TensorFlow 2.15.0
- CUDA 12 (for GPU support)
- 16GB+ GPU VRAM recommended
- Ubuntu 22.04 LTS

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3.11 -m venv lung_ai_env
source lung_ai_env/bin/activate  # On Windows: lung_ai_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Download datasets (see DATASETS.md for details)
# - NIH ChestX-ray14
# - CheXpert
# - VinDr-CXR
# - MIMIC-CXR

# Preprocess DICOM files
python scripts/preprocess_all.py --datasets nih chexpert vindr
```

### Training

```python
from src.training.trainer import Trainer
from src.data.dataset import LungCancerDataset
import pandas as pd

# Load data
train_df = pd.read_csv('data/splits/train.csv')
val_df = pd.read_csv('data/splits/val.csv')

# Build datasets
train_dataset = LungCancerDataset(train_df, mode='train').build()
val_dataset = LungCancerDataset(val_df, mode='val').build()

# Train model (3-phase)
trainer = Trainer(train_dataset, val_dataset)
model = trainer.run()

# Save model
model.save('artifacts/lung_ai_model')
```

### Inference

```python
from src.inference.predictor import LungCancerPredictor

# Initialize predictor
predictor = LungCancerPredictor(model_path='artifacts/lung_ai_model')

# Predict
result = predictor.predict(
    img_path='path/to/chest_xray.png',
    patient_id='P12345',
    metadata={'age': 65, 'smoking_pack_years': 30, 'symptom_score': 7},
    save_heatmap_path='output/heatmap.png'
)

print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['malignancy_probability']:.3f}")
print(f"Recommendation: {result['recommendation']}")
```

### API Deployment

```bash
# Run locally
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose -f docker/docker-compose.yml up --build

# Test API
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@chest_xray.png" \
  -F "patient_id=P12345" \
  -F "age=65" \
  -F "smoking_pack_years=30"
```

## 📊 Evaluation

### Run External Validation

```bash
python scripts/run_validation.py \
  --model-path artifacts/lung_ai_model \
  --datasets chexpert vindr mimic \
  --output-dir results/validation
```

### Key Metrics

- **AUC-ROC**: Area under ROC curve
- **Sensitivity**: True positive rate (target ≥ 0.90)
- **Specificity**: True negative rate
- **Brier Score**: Calibration metric
- **mAP@0.5**: Detection accuracy

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_model_shapes.py -v
pytest tests/test_data_pipeline.py -v
pytest tests/test_api.py -v
```

## 📁 Project Structure

```
├── data/                   # Data storage
│   ├── raw/               # Original DICOMs
│   ├── processed/         # Preprocessed PNGs
│   ├── splits/            # Train/val/test splits
│   └── annotations/       # Bounding box annotations
├── src/                   # Source code
│   ├── config.py         # Configuration
│   ├── data/             # Data pipeline
│   ├── models/           # Model architecture
│   ├── training/         # Training logic
│   ├── evaluation/       # Metrics and validation
│   ├── explainability/   # Grad-CAM
│   ├── inference/        # Prediction pipeline
│   └── api/              # FastAPI application
├── scripts/              # Utility scripts
├── tests/                # Unit tests
├── docker/               # Docker configuration
├── notebooks/            # Jupyter notebooks
└── artifacts/            # Trained models
```

## ⚙️ Configuration

All hyperparameters are centralized in `src/config.py`:

```python
from src.config import config

# Data configuration
config.data.image_size = (512, 512)
config.data.train_ratio = 0.70

# Model configuration
config.model.backbone = 'densenet121'
config.model.dropout_rate = 0.4

# Training configuration
config.training.batch_size = 16
config.training.lr_phase1 = 1e-3
```

## 🔬 Clinical Validation

### Threshold Optimization

The system optimizes decision thresholds for **target sensitivity ≥ 0.90** on validation data, prioritizing detection of positive cases (screening tool requirement).

### Calibration

Model calibration is evaluated using:
- Reliability diagrams
- Brier score
- Expected Calibration Error (ECE)

### External Validation

Cross-dataset validation ensures generalization:
- Train on CheXpert
- Validate on VinDr-CXR, MIMIC-CXR
- Statistical comparison using DeLong test

## ⚠️ Important Notes

### Data Leakage Prevention

- **ALWAYS** split by patient ID, never by image
- The `PatientSplitGenerator` includes mandatory leakage assertion
- Never disable the `_assert_no_leakage()` check

### Augmentation Rules

- Training: Geometric + intensity transforms
- Validation/Test: **ONLY normalization** - no augmentation

### Threshold Selection

- **NEVER** use 0.5 as decision threshold
- Always optimize on validation set for target sensitivity
- Apply optimized threshold to test set

### Grad-CAM Validation

- Heatmaps must focus on lung parenchyma
- Reject models with corner/edge/text activation
- Indicates spurious correlation if validation fails

## 📜 Disclaimer

**AI-assisted screening tool. Not a replacement for clinical judgment. Results must be reviewed by a qualified radiologist.**

## 📄 License

[Your License Here]

## 👥 Contributors

rises.io Medical AI Team

## 📧 Contact

[Your Contact Information]

---

**Version**: 1.0.0  
**Last Updated**: 2026-02-20
