# Quick Start Guide - Lung Cancer Detection System

## Step-by-Step Setup and Execution

### 1. Environment Setup (5 minutes)

```bash
# Create and activate virtual environment
python3.11 -m venv lung_ai_env

# Windows
lung_ai_env\Scripts\activate

# Linux/Mac
source lung_ai_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation (2 minutes)

```bash
# Test configuration
python -c "from src.config import config; print(config)"

# Run unit tests
pytest tests/test_model_shapes.py -v
pytest tests/test_data_pipeline.py -v
```

### 3. Data Preparation (varies by dataset size)

```bash
# Download datasets to data/raw/
# - NIH ChestX-ray14: https://www.kaggle.com/datasets/nih-chest-xrays/data
# - CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
# - VinDr-CXR: https://physionet.org/content/vindr-cxr/

# Preprocess DICOM files
python scripts/preprocess_all.py --datasets nih chexpert vindr
```

### 4. Generate Data Splits (5 minutes)

```python
# Create splits.py
from src.data.split_generator import PatientSplitGenerator
import pandas as pd

# Load your dataset CSV with columns: patient_id, image_path, label
df = pd.read_csv('your_dataset.csv')

# Generate patient-level splits
splitter = PatientSplitGenerator()
train_df, val_df, test_df = splitter.generate(df, 'patient_id', 'label')

# Save splits
train_df.to_csv('data/splits/train.csv', index=False)
val_df.to_csv('data/splits/val.csv', index=False)
test_df.to_csv('data/splits/test.csv', index=False)
```

### 5. Training (hours to days depending on dataset)

```python
# train.py
from src.training.trainer import Trainer
from src.data.dataset import LungCancerDataset
import pandas as pd

# Load splits
train_df = pd.read_csv('data/splits/train.csv')
val_df = pd.read_csv('data/splits/val.csv')

# Build datasets
train_dataset = LungCancerDataset(train_df, mode='train', use_metadata=False).build()
val_dataset = LungCancerDataset(val_df, mode='val', use_metadata=False).build()

# Train (3-phase)
trainer = Trainer(train_dataset, val_dataset, use_metadata=False)
model = trainer.run()

# Save model
model.save('artifacts/lung_ai_model')
```

Run training:
```bash
python train.py
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/
```

### 6. Evaluation (30 minutes)

```bash
# External validation
python scripts/run_validation.py \
  --model-path artifacts/lung_ai_model \
  --datasets chexpert vindr \
  --output-dir results/validation
```

### 7. Inference (seconds per image)

```python
from src.inference.predictor import LungCancerPredictor

# Initialize
predictor = LungCancerPredictor(model_path='artifacts/lung_ai_model')

# Predict
result = predictor.predict(
    img_path='path/to/chest_xray.png',
    patient_id='P12345',
    save_heatmap_path='output/heatmap.png'
)

print(f"Risk: {result['risk_level']}")
print(f"Probability: {result['malignancy_probability']:.3f}")
print(f"Recommendation: {result['recommendation']}")
```

### 8. API Deployment (5 minutes)

```bash
# Local deployment
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Docker deployment
docker-compose -f docker/docker-compose.yml up --build

# Test API
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@chest_xray.png" \
  -F "patient_id=P12345"
```

Access API docs at: http://localhost:8000/docs

## Execution Order Checklist

- [ ] 1. Install dependencies
- [ ] 2. Run unit tests
- [ ] 3. Download datasets
- [ ] 4. Preprocess DICOMs
- [ ] 5. Generate patient-level splits
- [ ] 6. Verify no patient leakage
- [ ] 7. Train model (Phase 1-3)
- [ ] 8. Evaluate on validation set
- [ ] 9. External validation
- [ ] 10. Generate Grad-CAM visualizations
- [ ] 11. Validate heatmap focus
- [ ] 12. Export model
- [ ] 13. Deploy API
- [ ] 14. Test API endpoints

## Common Issues

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `src/config.py`
```python
config.training.batch_size = 8  # Default is 16
```

### Issue: Patient leakage assertion fails
**Solution**: This is CRITICAL - do not disable. Check your data splitting logic.

### Issue: Grad-CAM focuses on image corners
**Solution**: Model learned spurious correlations. Retrain with cleaned data.

### Issue: Low sensitivity (<0.90)
**Solution**: Adjust threshold on validation set, not test set.

## Performance Benchmarks

| Task | Expected Time | Hardware |
|------|--------------|----------|
| DICOM Conversion (1000 images) | 5-10 min | CPU |
| Training Phase 1 (10 epochs) | 2-4 hours | GPU 16GB |
| Training Phase 2 (20 epochs) | 4-8 hours | GPU 16GB |
| Training Phase 3 (30 epochs) | 6-12 hours | GPU 16GB |
| Inference (single image) | 0.5-1 sec | GPU |
| Inference (single image) | 2-5 sec | CPU |

## Next Steps

1. Review `notebooks/01_eda.ipynb` for data exploration
2. Customize `src/config.py` for your use case
3. Read clinical validation requirements in README.md
4. Set up monitoring and logging for production

## Support

For issues, refer to:
- README.md for detailed documentation
- Test files in `tests/` for usage examples
- API docs at `/docs` endpoint
