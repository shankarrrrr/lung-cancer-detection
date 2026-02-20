# Next Steps - Implementation Guide

## Immediate Actions (Today)

### 1. Verify Installation ✓
```bash
# Activate environment
# Windows:
lung_ai_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.config import config; print('✓ Config loaded successfully')"
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__} with GPU: {tf.config.list_physical_devices(\"GPU\")}')"
```

### 2. Run Unit Tests ✓
```bash
# Test model architecture
pytest tests/test_model_shapes.py -v

# Test data pipeline
pytest tests/test_data_pipeline.py -v

# Test API (without model)
pytest tests/test_api.py -v
```

Expected output: All tests should pass ✓

## Week 1: Data Preparation

### Day 1-2: Download Datasets
1. **NIH ChestX-ray14**
   - Source: https://www.kaggle.com/datasets/nih-chest-xrays/data
   - Download to: `data/raw/nih/`
   - Size: ~45GB

2. **CheXpert**
   - Source: https://stanfordmlgroup.github.io/competitions/chexpert/
   - Download to: `data/raw/chexpert/`
   - Size: ~11GB

3. **VinDr-CXR**
   - Source: https://physionet.org/content/vindr-cxr/
   - Download to: `data/raw/vindr/`
   - Includes bounding box annotations
   - Size: ~15GB

4. **MIMIC-CXR** (Optional, requires credentialing)
   - Source: https://physionet.org/content/mimic-cxr/
   - Download to: `data/raw/mimic/`

### Day 3-4: Preprocess Data
```bash
# Convert DICOMs to PNGs
python scripts/preprocess_all.py --datasets nih chexpert vindr

# Expected output: Processed PNGs in data/processed/
```

### Day 5: Create Dataset CSV
Create a CSV file with columns:
- `patient_id`: Unique patient identifier
- `image_path`: Path to processed PNG
- `label`: 0 (negative) or 1 (positive for malignancy)
- `bbox`: Optional bounding box [x1, y1, x2, y2]
- `age`: Optional patient age
- `smoking_pack_years`: Optional smoking history
- `symptom_score`: Optional symptom severity (0-10)

Example:
```python
import pandas as pd

# Create dataset CSV (example structure)
data = {
    'patient_id': ['P001', 'P001', 'P002'],
    'image_path': ['data/processed/nih/img1.png', 'data/processed/nih/img2.png', 'data/processed/nih/img3.png'],
    'label': [1, 1, 0],
    'bbox': ['[0.3,0.2,0.6,0.5]', '[0.35,0.25,0.65,0.55]', None],
    'age': [65, 65, 45],
    'smoking_pack_years': [30, 30, 0],
    'symptom_score': [7, 7, 2]
}

df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)
```

### Day 6-7: Generate Splits
```python
from src.data.split_generator import PatientSplitGenerator
import pandas as pd

# Load dataset
df = pd.read_csv('dataset.csv')

# Generate patient-level splits
splitter = PatientSplitGenerator()
train_df, val_df, test_df = splitter.generate(df, 'patient_id', 'label')

# Save splits
train_df.to_csv('data/splits/train.csv', index=False)
val_df.to_csv('data/splits/val.csv', index=False)
test_df.to_csv('data/splits/test.csv', index=False)

print("✓ Splits created with zero patient leakage")
```

## Week 2-3: Model Training

### Create Training Script
```python
# train.py
from src.training.trainer import Trainer
from src.data.dataset import LungCancerDataset
import pandas as pd
from loguru import logger

# Load splits
train_df = pd.read_csv('data/splits/train.csv')
val_df = pd.read_csv('data/splits/val.csv')

logger.info(f"Train: {len(train_df)} samples")
logger.info(f"Val: {len(val_df)} samples")

# Build datasets
train_dataset = LungCancerDataset(
    train_df,
    mode='train',
    use_metadata=False  # Set True if using clinical metadata
).build()

val_dataset = LungCancerDataset(
    val_df,
    mode='val',
    use_metadata=False
).build()

# Train model (3-phase)
trainer = Trainer(
    train_dataset,
    val_dataset,
    use_metadata=False
)

model = trainer.run()

# Save model
model.save('artifacts/lung_ai_model')
logger.info("✓ Model saved to artifacts/lung_ai_model")
```

### Run Training
```bash
# Start training
python train.py

# Monitor with TensorBoard (in separate terminal)
tensorboard --logdir logs/

# Access at: http://localhost:6006
```

Expected duration:
- Phase 1 (10 epochs): 2-4 hours
- Phase 2 (20 epochs): 4-8 hours
- Phase 3 (30 epochs): 6-12 hours
- **Total**: 12-24 hours on GPU

## Week 4: Evaluation

### Day 1-2: Validation Set Evaluation
```python
from src.evaluation.metrics import ClinicalMetrics
from src.evaluation.calibration import CalibrationEvaluator
import tensorflow as tf
import pandas as pd
import numpy as np

# Load model
model = tf.keras.models.load_model('artifacts/lung_ai_model')

# Load test data
test_df = pd.read_csv('data/splits/test.csv')

# Get predictions (simplified - use actual dataset in production)
# y_true = ...
# y_prob = ...

# Compute metrics
metrics_calc = ClinicalMetrics()
metrics = metrics_calc.compute_all(y_true, y_prob)

print(f"AUC: {metrics['auc']:.3f}")
print(f"Sensitivity: {metrics['sensitivity']:.3f}")
print(f"Specificity: {metrics['specificity']:.3f}")
print(f"Brier Score: {metrics['brier_score']:.3f}")

# Calibration
cal_eval = CalibrationEvaluator()
cal_metrics = cal_eval.evaluate(y_true, y_prob, save_dir='results/calibration')
```

### Day 3-4: External Validation
```bash
python scripts/run_validation.py \
  --model-path artifacts/lung_ai_model \
  --datasets chexpert vindr \
  --output-dir results/external_validation
```

### Day 5: Grad-CAM Validation
```python
from src.inference.predictor import LungCancerPredictor
from pathlib import Path

predictor = LungCancerPredictor()

# Generate heatmaps for 50 test images
test_images = list(Path('data/processed/test').glob('*.png'))[:50]

valid_count = 0
for img_path in test_images:
    result = predictor.predict(
        img_path=img_path,
        patient_id=img_path.stem,
        save_heatmap_path=f'results/heatmaps/{img_path.stem}_heatmap.png'
    )
    
    if result['heatmap_valid']:
        valid_count += 1

print(f"Heatmap validation: {valid_count}/50 ({valid_count/50*100:.1f}%) focused on lungs")
# Target: >80% lung-focused
```

## Week 5: Deployment

### Day 1-2: API Testing
```bash
# Start API locally
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@test_image.png" \
  -F "patient_id=TEST001" \
  -F "age=65" \
  -F "smoking_pack_years=30"

# Access API docs
# http://localhost:8000/docs
```

### Day 3-4: Docker Deployment
```bash
# Build Docker image
docker build -f docker/Dockerfile -t rises-lung-ai:latest .

# Run with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# Check logs
docker-compose -f docker/docker-compose.yml logs -f

# Test deployed API
curl http://localhost:8000/health
```

### Day 5: Production Checklist
- [ ] Model achieves sensitivity ≥ 0.90
- [ ] Brier score < 0.15
- [ ] >80% heatmaps lung-focused
- [ ] External validation on 3+ datasets
- [ ] API load testing completed
- [ ] Monitoring setup
- [ ] Error handling tested
- [ ] Documentation complete

## Ongoing: Monitoring & Maintenance

### Weekly
- Monitor prediction distribution
- Check for data drift
- Review failed predictions
- Update heatmap validation

### Monthly
- Retrain on new data
- External validation
- Performance metrics review
- Calibration check

### Quarterly
- Clinical validation study
- Bias assessment
- Model comparison
- Documentation update

## Troubleshooting

### Issue: CUDA out of memory
```python
# Reduce batch size in src/config.py
config.training.batch_size = 8  # or 4
```

### Issue: Training too slow
```python
# Enable mixed precision
config.training.mixed_precision = True
```

### Issue: Low sensitivity
```python
# Adjust threshold on validation set
config.inference.target_sensitivity = 0.95  # Increase target
```

### Issue: Poor calibration
- Collect more diverse training data
- Apply temperature scaling
- Increase training epochs

## Resources

- **TensorBoard**: Monitor training at http://localhost:6006
- **API Docs**: http://localhost:8000/docs
- **Logs**: Check `logs/` directory
- **Checkpoints**: Saved in `checkpoints/phase{1,2,3}/`
- **Results**: Validation results in `results/`

## Support Contacts

- Technical Issues: [Your Email]
- Clinical Questions: [Radiologist Contact]
- Data Questions: [Data Team Contact]

---

**Remember**: This is a screening tool. All predictions must be reviewed by qualified radiologists.
