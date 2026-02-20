"""
External validation runner script.
"""
import argparse
from pathlib import Path
import tensorflow as tf
from loguru import logger
import sys
import pandas as pd

from src.evaluation.validator import ExternalValidator
from src.config import config

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/validation.log", rotation="500 MB")


def load_model(model_path: Path) -> tf.keras.Model:
    """Load trained model"""
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded successfully")
    return model


def load_dataset(dataset_name: str, split: str = 'test') -> tf.data.Dataset:
    """
    Load dataset for validation.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split to load
        
    Returns:
        TensorFlow dataset
    """
    # This is a placeholder - implement actual dataset loading
    logger.info(f"Loading {dataset_name} {split} dataset")
    
    # In production, load from processed data
    csv_path = config.data.splits_dir / dataset_name / f"{split}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {dataset_name}")
    
    # Build dataset (simplified - use actual dataset.py in production)
    # This is just a placeholder
    return None


def main():
    """Main validation pipeline"""
    parser = argparse.ArgumentParser(description="Run external validation")
    parser.add_argument(
        '--model-path',
        type=Path,
        default=config.inference.model_path,
        help='Path to trained model'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['chexpert', 'vindr', 'mimic'],
        help='Datasets for external validation'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/external_validation'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Compare with baseline model'
    )
    parser.add_argument(
        '--baseline-model-path',
        type=Path,
        help='Path to baseline model for comparison'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EXTERNAL VALIDATION")
    logger.info("=" * 60)
    
    # Load model
    try:
        model = load_model(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    
    # Initialize validator
    validator = ExternalValidator(model)
    
    # Load datasets
    datasets = {}
    for dataset_name in args.datasets:
        try:
            dataset = load_dataset(dataset_name, split='test')
            if dataset is not None:
                datasets[dataset_name] = dataset
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {str(e)}")
    
    if not datasets:
        logger.error("No datasets loaded. Exiting.")
        return
    
    # Run validation
    logger.info(f"\nValidating on {len(datasets)} datasets...")
    
    try:
        results_df = validator.validate_multiple(
            datasets,
            save_dir=args.output_dir
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"\n{results_df.to_string()}")
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return
    
    # Model comparison if requested
    if args.compare_models and args.baseline_model_path:
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        try:
            baseline_model = load_model(args.baseline_model_path)
            
            for dataset_name, dataset in datasets.items():
                comparison = validator.compare_models(
                    baseline_model,
                    dataset,
                    dataset_name
                )
                
                logger.info(f"\n{dataset_name}:")
                logger.info(f"  Model A AUC: {comparison['auc_a']:.3f}")
                logger.info(f"  Model B AUC: {comparison['auc_b']:.3f}")
                logger.info(f"  Delta AUC: {comparison['delta_auc']:.3f}")
                logger.info(f"  p-value: {comparison['p_value']:.4f}")
                
                if comparison['p_value'] < 0.05:
                    logger.info("  ✓ Statistically significant difference")
                else:
                    logger.info("  ✗ No significant difference")
        
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
