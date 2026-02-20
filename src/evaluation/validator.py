"""
Cross-dataset external validation runner.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from loguru import logger
import tensorflow as tf

from src.evaluation.metrics import ClinicalMetrics
from src.evaluation.calibration import CalibrationEvaluator
from src.config import config


class ExternalValidator:
    """
    Run external validation across multiple datasets.
    """
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize validator.
        
        Args:
            model: Trained Keras model
        """
        self.model = model
        self.metrics_calculator = ClinicalMetrics()
        self.calibration_evaluator = CalibrationEvaluator()
    
    def validate_dataset(
        self,
        dataset: tf.data.Dataset,
        dataset_name: str,
        save_dir: Path = None
    ) -> Dict:
        """
        Validate model on a single dataset.
        
        Args:
            dataset: TensorFlow dataset
            dataset_name: Name of the dataset
            save_dir: Directory to save results
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info(f"Validating on {dataset_name}...")
        
        # Collect predictions and ground truth
        y_true_list = []
        y_prob_list = []
        
        for batch in dataset:
            # Extract inputs and labels
            if isinstance(batch, tuple):
                inputs, labels = batch
            else:
                inputs = batch
                labels = batch
            
            # Get predictions
            predictions = self.model.predict(inputs, verbose=0)
            
            # Extract malignancy predictions
            if isinstance(predictions, dict):
                y_prob = predictions['malignancy']
            else:
                y_prob = predictions[0]
            
            # Extract ground truth
            if isinstance(labels, dict):
                y_true = labels['malignancy']
            else:
                y_true = labels
            
            y_true_list.append(y_true.numpy().flatten())
            y_prob_list.append(y_prob.flatten())
        
        # Concatenate all batches
        y_true = np.concatenate(y_true_list)
        y_prob = np.concatenate(y_prob_list)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_all(y_true, y_prob)
        
        # Compute calibration
        if save_dir:
            save_dir = Path(save_dir) / dataset_name
            save_dir.mkdir(parents=True, exist_ok=True)
            calibration_metrics = self.calibration_evaluator.evaluate(
                y_true,
                y_prob,
                save_dir=save_dir
            )
            metrics.update(calibration_metrics)
        
        # Log results
        logger.info(f"\n{dataset_name} Results:")
        logger.info(f"  AUC: {metrics['auc']:.3f}")
        logger.info(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        logger.info(f"  Specificity: {metrics['specificity']:.3f}")
        logger.info(f"  Brier Score: {metrics['brier_score']:.3f}")
        
        return metrics
    
    def validate_multiple(
        self,
        datasets: Dict[str, tf.data.Dataset],
        save_dir: Path = None
    ) -> pd.DataFrame:
        """
        Validate on multiple datasets.
        
        Args:
            datasets: Dictionary mapping dataset names to tf.data.Dataset
            save_dir: Directory to save results
            
        Returns:
            DataFrame with results for all datasets
        """
        results = {}
        
        for dataset_name, dataset in datasets.items():
            metrics = self.validate_dataset(
                dataset,
                dataset_name,
                save_dir=save_dir
            )
            results[dataset_name] = metrics
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results).T
        
        # Save to CSV
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            csv_path = save_dir / 'external_validation_results.csv'
            results_df.to_csv(csv_path)
            logger.info(f"Results saved to {csv_path}")
        
        return results_df
    
    def compare_models(
        self,
        model_b: tf.keras.Model,
        dataset: tf.data.Dataset,
        dataset_name: str = "Test"
    ) -> Dict:
        """
        Compare two models using DeLong test.
        
        Args:
            model_b: Second model to compare
            dataset: Test dataset
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing models on {dataset_name}...")
        
        # Collect ground truth and predictions from both models
        y_true_list = []
        y_prob_a_list = []
        y_prob_b_list = []
        
        for batch in dataset:
            if isinstance(batch, tuple):
                inputs, labels = batch
            else:
                inputs = batch
                labels = batch
            
            # Model A predictions
            pred_a = self.model.predict(inputs, verbose=0)
            if isinstance(pred_a, dict):
                y_prob_a = pred_a['malignancy']
            else:
                y_prob_a = pred_a[0]
            
            # Model B predictions
            pred_b = model_b.predict(inputs, verbose=0)
            if isinstance(pred_b, dict):
                y_prob_b = pred_b['malignancy']
            else:
                y_prob_b = pred_b[0]
            
            # Ground truth
            if isinstance(labels, dict):
                y_true = labels['malignancy']
            else:
                y_true = labels
            
            y_true_list.append(y_true.numpy().flatten())
            y_prob_a_list.append(y_prob_a.flatten())
            y_prob_b_list.append(y_prob_b.flatten())
        
        y_true = np.concatenate(y_true_list)
        y_prob_a = np.concatenate(y_prob_a_list)
        y_prob_b = np.concatenate(y_prob_b_list)
        
        # DeLong test
        comparison = self.metrics_calculator.delong_test(
            y_true,
            y_prob_a,
            y_prob_b
        )
        
        return comparison
