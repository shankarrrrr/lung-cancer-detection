"""
Patient-level data splitting with leakage prevention.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from loguru import logger

from src.config import config


class PatientSplitGenerator:
    """Generates train/val/test splits at patient level to prevent data leakage"""
    
    def __init__(self):
        self.train_ratio = config.data.train_ratio
        self.val_ratio = config.data.val_ratio
        self.test_ratio = config.data.test_ratio
        self.seed = config.data.seed
        
        # Validate ratios
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
    
    def _assert_no_leakage(
        self,
        train_patients: set,
        val_patients: set,
        test_patients: set
    ) -> None:
        """
        Assert that there is zero patient overlap between splits.
        This assertion must NEVER be commented out or disabled.
        
        Args:
            train_patients: Set of training patient IDs
            val_patients: Set of validation patient IDs
            test_patients: Set of test patient IDs
            
        Raises:
            AssertionError: If any patient overlap is detected
        """
        train_val_overlap = train_patients & val_patients
        train_test_overlap = train_patients & test_patients
        val_test_overlap = val_patients & test_patients
        
        assert len(train_val_overlap) == 0, \
            f"LEAKAGE DETECTED: {len(train_val_overlap)} patients in both train and val"
        
        assert len(train_test_overlap) == 0, \
            f"LEAKAGE DETECTED: {len(train_test_overlap)} patients in both train and test"
        
        assert len(val_test_overlap) == 0, \
            f"LEAKAGE DETECTED: {len(val_test_overlap)} patients in both val and test"
        
        logger.info("✓ No patient leakage detected across splits")
    
    def generate(
        self,
        df: pd.DataFrame,
        patient_col: str,
        label_col: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate patient-level train/val/test splits.
        
        Args:
            df: DataFrame with patient data
            patient_col: Column name containing patient IDs
            label_col: Optional column for stratification
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Get unique patient IDs
        unique_patients = df[patient_col].unique()
        n_patients = len(unique_patients)
        
        logger.info(f"Splitting {n_patients} unique patients")
        
        # Shuffle patients
        np.random.seed(self.seed)
        shuffled_patients = np.random.permutation(unique_patients)
        
        # Calculate split indices
        train_end = int(n_patients * self.train_ratio)
        val_end = train_end + int(n_patients * self.val_ratio)
        
        # Split patient IDs
        train_patients = set(shuffled_patients[:train_end])
        val_patients = set(shuffled_patients[train_end:val_end])
        test_patients = set(shuffled_patients[val_end:])
        
        # CRITICAL: Assert no leakage
        self._assert_no_leakage(train_patients, val_patients, test_patients)
        
        # Create dataframe splits
        train_df = df[df[patient_col].isin(train_patients)].copy()
        val_df = df[df[patient_col].isin(val_patients)].copy()
        test_df = df[df[patient_col].isin(test_patients)].copy()
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        if label_col:
            logger.info(f"Train positive rate: {train_df[label_col].mean():.3f}")
            logger.info(f"Val positive rate: {val_df[label_col].mean():.3f}")
            logger.info(f"Test positive rate: {test_df[label_col].mean():.3f}")
        
        return train_df, val_df, test_df
