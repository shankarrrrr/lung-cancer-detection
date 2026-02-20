"""
Model calibration evaluation with reliability diagrams.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from pathlib import Path
from typing import Tuple

from src.config import config


class CalibrationEvaluator:
    """Evaluate and visualize model calibration"""
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration evaluator.
        
        Args:
            n_bins: Number of bins for reliability diagram
        """
        self.n_bins = n_bins
    
    def compute_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute calibration curve and Brier score.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            
        Returns:
            Tuple of (fraction_of_positives, mean_predicted_value, brier_score)
        """
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true,
            y_prob,
            n_bins=self.n_bins,
            strategy='uniform'
        )
        
        # Brier score
        brier = brier_score_loss(y_true, y_prob)
        
        return fraction_of_positives, mean_predicted_value, brier
    
    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Path = None,
        title: str = "Reliability Diagram"
    ) -> None:
        """
        Plot reliability diagram.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            save_path: Optional path to save figure
            title: Plot title
        """
        fraction_of_positives, mean_predicted_value, brier = self.compute_calibration(
            y_true, y_prob
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot calibration curve
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker='o',
            linewidth=2,
            label=f'Model (Brier={brier:.3f})'
        )
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Formatting
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add text box with Brier score
        textstr = f'Brier Score: {brier:.4f}\nBins: {self.n_bins}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(
            0.05, 0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props
        )
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reliability diagram saved to {save_path}")
        
        plt.close()
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_dir: Path = None
    ) -> dict:
        """
        Complete calibration evaluation.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with calibration metrics
        """
        fraction_of_positives, mean_predicted_value, brier = self.compute_calibration(
            y_true, y_prob
        )
        
        # Expected Calibration Error (ECE)
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        
        results = {
            'brier_score': float(brier),
            'ece': float(ece),
            'mce': float(mce)
        }
        
        # Plot if save directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.plot_reliability_diagram(
                y_true,
                y_prob,
                save_path=save_dir / 'reliability_diagram.png'
            )
        
        return results
