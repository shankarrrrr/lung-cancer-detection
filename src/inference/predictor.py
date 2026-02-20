"""
End-to-end inference with MC Dropout uncertainty estimation.
"""
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple
from loguru import logger

from src.config import config
from src.explainability.gradcam import GradCAM


class LungCancerPredictor:
    """
    Production inference pipeline for lung cancer detection.
    """
    
    def __init__(self, model_path: Path = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model
        """
        if model_path is None:
            model_path = config.inference.model_path
        
        self.model_path = Path(model_path)
        
        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(str(self.model_path))
        
        # Initialize Grad-CAM
        self.gradcam = GradCAM(self.model)
        
        logger.info("Predictor initialized successfully")
    
    def preprocess_image(self, img_path: Path) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image array (1, H, W, C)
        """
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, config.data.image_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_with_uncertainty(
        self,
        img_array: np.ndarray,
        n_samples: int = None
    ) -> Dict[str, float]:
        """
        Predict with Monte Carlo Dropout uncertainty estimation.
        
        Args:
            img_array: Preprocessed image (1, H, W, C)
            n_samples: Number of MC dropout samples
            
        Returns:
            Dictionary with mean, std, and samples
        """
        if n_samples is None:
            n_samples = config.model.mc_dropout_samples
        
        predictions = []
        
        # Run multiple forward passes with dropout enabled
        for _ in range(n_samples):
            pred = self.model(img_array, training=True)  # training=True keeps dropout active
            
            if isinstance(pred, dict):
                prob = pred['malignancy'].numpy()[0, 0]
            else:
                prob = pred[0].numpy()[0, 0]
            
            predictions.append(float(prob))
        
        predictions = np.array(predictions)
        
        return {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'samples': predictions.tolist()
        }
    
    def get_risk_level(self, prob: float) -> str:
        """
        Determine risk level based on probability.
        
        Args:
            prob: Malignancy probability
            
        Returns:
            Risk level string: 'HIGH', 'MODERATE', or 'LOW'
        """
        if prob >= config.inference.high_risk_threshold:
            return 'HIGH'
        elif prob >= config.inference.moderate_risk_threshold:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def get_recommendation(self, risk_level: str, prob: float) -> str:
        """
        Generate clinical recommendation based on risk level.
        
        Args:
            risk_level: Risk level string
            prob: Malignancy probability
            
        Returns:
            Recommendation string
        """
        if risk_level == 'HIGH':
            return f"HIGH RISK (p={prob:.3f}): Immediate CT scan and specialist consultation recommended."
        elif risk_level == 'MODERATE':
            return f"MODERATE RISK (p={prob:.3f}): Follow-up CT scan recommended within 3-6 months."
        else:
            return f"LOW RISK (p={prob:.3f}): Routine follow-up recommended. Continue regular screening."
    
    def predict(
        self,
        img_path: Path,
        metadata: Optional[Dict] = None,
        patient_id: str = "UNKNOWN",
        save_heatmap_path: Optional[Path] = None
    ) -> Dict:
        """
        Complete prediction pipeline.
        
        Args:
            img_path: Path to chest X-ray image
            metadata: Optional clinical metadata dict
            patient_id: Patient identifier
            save_heatmap_path: Optional path to save Grad-CAM heatmap
            
        Returns:
            Complete clinical report dictionary
        """
        logger.info(f"Processing image for patient {patient_id}")
        
        # Preprocess image
        img_array = self.preprocess_image(img_path)
        
        # Uncertainty estimation
        uncertainty = self.predict_with_uncertainty(img_array)
        prob = uncertainty['mean']
        std = uncertainty['std']
        
        # Get full prediction (including bbox and objectness)
        prediction = self.model.predict(img_array, verbose=0)
        
        if isinstance(prediction, dict):
            bbox = prediction['bbox'][0].tolist()
            objectness = float(prediction['objectness'][0, 0])
        else:
            bbox = prediction[1][0].tolist() if len(prediction) > 1 else [0, 0, 0, 0]
            objectness = float(prediction[2][0, 0]) if len(prediction) > 2 else 0.0
        
        # Risk assessment
        risk_level = self.get_risk_level(prob)
        recommendation = self.get_recommendation(risk_level, prob)
        
        # Generate Grad-CAM heatmap
        heatmap = self.gradcam.generate_heatmap(img_array)
        
        # Validate heatmap focus
        heatmap_validation = self.gradcam.validate_heatmap_focus(heatmap)
        
        # Save heatmap if requested
        heatmap_url = None
        if save_heatmap_path:
            # Load original image for overlay
            original_img = cv2.imread(str(img_path))
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.resize(original_img, config.data.image_size)
            
            self.gradcam.save_visualization(
                original_img,
                heatmap,
                save_heatmap_path,
                patient_id,
                prob
            )
            heatmap_url = str(save_heatmap_path)
            logger.info(f"Heatmap saved to {heatmap_url}")
        
        # Build clinical report
        report = {
            'patient_id': patient_id,
            'malignancy_probability': float(prob),
            'uncertainty_std': float(std),
            'risk_level': risk_level,
            'bbox_normalized': bbox,
            'bbox_confidence': float(objectness),
            'recommendation': recommendation,
            'heatmap_url': heatmap_url,
            'heatmap_valid': heatmap_validation['valid'],
            'metadata': metadata,
            'disclaimer': "AI-assisted screening tool. Not a replacement for clinical judgment. Results must be reviewed by a qualified radiologist."
        }
        
        logger.info(f"Prediction complete: {risk_level} risk (p={prob:.3f}, std={std:.3f})")
        
        return report
