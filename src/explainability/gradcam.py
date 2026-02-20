"""
Grad-CAM implementation for model explainability.
"""
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional

from src.config import config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visual explanations.
    """
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model with last_conv_layer_name attribute
        """
        self.model = model
        
        # Get last convolutional layer name
        if hasattr(model, 'last_conv_layer_name'):
            self.last_conv_layer_name = model.last_conv_layer_name
        else:
            raise ValueError("Model must have 'last_conv_layer_name' attribute")
        
        # Build grad model
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self) -> tf.keras.Model:
        """
        Build model that outputs conv layer activations and predictions.
        
        Returns:
            Keras Model for gradient computation
        """
        # Get the last conv layer
        last_conv_layer = self.model.get_layer(self.last_conv_layer_name)
        
        # Build model that outputs both conv activations and final predictions
        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[last_conv_layer.output, self.model.output]
        )
        
        return grad_model
    
    def generate_heatmap(
        self,
        img_array: np.ndarray,
        pred_index: int = 0
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            img_array: Input image array (1, H, W, C) or (H, W, C)
            pred_index: Index of prediction output (for multi-output models)
            
        Returns:
            2D heatmap array normalized to [0, 1]
        """
        # Ensure batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Record gradients
        with tf.GradientTape() as tape:
            # Get conv outputs and predictions
            conv_outputs, predictions = self.grad_model(img_array)
            
            # Extract malignancy prediction
            if isinstance(predictions, dict):
                pred = predictions['malignancy']
            else:
                pred = predictions[pred_index] if isinstance(predictions, list) else predictions
            
            # Get the predicted class score
            class_channel = pred[:, 0]
        
        # Compute gradients of class score w.r.t. conv outputs
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs by importance
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Multiply each channel by its importance weight
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over all channels
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # ReLU - only positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def overlay_on_image(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            original_img: Original image (H, W, C) in RGB
            heatmap: Grad-CAM heatmap (H', W')
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap
            
        Returns:
            Overlaid image in RGB
        """
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(
            heatmap,
            (original_img.shape[1], original_img.shape[0])
        )
        
        # Convert heatmap to uint8
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Convert BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is uint8
        if original_img.dtype != np.uint8:
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)
            else:
                original_img = original_img.astype(np.uint8)
        
        # Blend images
        overlay = cv2.addWeighted(
            original_img,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return overlay
    
    def save_visualization(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        out_path: Path,
        patient_id: str,
        prob: float,
        alpha: float = 0.4
    ) -> None:
        """
        Save Grad-CAM visualization with patient info.
        
        Args:
            original_img: Original image (H, W, C) in RGB
            heatmap: Grad-CAM heatmap
            out_path: Output file path
            patient_id: Patient identifier
            prob: Malignancy probability
            alpha: Transparency of overlay
        """
        # Create overlay
        overlay = self.overlay_on_image(original_img, heatmap, alpha=alpha)
        
        # Add info bar at bottom
        bar_height = 40
        img_with_bar = np.zeros(
            (overlay.shape[0] + bar_height, overlay.shape[1], 3),
            dtype=np.uint8
        )
        img_with_bar[:overlay.shape[0], :, :] = overlay
        img_with_bar[overlay.shape[0]:, :, :] = [255, 255, 255]  # White bar
        
        # Add text
        text = f"Patient: {patient_id} | Malignancy Prob: {prob:.3f}"
        cv2.putText(
            img_with_bar,
            text,
            (10, overlay.shape[0] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        # Ensure output directory exists
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for cv2.imwrite
        img_bgr = cv2.cvtColor(img_with_bar, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(str(out_path), img_bgr)
    
    def validate_heatmap_focus(
        self,
        heatmap: np.ndarray,
        lung_mask: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> dict:
        """
        Validate that heatmap focuses on lung regions.
        
        Args:
            heatmap: Grad-CAM heatmap (H, W)
            lung_mask: Optional binary mask of lung regions
            threshold: Threshold for high-activation regions
            
        Returns:
            Dictionary with validation metrics
        """
        # Find high-activation regions
        high_activation = (heatmap >= threshold).astype(float)
        
        # Calculate center of mass
        y_coords, x_coords = np.where(high_activation > 0)
        
        if len(y_coords) == 0:
            return {
                'valid': False,
                'reason': 'No high-activation regions found'
            }
        
        center_y = np.mean(y_coords) / heatmap.shape[0]
        center_x = np.mean(x_coords) / heatmap.shape[1]
        
        # Check if center is in reasonable lung region (center of image)
        # Lungs typically occupy central 60% of image
        in_center = (0.2 <= center_x <= 0.8) and (0.2 <= center_y <= 0.8)
        
        # Check if activation is not concentrated at edges
        edge_activation = (
            high_activation[0, :].sum() +
            high_activation[-1, :].sum() +
            high_activation[:, 0].sum() +
            high_activation[:, -1].sum()
        )
        total_activation = high_activation.sum()
        edge_ratio = edge_activation / (total_activation + 1e-7)
        
        valid = in_center and (edge_ratio < 0.3)
        
        return {
            'valid': valid,
            'center_x': float(center_x),
            'center_y': float(center_y),
            'edge_ratio': float(edge_ratio),
            'in_center': in_center
        }
