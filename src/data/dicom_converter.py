"""
DICOM to PNG converter with MONOCHROME1 handling and CLAHE preprocessing.
"""
import cv2
import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple
from loguru import logger
from tqdm import tqdm

from src.config import config


class DICOMConverter:
    """Converts DICOM files to preprocessed PNG images"""
    
    def __init__(self):
        self.image_size = config.data.image_size
        
    def convert(self, dicom_path: Path, out_path: Path) -> bool:
        """
        Convert single DICOM file to PNG with preprocessing.
        
        Args:
            dicom_path: Path to DICOM file
            out_path: Output PNG path
            
        Returns:
            bool: Success status
        """
        try:
            # Read DICOM
            dcm = pydicom.dcmread(str(dicom_path))
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            # Handle MONOCHROME1 (inverted grayscale)
            if hasattr(dcm, 'PhotometricInterpretation'):
                if dcm.PhotometricInterpretation == 'MONOCHROME1':
                    pixel_array = pixel_array.max() - pixel_array
            
            # Normalize to [0, 255]
            pixel_array = pixel_array - pixel_array.min()
            pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            pixel_array = clahe.apply(pixel_array)
            
            # Convert grayscale to 3-channel BGR
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
            
            # Resize to target size
            pixel_array = cv2.resize(
                pixel_array,
                self.image_size,
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Ensure output directory exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write PNG
            cv2.imwrite(str(out_path), pixel_array)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {dicom_path}: {str(e)}")
            return False
    
    def batch_convert(self, input_dir: Path, output_dir: Path) -> dict:
        """
        Batch convert all DICOM files in directory.
        
        Args:
            input_dir: Directory containing DICOM files
            output_dir: Output directory for PNGs
            
        Returns:
            dict: Statistics with success, failed, failed_files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all DICOM files
        dicom_files = list(input_dir.rglob('*.dcm'))
        
        success_count = 0
        failed_count = 0
        failed_files = []
        
        logger.info(f"Found {len(dicom_files)} DICOM files to convert")
        
        for dicom_path in tqdm(dicom_files, desc="Converting DICOMs"):
            # Preserve relative directory structure
            rel_path = dicom_path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix('.png')
            
            if self.convert(dicom_path, out_path):
                success_count += 1
            else:
                failed_count += 1
                failed_files.append(str(dicom_path))
        
        stats = {
            'success': success_count,
            'failed': failed_count,
            'failed_files': failed_files
        }
        
        logger.info(f"Conversion complete: {success_count} success, {failed_count} failed")
        
        return stats
