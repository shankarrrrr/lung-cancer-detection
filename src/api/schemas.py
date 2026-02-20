"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    patient_id: str = Field(..., description="Patient identifier")
    malignancy_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of malignancy")
    uncertainty_std: float = Field(..., ge=0.0, description="Uncertainty standard deviation from MC Dropout")
    risk_level: RiskLevel = Field(..., description="Risk classification: HIGH, MODERATE, or LOW")
    bbox_normalized: List[float] = Field(..., min_length=4, max_length=4, description="Normalized bounding box [x1, y1, x2, y2]")
    bbox_confidence: float = Field(..., ge=0.0, le=1.0, description="Bounding box confidence score")
    recommendation: str = Field(..., description="Clinical recommendation based on risk level")
    heatmap_url: Optional[str] = Field(None, description="URL to Grad-CAM heatmap visualization")
    heatmap_valid: bool = Field(..., description="Whether heatmap focuses on lung regions")
    disclaimer: str = Field(..., description="Medical disclaimer")
    
    @field_validator('bbox_normalized')
    @classmethod
    def validate_bbox(cls, v):
        """Validate bounding box coordinates"""
        if len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        if not all(0 <= coord <= 1 for coord in v):
            raise ValueError("Bounding box coordinates must be normalized between 0 and 1")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P12345",
                "malignancy_probability": 0.87,
                "uncertainty_std": 0.03,
                "risk_level": "HIGH",
                "bbox_normalized": [0.35, 0.25, 0.65, 0.55],
                "bbox_confidence": 0.92,
                "recommendation": "HIGH RISK (p=0.870): Immediate CT scan and specialist consultation recommended.",
                "heatmap_url": "/static/heatmaps/P12345_heatmap.png",
                "heatmap_valid": True,
                "disclaimer": "AI-assisted screening tool. Not a replacement for clinical judgment. Results must be reviewed by a qualified radiologist."
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded successfully")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid file format",
                "detail": "Only PNG, JPEG, and DICOM files are supported"
            }
        }
