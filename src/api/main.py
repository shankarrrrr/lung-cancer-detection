"""
FastAPI application for lung cancer detection API.
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
from typing import Optional
from loguru import logger
import sys

from src.api.schemas import PredictionResponse, HealthResponse, ErrorResponse
from src.inference.predictor import LungCancerPredictor
from src.config import config

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/api.log", rotation="500 MB", level="DEBUG")

# Initialize FastAPI app
app = FastAPI(
    title="Lung Cancer Detection API",
    description="AI-powered lung cancer detection from chest X-rays with explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[LungCancerPredictor] = None

# Directories
UPLOAD_DIR = Path("uploads")
STATIC_DIR = Path("static")
HEATMAP_DIR = STATIC_DIR / "heatmaps"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor
    try:
        logger.info("Loading lung cancer detection model...")
        predictor = LungCancerPredictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        predictor = None


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns service status and model availability.
    """
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Prediction"]
)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image (PNG, JPEG, or DICOM)"),
    patient_id: str = Form(..., description="Patient identifier"),
    age: Optional[float] = Form(None, description="Patient age in years"),
    smoking_pack_years: Optional[float] = Form(None, description="Smoking pack-years"),
    symptom_score: Optional[float] = Form(None, description="Symptom severity score (0-10)")
):
    """
    Predict lung cancer malignancy from chest X-ray.
    
    Accepts chest X-ray images and optional clinical metadata.
    Returns malignancy probability, risk level, bounding box, and Grad-CAM heatmap.
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable."
        )
    
    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.dcm'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{patient_id}_{file.filename}"
    
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {upload_path}")
        
        # Prepare metadata
        metadata = None
        if age is not None or smoking_pack_years is not None or symptom_score is not None:
            metadata = {
                'age': age,
                'smoking_pack_years': smoking_pack_years,
                'symptom_score': symptom_score
            }
        
        # Generate heatmap path
        heatmap_path = HEATMAP_DIR / f"{patient_id}_heatmap.png"
        
        # Run prediction
        result = predictor.predict(
            img_path=upload_path,
            metadata=metadata,
            patient_id=patient_id,
            save_heatmap_path=heatmap_path
        )
        
        # Convert heatmap path to URL
        if result['heatmap_url']:
            result['heatmap_url'] = f"/static/heatmaps/{patient_id}_heatmap.png"
        
        logger.info(f"Prediction successful for patient {patient_id}")
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    
    finally:
        # Clean up uploaded file
        if upload_path.exists():
            upload_path.unlink()


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get API metrics (placeholder for production monitoring).
    """
    return {
        "total_predictions": 0,
        "average_response_time_ms": 0,
        "model_version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
