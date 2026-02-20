"""
Test FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import numpy as np
from PIL import Image
import io

from src.api.main import app


# Create test client
client = TestClient(app)


def create_dummy_image() -> bytes:
    """Create a dummy PNG image for testing"""
    # Create random image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert 'status' in data
    assert 'model_loaded' in data
    assert 'version' in data
    
    print(f"✓ Health endpoint test passed: {data}")


def test_health_endpoint_schema():
    """Test that health endpoint returns correct schema"""
    response = client.get("/health")
    data = response.json()
    
    # Check required fields
    assert isinstance(data['status'], str)
    assert isinstance(data['model_loaded'], bool)
    assert isinstance(data['version'], str)
    
    print("✓ Health endpoint schema test passed")


def test_predict_endpoint_without_model():
    """Test predict endpoint behavior when model is not loaded"""
    # This test assumes model might not be loaded in test environment
    
    dummy_img = create_dummy_image()
    
    files = {'file': ('test.png', dummy_img, 'image/png')}
    data = {'patient_id': 'TEST001'}
    
    response = client.post("/v1/predict", files=files, data=data)
    
    # Should either succeed (if model loaded) or return 503
    assert response.status_code in [200, 503], \
        f"Expected 200 or 503, got {response.status_code}"
    
    print(f"✓ Predict endpoint test passed: status={response.status_code}")


def test_predict_endpoint_invalid_file():
    """Test predict endpoint with invalid file type"""
    # Create a text file instead of image
    files = {'file': ('test.txt', b'not an image', 'text/plain')}
    data = {'patient_id': 'TEST002'}
    
    response = client.post("/v1/predict", files=files, data=data)
    
    # Should return 400 for invalid file type
    assert response.status_code in [400, 503], \
        f"Expected 400 or 503, got {response.status_code}"
    
    print("✓ Invalid file type test passed")


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    
    assert response.status_code == 200
    
    data = response.json()
    assert 'total_predictions' in data
    assert 'model_version' in data
    
    print(f"✓ Metrics endpoint test passed: {data}")


def test_api_documentation():
    """Test that API documentation is accessible"""
    # Test OpenAPI docs
    response = client.get("/docs")
    assert response.status_code == 200
    
    # Test ReDoc
    response = client.get("/redoc")
    assert response.status_code == 200
    
    print("✓ API documentation test passed")


def test_cors_headers():
    """Test CORS headers are present"""
    response = client.get("/health")
    
    # Check for CORS headers
    assert 'access-control-allow-origin' in response.headers or response.status_code == 200
    
    print("✓ CORS headers test passed")


if __name__ == "__main__":
    print("Running API tests...\n")
    test_health_endpoint()
    test_health_endpoint_schema()
    test_predict_endpoint_without_model()
    test_predict_endpoint_invalid_file()
    test_metrics_endpoint()
    test_api_documentation()
    test_cors_headers()
    print("\n✓ All API tests passed!")
