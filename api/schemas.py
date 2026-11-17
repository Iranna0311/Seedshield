"""
SeedShield - API Schemas
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class SeedClass(str, Enum):
    """Seed classification classes."""
    REAL = "REAL"
    FAKE = "FAKE"
    LOW_QUALITY = "LOW_QUALITY"
    UNCERTAIN = "UNCERTAIN"


class VerdictType(str, Enum):
    """Verdict types."""
    GENUINE = "genuine"
    SUSPECTED_FAKE = "suspected_fake"
    LOW_QUALITY = "low_quality"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x: int = Field(..., description="X coordinate of top-left corner")
    y: int = Field(..., description="Y coordinate of top-left corner")
    width: int = Field(..., description="Width of bounding box")
    height: int = Field(..., description="Height of bounding box")


class SeedDetectionResult(BaseModel):
    """Result for a single seed detection."""
    id: int = Field(..., description="Unique ID for this detection")
    bbox: List[int] = Field(..., description="Bounding box [x, y, width, height]")
    class_name: SeedClass = Field(..., description="Classification result")
    class_id: int = Field(..., description="Class ID (0=REAL, 1=FAKE, 2=LOW_QUALITY, 3=UNCERTAIN)")
    detection_confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    classification_confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    bbox_norm: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0], description="Normalized bbox [x/w, y/h, bw/w, bh/h]")
    probs: Dict[str, float] = Field(default_factory=lambda: {"REAL": 0.0, "FAKE": 0.0, "LOW_QUALITY": 0.0}, description="Per-class probabilities")
    
    class Config:
        use_enum_values = True


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    success: bool = Field(True, description="Whether prediction was successful")
    
    # Counts
    total_seeds: int = Field(..., ge=0, description="Total number of detected seeds")
    real_count: int = Field(..., ge=0, description="Number of real seeds")
    fake_count: int = Field(..., ge=0, description="Number of fake seeds")
    low_quality_count: int = Field(..., ge=0, description="Number of low quality seeds")
    uncertain_count: int = Field(0, ge=0, description="Number of uncertain seeds")
    
    # Percentages
    real_percentage: float = Field(..., ge=0, le=100, description="Percentage of real seeds")
    fake_percentage: float = Field(..., ge=0, le=100, description="Percentage of fake seeds")
    low_quality_percentage: float = Field(..., ge=0, le=100, description="Percentage of low quality seeds")
    uncertain_percentage: float = Field(0.0, ge=0, le=100, description="Percentage of uncertain seeds")
    
    # Verdict
    verdict: VerdictType = Field(..., description="Final verdict")
    verdict_message: str = Field(..., description="Human-readable verdict message")
    
    # Statistical extras
    ci95: List[float] = Field(default_factory=lambda: [0.0, 0.0], description="95% Wilson CI for fake proportion [low, high]")
    model_version: str = Field("", description="Model version string")
    scan_id: str = Field("", description="Unique scan identifier (uuid4)")
    
    # Detections
    detections: List[SeedDetectionResult] = Field(default_factory=list, description="Individual seed detections")
    
    # Image
    annotated_image_base64: Optional[str] = Field(None, description="Base64 encoded annotated image")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "success": True,
                "total_seeds": 10,
                "real_count": 7,
                "fake_count": 2,
                "low_quality_count": 1,
                "uncertain_count": 0,
                "real_percentage": 70.0,
                "fake_percentage": 20.0,
                "low_quality_percentage": 10.0,
                "uncertain_percentage": 0.0,
                "verdict": "genuine",
                "verdict_message": "✅ Mostly Genuine",
                "ci95": [0.025, 0.415],
                "model_version": "1.0.0-sklearn",
                "scan_id": "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx",
                "detections": [
                    {
                        "id": 0,
                        "bbox": [100, 150, 50, 45],
                        "class_name": "REAL",
                        "class_id": 0,
                        "detection_confidence": 0.95,
                        "classification_confidence": 0.92,
                        "bbox_norm": [0.1, 0.15, 0.05, 0.045],
                        "probs": {"REAL": 0.92, "FAKE": 0.05, "LOW_QUALITY": 0.03}
                    }
                ],
                "annotated_image_base64": "..."
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid image format",
                "detail": "Supported formats: JPEG, PNG, WebP"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    device: str = Field(..., description="Compute device (cpu/cuda)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": True,
                "device": "cuda"
            }
        }


class StatsResponse(BaseModel):
    """Statistics response for processed images."""
    total_predictions: int = Field(..., description="Total predictions made")
    avg_processing_time_ms: float = Field(..., description="Average processing time in milliseconds")
    class_distribution: Dict[str, int] = Field(..., description="Distribution of detected classes")
    verdict_distribution: Dict[str, int] = Field(..., description="Distribution of verdicts")


class ConfigResponse(BaseModel):
    """Configuration response."""
    detector_confidence_threshold: float
    classifier_confidence_threshold: float
    fake_verdict_threshold: float
    low_quality_verdict_threshold: float
    max_detections: int
    image_size: int
