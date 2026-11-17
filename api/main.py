"""
SeedShield - FastAPI Backend
Production-ready API for seed detection and classification.
"""

import os
import sys
import io
import time
import uuid
import base64
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
import logging

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# Image processing
from PIL import Image, ImageOps
import numpy as np
import cv2

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import schemas
from api.schemas import (
    PredictionResponse, ErrorResponse, HealthResponse,
    SeedDetectionResult, ConfigResponse
)

# Import inference pipeline
try:
    from inference.pipeline import SeedShieldPipeline, create_demo_prediction
    from inference.visualizer import (
        create_result_visualization, draw_legend, numpy_to_base64
    )
    from models.model_config import (
        get_inference_config, get_path_config, VERDICT_MESSAGES, CLASS_NAMES
    )
except ImportError:
    # Alternative import paths
    from seedshield.inference.pipeline import SeedShieldPipeline, create_demo_prediction
    from seedshield.inference.visualizer import (
        create_result_visualization, draw_legend, numpy_to_base64
    )
    from seedshield.models.model_config import (
        get_inference_config, get_path_config, VERDICT_MESSAGES, CLASS_NAMES
    )


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global variables
pipeline: Optional[SeedShieldPipeline] = None
config = get_inference_config()
paths = get_path_config()
stats = {
    'total_predictions': 0,
    'processing_times': [],
    'class_counts': {'REAL': 0, 'FAKE': 0, 'LOW_QUALITY': 0},
    'verdict_counts': {'genuine': 0, 'suspected_fake': 0, 'low_quality': 0}
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load models on startup."""
    global pipeline
    
    logger.info("Starting SeedShield API...")
    
    # Check if we should run in demo mode
    demo_mode = os.getenv("SEEDSHIELD_DEMO_MODE", "false").lower() == "true"
    
    if demo_mode:
        logger.info("Running in DEMO MODE - models will not be loaded")
        pipeline = None
    else:
        try:
            logger.info("Loading inference pipeline...")
            pipeline = SeedShieldPipeline()
            logger.info("Pipeline loaded successfully!")
        except Exception as e:
            logger.warning(f"Could not load pipeline: {e}")
            logger.info("Falling back to demo mode")
            pipeline = None
    
    app.state.pipeline = pipeline
    yield
    
    # Cleanup
    logger.info("Shutting down SeedShield API...")


# Create FastAPI application
app = FastAPI(
    title="SeedShield API",
    description="""
    ## SeedShield - AI-Powered Seed Authenticity Detection
    
    Detect FAKE, REAL, and LOW QUALITY seeds from smartphone images.
    
    ### Features:
    - **Seed Detection**: Automatically detect individual seeds in images
    - **Classification**: Classify each seed as REAL, FAKE, or LOW_QUALITY
    - **Visual Results**: Get annotated images with color-coded bounding boxes
    - **Verdict**: Receive overall assessment of seed batch quality
    
    ### Colors:
    - 🟢 **Green**: REAL seeds
    - 🔴 **Red**: FAKE seeds
    - 🟡 **Yellow**: LOW QUALITY seeds
    """,
    version="1.0.0",
    lifespan=lifespan
)


# CORS configuration - Connected to Vercel and local dev
raw_origins = os.getenv("ALLOWED_ORIGINS", "https://seedshield.vercel.app,http://localhost:3000,http://127.0.0.1:3000")
if raw_origins == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    if "https://seedshield.vercel.app" not in allowed_origins:
        allowed_origins.append("https://seedshield.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request-ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.time()
        response: Response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"{request.method} {request.url.path} → {response.status_code} in {elapsed_ms:.1f}ms [rid={request_id}]")
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(RequestIDMiddleware)


# Constants
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_BYTES = 8 * 1024 * 1024  # 8 MB
MAX_FILE_SIZE = MAX_BYTES  # keep old name for compat
Image.MAX_IMAGE_PIXELS = 25_000_000


def validate_image(file: UploadFile) -> bool:
    """Validate uploaded image file."""
    # Check extension
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    return True


def process_demo_image(image: np.ndarray) -> dict:
    """Process image in demo mode using contour detection to find actual seeds.
    
    DETERMINISTIC: Same image will always produce the same results.
    """
    import hashlib
    
    h, w = image.shape[:2]
    
    # Create a hash of the image for deterministic random seed
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    image_seed = int(image_hash[:8], 16)  # Use first 8 hex chars as seed
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use Otsu's thresholding for better binarization
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Also try adaptive thresholding
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Combine both thresholding methods
    thresh = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
    
    # Morphological operations to separate touching seeds
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use smaller minimum area to detect small seeds
    min_area = max(50, (h * w) * 0.0005)  # At least 0.05% of image or 50 pixels
    max_area = (h * w) * 0.25   # At most 25% of image
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            valid_contours.append(cnt)
    
    detections = []
    
    if len(valid_contours) == 0:
        # No contours found - try edge detection approach
        edges = cv2.Canny(blurred, 30, 100)
        # Dilate edges to close gaps
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                valid_contours.append(cnt)
    
    if len(valid_contours) == 0:
        raise HTTPException(
            status_code=400,
            detail="No seeds detected in this image. Only seed photos are accepted. Please place seeds on a plain background and try again."
        )
    else:
        # Create detections from actual contours - DETERMINISTIC classification
        # Sort by area (largest first) and limit to 100 seeds max
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:100]
        
        for i, cnt in enumerate(valid_contours):
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Classify based on contour properties (deterministic)
            aspect_ratio = bw / bh if bh > 0 else 1
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Extract region for color analysis
            roi = image[max(0,y):min(h,y+bh), max(0,x):min(w,x+bw)]
            mean_brightness = np.mean(roi) if roi.size > 0 else 128
            
            # Deterministic classification based on shape and color
            if 0.6 < aspect_ratio < 1.8 and circularity > 0.4:
                # Good oval/circular shape = likely real seed
                cls = 'REAL'
                conf = min(0.95, 0.75 + circularity * 0.2)
            elif aspect_ratio < 0.4 or aspect_ratio > 2.5:
                # Very elongated or flat = damaged/low quality
                cls = 'LOW_QUALITY'
                conf = 0.70 + (1 - min(circularity, 0.5)) * 0.2
            elif mean_brightness < 80 or mean_brightness > 220:
                # Unusually dark or bright = suspicious
                cls = 'FAKE'
                conf = 0.75
            else:
                # Default to real for normal-looking objects
                cls = 'REAL'
                conf = 0.80 + circularity * 0.1
            
            detections.append({
                'id': i,
                'bbox': (x, y, bw, bh),
                'class_name': cls,
                'class_id': CLASS_NAMES.index(cls),
                'detection_confidence': round(conf, 2),
                'classification_confidence': round(conf, 2),
                'confidence': round(conf, 2)
            })
    
    # Calculate stats
    total = len(detections)
    real_count = sum(1 for d in detections if d['class_name'] == 'REAL')
    fake_count = sum(1 for d in detections if d['class_name'] == 'FAKE')
    low_quality_count = sum(1 for d in detections if d['class_name'] == 'LOW_QUALITY')
    
    # Determine verdict
    if total == 0:
        verdict = "genuine"
    else:
        fake_ratio = fake_count / total
        low_quality_ratio = low_quality_count / total
        
        if fake_ratio > 0.3:
            verdict = "suspected_fake"
        elif low_quality_ratio > 0.4 or (fake_ratio + low_quality_ratio) > 0.5:
            verdict = "low_quality"
        else:
            verdict = "genuine"
    
    # Create visualization
    stats_dict = {'REAL': real_count, 'FAKE': fake_count, 'LOW_QUALITY': low_quality_count}
    annotated = create_result_visualization(image, detections, verdict, stats_dict)
    annotated = draw_legend(annotated)
    annotated_b64 = numpy_to_base64(annotated)
    
    return {
        'total_seeds': total,
        'real_count': real_count,
        'fake_count': fake_count,
        'low_quality_count': low_quality_count,
        'real_percentage': round(real_count / total * 100, 2) if total > 0 else 0,
        'fake_percentage': round(fake_count / total * 100, 2) if total > 0 else 0,
        'low_quality_percentage': round(low_quality_count / total * 100, 2) if total > 0 else 0,
        'verdict': verdict,
        'verdict_message': VERDICT_MESSAGES[verdict],
        'detections': detections,
        'annotated_image_base64': annotated_b64
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SeedShield API",
        "version": "1.0.0",
        "description": "AI-powered seed authenticity detection",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=pipeline is not None,
        device=pipeline.device if pipeline else "demo_mode"
    )


@app.get("/ready", tags=["Health"])
async def ready_check():
    """Readiness probe."""
    return {
        "ready": True,
        "models_loaded": pipeline is not None,
        "model_version": "1.0.0-sklearn",
    }


@app.post("/predict",
          response_model=PredictionResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
          tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Image file containing seeds"),
    return_image: bool = Query(True, description="Whether to return annotated image"),
    confidence_threshold: float = Query(0.3, ge=0.1, le=0.9, description="Detection confidence threshold")
):
    """
    Analyze an image of seeds and detect fake/real/low-quality seeds.
    
    Upload an image containing seeds. The API will:
    1. Detect all individual seeds
    2. Classify each seed as REAL, FAKE, or LOW_QUALITY
    3. Return annotated image with color-coded bounding boxes
    4. Provide overall verdict on seed batch quality
    
    ### Request:
    - **file**: Image file (JPEG, PNG, WebP)
    - **return_image**: Whether to include base64 annotated image in response
    - **confidence_threshold**: Minimum confidence for detections (0.1-0.9)
    
    ### Response:
    - Total seed count and breakdown by class
    - Percentage distribution
    - Final verdict (Genuine / Suspected Fake / Low Quality)
    - Individual detection details
    - Base64 encoded annotated image (optional)
    """
    global stats
    start_time = time.time()
    
    try:
        # Validate file
        validate_image(file)
        
        # Read image with size cap
        contents = await file.read(MAX_BYTES + 1)
        if len(contents) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_BYTES // 1024 // 1024} MB"
            )
        
        # PIL validation + EXIF correction
        try:
            img_pil = Image.open(io.BytesIO(contents))
            img_pil.verify()
            img_pil = Image.open(io.BytesIO(contents))  # reopen after verify
            img_pil = ImageOps.exif_transpose(img_pil)
            img_pil = img_pil.convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Convert validated PIL image to OpenCV BGR for downstream code
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        logger.info(f"Processing image: {file.filename}, shape: {image.shape}")
        
        # Process image
        if pipeline is not None:
            # Use actual pipeline (blocking call → threadpool)
            try:
                result = await run_in_threadpool(
                    pipeline.process_image, image, return_image
                )
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

            if result.total_seeds == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No seeds detected in this image. Only seed photos are accepted. Please place seeds on a plain background and try again."
                )
            
            response_data = {
                'success': True,
                'total_seeds': result.total_seeds,
                'real_count': result.real_count,
                'fake_count': result.fake_count,
                'low_quality_count': result.low_quality_count,
                'uncertain_count': result.uncertain_count,
                'real_percentage': result.real_percentage,
                'fake_percentage': result.fake_percentage,
                'low_quality_percentage': result.low_quality_percentage,
                'uncertain_percentage': result.uncertain_percentage,
                'verdict': result.verdict,
                'verdict_message': result.verdict_message,
                'ci95': result.ci95,
                'model_version': result.model_version,
                'scan_id': result.scan_id,
                'detections': result.detections,
                'annotated_image_base64': result.annotated_image_base64 if return_image else None
            }
        else:
            # Demo mode
            logger.info("Using demo mode for prediction")
            demo_result = process_demo_image(image)
            response_data = {
                'success': True,
                **demo_result,
                'uncertain_count': 0,
                'uncertain_percentage': 0.0,
                'ci95': [0.0, 0.0],
                'model_version': '1.0.0-sklearn',
                'scan_id': str(uuid.uuid4()),
                'annotated_image_base64': demo_result['annotated_image_base64'] if return_image else None
            }
            # Convert detections to proper format
            response_data['detections'] = [
                {
                    'id': d['id'],
                    'bbox': list(d['bbox']),
                    'class_name': d['class_name'],
                    'class_id': d['class_id'],
                    'detection_confidence': d['detection_confidence'],
                    'classification_confidence': d['classification_confidence'],
                    'bbox_norm': d.get('bbox_norm', [0.0, 0.0, 0.0, 0.0]),
                    'probs': d.get('probs', {"REAL": 0.0, "FAKE": 0.0, "LOW_QUALITY": 0.0}),
                }
                for d in demo_result['detections']
            ]
        
        # Update stats
        processing_time = (time.time() - start_time) * 1000
        stats['total_predictions'] += 1
        stats['processing_times'].append(processing_time)
        if len(stats['processing_times']) > 100:
            stats['processing_times'] = stats['processing_times'][-100:]
        
        stats['class_counts']['REAL'] += response_data['real_count']
        stats['class_counts']['FAKE'] += response_data['fake_count']
        stats['class_counts']['LOW_QUALITY'] += response_data['low_quality_count']
        stats['verdict_counts'][response_data['verdict']] += 1
        
        logger.info(f"Prediction complete in {processing_time:.1f}ms - Verdict: {response_data['verdict_message']}")
        
        return PredictionResponse(**response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/predict/image",
          responses={400: {"model": ErrorResponse}},
          tags=["Prediction"])
async def predict_image_only(
    file: UploadFile = File(..., description="Image file containing seeds")
):
    """
    Get only the annotated image (for direct display).
    
    Returns the annotated image as JPEG directly (not base64 encoded).
    Useful for embedding directly in HTML img tags.
    """
    try:
        # Validate and process
        validate_image(file)
        contents = await file.read(MAX_BYTES + 1)
        if len(contents) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="File too large. Maximum size: 8 MB")

        # PIL validation + EXIF correction
        try:
            img_pil = Image.open(io.BytesIO(contents))
            img_pil.verify()
            img_pil = Image.open(io.BytesIO(contents))
            img_pil = ImageOps.exif_transpose(img_pil)
            img_pil = img_pil.convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image")

        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Process
        if pipeline is not None:
            result = pipeline.process_image(image, return_annotated=False)
            annotated = pipeline.get_annotated_image(image, result)
        else:
            demo_result = process_demo_image(image)
            # Decode the base64 back to image
            img_data = base64.b64decode(demo_result['annotated_image_base64'])
            nparr = np.frombuffer(img_data, np.uint8)
            annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=annotated_{file.filename}"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """Get prediction statistics."""
    avg_time = sum(stats['processing_times']) / len(stats['processing_times']) if stats['processing_times'] else 0
    
    return {
        "total_predictions": stats['total_predictions'],
        "avg_processing_time_ms": round(avg_time, 2),
        "class_distribution": stats['class_counts'],
        "verdict_distribution": stats['verdict_counts']
    }


@app.get("/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """Get current API configuration."""
    return ConfigResponse(
        detector_confidence_threshold=config.detector_conf_threshold,
        classifier_confidence_threshold=config.classifier_conf_threshold,
        fake_verdict_threshold=config.fake_threshold,
        low_quality_verdict_threshold=config.low_quality_threshold,
        max_detections=config.max_detections,
        image_size=640
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SeedShield API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    
    args = parser.parse_args()
    
    if args.demo:
        os.environ["SEEDSHIELD_DEMO_MODE"] = "true"
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
