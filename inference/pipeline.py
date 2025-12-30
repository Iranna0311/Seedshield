"""
SeedShield - Inference Pipeline
Complete pipeline for seed detection and classification from a single image.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
from PIL import Image

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from inference.visualizer import (
        draw_seed_detection, draw_verdict_overlay, draw_legend,
        create_result_visualization, numpy_to_base64, save_annotated_image
    )
    from models.model_config import (
        get_inference_config, get_path_config,
        CLASS_NAMES, ID_TO_CLASS, VERDICT_MESSAGES
    )
except ImportError:
    from visualizer import (
        draw_seed_detection, draw_verdict_overlay, draw_legend,
        create_result_visualization, numpy_to_base64, save_annotated_image
    )
    sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
    from model_config import (
        get_inference_config, get_path_config,
        CLASS_NAMES, ID_TO_CLASS, VERDICT_MESSAGES
    )


@dataclass
class SeedDetection:
    """Data class for a single seed detection."""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    class_name: str
    class_id: int
    detection_confidence: float
    classification_confidence: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InferenceResult:
    """Data class for complete inference result."""
    total_seeds: int
    real_count: int
    fake_count: int
    low_quality_count: int
    real_percentage: float
    fake_percentage: float
    low_quality_percentage: float
    verdict: str
    verdict_message: str
    detections: List[Dict]
    annotated_image_base64: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = asdict(self)
        # Remove base64 if too large for JSON response
        if self.annotated_image_base64 and len(self.annotated_image_base64) > 1000000:
            result['annotated_image_base64'] = None
            result['image_available'] = True
        return result


class SeedShieldPipeline:
    """
    Complete inference pipeline for SeedShield.
    
    Pipeline steps:
    1. Load image
    2. Detect all seeds using YOLOv8
    3. Crop each detected seed
    4. Classify each seed using MobileNetV2
    5. Draw bounding boxes with colors
    6. Calculate final verdict
    7. Return annotated image and JSON results
    """
    
    def __init__(self, 
                 detector_weights: str = None,
                 classifier_weights: str = None,
                 device: str = "auto"):
        """
        Initialize the pipeline.
        
        Args:
            detector_weights: Path to YOLOv8 detector weights
            classifier_weights: Path to MobileNetV2 classifier weights
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.config = get_inference_config()
        self.paths = get_path_config()
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"SeedShield Pipeline initialized on: {self.device}")
        
        # Model paths
        self.detector_weights = detector_weights or str(
            self.paths.detector_weights / "seedshield_detector" / "weights" / "best.pt"
        )
        self.classifier_weights = classifier_weights or str(
            self.paths.classifier_weights / "best_classifier.pth"
        )
        
        # Models (lazy loading)
        self._detector = None
        self._classifier = None
        
        # Classification preprocessing
        self._classifier_transform = None
    
    @property
    def detector(self):
        """Lazy load the detector model."""
        if self._detector is None:
            self._load_detector()
        return self._detector
    
    @property
    def classifier(self):
        """Lazy load the classifier model."""
        if self._classifier is None:
            self._load_classifier()
        return self._classifier
    
    def _load_detector(self):
        """Load the YOLOv8 detector model."""
        from ultralytics import YOLO
        
        print(f"Loading detector from: {self.detector_weights}")
        
        if Path(self.detector_weights).exists():
            self._detector = YOLO(self.detector_weights)
        else:
            # Fall back to pretrained model for testing
            print("Warning: Trained detector not found. Using pretrained YOLOv8n.")
            self._detector = YOLO("yolov8n.pt")
        
        print("Detector loaded successfully")
    
    def _load_classifier(self):
        """Load the MobileNetV2 classifier model."""
        from torchvision import transforms
        
        # Import classifier model
        try:
            from models.train_classifier import SeedClassifier
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
            from train_classifier import SeedClassifier
        
        print(f"Loading classifier from: {self.classifier_weights}")
        
        # Build model
        self._classifier = SeedClassifier(num_classes=3, pretrained=False)
        
        # Load weights if available
        if Path(self.classifier_weights).exists():
            checkpoint = torch.load(self.classifier_weights, map_location=self.device)
            self._classifier.load_state_dict(checkpoint['model_state_dict'])
            print("Classifier loaded successfully")
        else:
            print("Warning: Trained classifier not found. Using untrained model.")
        
        self._classifier = self._classifier.to(self.device)
        self._classifier.eval()
        
        # Setup transforms
        self._classifier_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_source: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Load image from various sources.
        
        Args:
            image_source: File path, numpy array, or PIL Image
        
        Returns:
            Image in BGR format (OpenCV)
        """
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
            if image is None:
                raise ValueError(f"Could not load image: {image_source}")
        elif isinstance(image_source, Image.Image):
            image = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        elif isinstance(image_source, np.ndarray):
            if len(image_source.shape) == 2:
                image = cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
            elif image_source.shape[2] == 4:
                image = cv2.cvtColor(image_source, cv2.COLOR_RGBA2BGR)
            else:
                image = image_source.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")
        
        return image
    
    def detect_seeds(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all seeds in the image using YOLOv8.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detection dictionaries with bounding boxes
        """
        # Run detection
        results = self.detector.predict(
            source=image,
            conf=self.config.detector_conf_threshold,
            iou=self.config.detector_iou_threshold,
            max_det=self.config.max_detections,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'detection_confidence': conf,
                    'detection_class_id': cls_id
                })
        
        return detections
    
    def crop_seed(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                  padding: int = 5) -> np.ndarray:
        """
        Crop a seed region from the image.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            padding: Padding around the crop
        
        Returns:
            Cropped seed image
        """
        h, w = image.shape[:2]
        x, y, bw, bh = bbox
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)
        
        crop = image[y1:y2, x1:x2]
        
        # Ensure minimum size
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        
        return crop
    
    def classify_seed(self, seed_image: np.ndarray) -> Tuple[str, int, float]:
        """
        Classify a single seed image.
        
        Args:
            seed_image: Cropped seed image (BGR format)
        
        Returns:
            Tuple of (class_name, class_id, confidence)
        """
        # Ensure classifier is loaded
        _ = self.classifier
        
        # Convert to PIL and apply transforms
        seed_rgb = cv2.cvtColor(seed_image, cv2.COLOR_BGR2RGB)
        seed_pil = Image.fromarray(seed_rgb)
        seed_tensor = self._classifier_transform(seed_pil).unsqueeze(0)
        seed_tensor = seed_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self._classifier(seed_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, class_id = torch.max(probs, dim=1)
        
        class_id = int(class_id.cpu().numpy()[0])
        confidence = float(confidence.cpu().numpy()[0])
        class_name = CLASS_NAMES[class_id]
        
        return class_name, class_id, confidence
    
    def calculate_verdict(self, detections: List[Dict]) -> Tuple[str, str]:
        """
        Calculate final verdict based on detection results.
        
        Args:
            detections: List of detection dictionaries with classifications
        
        Returns:
            Tuple of (verdict_key, verdict_message)
        """
        if not detections:
            return "genuine", VERDICT_MESSAGES["genuine"]
        
        total = len(detections)
        fake_count = sum(1 for d in detections if d.get('class_name') == 'FAKE')
        low_quality_count = sum(1 for d in detections if d.get('class_name') == 'LOW_QUALITY')
        
        fake_ratio = fake_count / total
        low_quality_ratio = low_quality_count / total
        
        if fake_ratio > self.config.fake_threshold:
            return "suspected_fake", VERDICT_MESSAGES["suspected_fake"]
        elif low_quality_ratio > self.config.low_quality_threshold:
            return "low_quality", VERDICT_MESSAGES["low_quality"]
        elif (fake_ratio + low_quality_ratio) > 0.5:
            return "low_quality", VERDICT_MESSAGES["low_quality"]
        else:
            return "genuine", VERDICT_MESSAGES["genuine"]
    
    def process_image(self, 
                      image_source: Union[str, np.ndarray, Image.Image],
                      return_annotated: bool = True,
                      save_path: Optional[str] = None) -> InferenceResult:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_source: Input image (path, numpy array, or PIL Image)
            return_annotated: Whether to include annotated image in result
            save_path: Optional path to save annotated image
        
        Returns:
            InferenceResult with all detections and verdict
        """
        # Load image
        image = self.load_image(image_source)
        original_shape = image.shape[:2]
        
        print(f"Processing image: {original_shape}")
        
        # Step 1: Detect all seeds
        detections = self.detect_seeds(image)
        print(f"Detected {len(detections)} seeds")
        
        # Step 2: Classify each detected seed
        for det in detections:
            seed_crop = self.crop_seed(image, det['bbox'])
            
            if seed_crop is not None:
                class_name, class_id, confidence = self.classify_seed(seed_crop)
                det['class_name'] = class_name
                det['class_id'] = class_id
                det['classification_confidence'] = confidence
                det['confidence'] = confidence  # For visualization
            else:
                # Default to REAL if crop failed
                det['class_name'] = 'REAL'
                det['class_id'] = 0
                det['classification_confidence'] = 0.5
                det['confidence'] = 0.5
        
        # Step 3: Calculate statistics
        total = len(detections)
        real_count = sum(1 for d in detections if d.get('class_name') == 'REAL')
        fake_count = sum(1 for d in detections if d.get('class_name') == 'FAKE')
        low_quality_count = sum(1 for d in detections if d.get('class_name') == 'LOW_QUALITY')
        
        real_pct = (real_count / total * 100) if total > 0 else 0
        fake_pct = (fake_count / total * 100) if total > 0 else 0
        low_quality_pct = (low_quality_count / total * 100) if total > 0 else 0
        
        # Step 4: Calculate verdict
        verdict_key, verdict_message = self.calculate_verdict(detections)
        
        # Step 5: Create visualization
        annotated_b64 = None
        if return_annotated or save_path:
            stats = {'REAL': real_count, 'FAKE': fake_count, 'LOW_QUALITY': low_quality_count}
            annotated = create_result_visualization(image, detections, verdict_key, stats)
            annotated = draw_legend(annotated, position="bottom-right")
            
            if save_path:
                save_annotated_image(annotated, save_path)
            
            if return_annotated:
                annotated_b64 = numpy_to_base64(annotated)
        
        # Step 6: Create result
        result = InferenceResult(
            total_seeds=total,
            real_count=real_count,
            fake_count=fake_count,
            low_quality_count=low_quality_count,
            real_percentage=round(real_pct, 2),
            fake_percentage=round(fake_pct, 2),
            low_quality_percentage=round(low_quality_pct, 2),
            verdict=verdict_key,
            verdict_message=verdict_message,
            detections=[SeedDetection(
                id=d['id'],
                bbox=d['bbox'],
                class_name=d['class_name'],
                class_id=d['class_id'],
                detection_confidence=d['detection_confidence'],
                classification_confidence=d['classification_confidence']
            ).to_dict() for d in detections],
            annotated_image_base64=annotated_b64
        )
        
        return result
    
    def get_annotated_image(self, 
                            image_source: Union[str, np.ndarray, Image.Image],
                            result: Optional[InferenceResult] = None) -> np.ndarray:
        """
        Get annotated image as numpy array.
        
        Args:
            image_source: Input image
            result: Optional pre-computed result
        
        Returns:
            Annotated image as numpy array
        """
        if result is None:
            result = self.process_image(image_source, return_annotated=False)
        
        image = self.load_image(image_source)
        
        stats = {
            'REAL': result.real_count,
            'FAKE': result.fake_count,
            'LOW_QUALITY': result.low_quality_count
        }
        
        # Recreate detections with proper format
        detections = []
        for det in result.detections:
            detections.append({
                'bbox': det['bbox'],
                'class_name': det['class_name'],
                'confidence': det['classification_confidence']
            })
        
        annotated = create_result_visualization(image, detections, result.verdict, stats)
        annotated = draw_legend(annotated, position="bottom-right")
        
        return annotated


def create_demo_prediction():
    """Create a demo prediction for testing without trained models."""
    import random
    
    # Generate fake detections
    detections = []
    for i in range(random.randint(5, 15)):
        cls = random.choices(
            ['REAL', 'FAKE', 'LOW_QUALITY'],
            weights=[0.6, 0.2, 0.2]
        )[0]
        
        detections.append({
            'id': i,
            'bbox': (
                random.randint(50, 500),
                random.randint(50, 500),
                random.randint(40, 100),
                random.randint(40, 100)
            ),
            'class_name': cls,
            'class_id': CLASS_NAMES.index(cls),
            'detection_confidence': random.uniform(0.7, 0.99),
            'classification_confidence': random.uniform(0.7, 0.99)
        })
    
    return detections


def main():
    """Main function for testing the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SeedShield Inference Pipeline')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save annotated image')
    parser.add_argument('--json-output', type=str, default=None,
                       help='Path to save JSON results')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode (without trained models)')
    
    args = parser.parse_args()
    
    if args.demo:
        # Demo mode - create visualization without actual models
        print("Running in demo mode...")
        image = cv2.imread(args.image)
        if image is None:
            # Create dummy image
            image = np.ones((640, 640, 3), dtype=np.uint8) * 200
        
        detections = create_demo_prediction()
        
        # Calculate stats
        real_count = sum(1 for d in detections if d['class_name'] == 'REAL')
        fake_count = sum(1 for d in detections if d['class_name'] == 'FAKE')
        low_quality_count = sum(1 for d in detections if d['class_name'] == 'LOW_QUALITY')
        total = len(detections)
        
        stats = {'REAL': real_count, 'FAKE': fake_count, 'LOW_QUALITY': low_quality_count}
        
        # Determine verdict
        if fake_count / total > 0.3:
            verdict = "suspected_fake"
        elif low_quality_count / total > 0.4:
            verdict = "low_quality"
        else:
            verdict = "genuine"
        
        # Create visualization
        annotated = create_result_visualization(image, detections, verdict, stats)
        annotated = draw_legend(annotated)
        
        if args.output:
            save_annotated_image(annotated, args.output)
        
        print(f"Demo results: {total} seeds detected")
        print(f"  REAL: {real_count} ({real_count/total*100:.1f}%)")
        print(f"  FAKE: {fake_count} ({fake_count/total*100:.1f}%)")
        print(f"  LOW_QUALITY: {low_quality_count} ({low_quality_count/total*100:.1f}%)")
        print(f"  Verdict: {VERDICT_MESSAGES[verdict]}")
        
    else:
        # Full pipeline
        pipeline = SeedShieldPipeline()
        result = pipeline.process_image(
            args.image,
            return_annotated=True,
            save_path=args.output
        )
        
        # Print results
        print("\n" + "=" * 50)
        print("SEEDSHIELD ANALYSIS RESULTS")
        print("=" * 50)
        print(f"\nTotal Seeds: {result.total_seeds}")
        print(f"  REAL: {result.real_count} ({result.real_percentage:.1f}%)")
        print(f"  FAKE: {result.fake_count} ({result.fake_percentage:.1f}%)")
        print(f"  LOW QUALITY: {result.low_quality_count} ({result.low_quality_percentage:.1f}%)")
        print(f"\nVerdict: {result.verdict_message}")
        
        # Save JSON if requested
        if args.json_output:
            with open(args.json_output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nJSON results saved to: {args.json_output}")


if __name__ == "__main__":
    main()
