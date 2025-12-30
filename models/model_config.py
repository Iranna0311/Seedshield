"""
SeedShield - Model Configuration
Central configuration for all models in the SeedShield system.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DetectorConfig:
    """Configuration for YOLOv8 seed detector."""
    
    # Model architecture
    model_name: str = "yolov8n.pt"  # Nano version for mobile
    model_variant: str = "yolov8n"
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    
    # Optimizer settings
    optimizer: str = "AdamW"
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Augmentation
    augment: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    
    # Early stopping
    patience: int = 50
    
    # Confidence thresholds
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Device
    device: str = "auto"  # "cpu", "cuda", or "auto"


@dataclass
class ClassifierConfig:
    """Configuration for MobileNetV2 seed classifier."""
    
    # Model architecture
    model_name: str = "MobileNetV2"
    pretrained: bool = True
    
    # Input
    image_size: int = 224
    num_channels: int = 3
    
    # Classes
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["REAL", "FAKE", "LOW_QUALITY"])
    
    # Training parameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Optimizer
    optimizer: str = "Adam"
    weight_decay: float = 0.0001
    
    # Learning rate scheduler
    lr_scheduler: str = "ReduceLROnPlateau"
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-7
    
    # Early stopping
    early_stop_patience: int = 15
    
    # Regularization
    dropout_rate: float = 0.3
    
    # Data augmentation
    augment: bool = True
    
    # Device
    device: str = "auto"


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    
    # Detection settings
    detector_conf_threshold: float = 0.3
    detector_iou_threshold: float = 0.5
    max_detections: int = 100
    
    # Classification settings
    classifier_conf_threshold: float = 0.5
    
    # Visualization
    box_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
    
    # Colors (BGR format)
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "REAL": (0, 255, 0),       # Green
        "FAKE": (0, 0, 255),       # Red
        "LOW_QUALITY": (0, 255, 255)  # Yellow
    })
    
    # Verdict thresholds
    fake_threshold: float = 0.30   # If FAKE > 30%, suspicious
    low_quality_threshold: float = 0.40  # If LOW_QUALITY > 40%, mixed
    
    # Mobile optimization
    use_tflite: bool = False
    use_onnx: bool = False


@dataclass
class PathConfig:
    """Path configuration for all project files."""
    
    # Base directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def dataset_dir(self) -> Path:
        return self.base_dir / "dataset"
    
    @property
    def processed_dir(self) -> Path:
        return self.dataset_dir / "processed"
    
    @property
    def detection_dataset(self) -> Path:
        return self.processed_dir / "detection"
    
    @property
    def classification_dataset(self) -> Path:
        return self.processed_dir / "classification"
    
    @property
    def data_yaml(self) -> Path:
        return self.dataset_dir / "data.yaml"
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"
    
    @property
    def weights_dir(self) -> Path:
        return self.models_dir / "weights"
    
    @property
    def detector_weights(self) -> Path:
        return self.weights_dir / "detector"
    
    @property
    def classifier_weights(self) -> Path:
        return self.weights_dir / "classifier"
    
    @property
    def inference_dir(self) -> Path:
        return self.base_dir / "inference"
    
    @property
    def api_dir(self) -> Path:
        return self.base_dir / "api"
    
    @property
    def frontend_dir(self) -> Path:
        return self.base_dir / "frontend"
    
    @property
    def deployment_dir(self) -> Path:
        return self.base_dir / "deployment"
    
    def ensure_directories(self):
        """Create all necessary directories."""
        directories = [
            self.weights_dir,
            self.detector_weights,
            self.classifier_weights,
            self.inference_dir,
            self.api_dir,
            self.deployment_dir
        ]
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)


# Global configuration instances
DETECTOR_CONFIG = DetectorConfig()
CLASSIFIER_CONFIG = ClassifierConfig()
INFERENCE_CONFIG = InferenceConfig()
PATH_CONFIG = PathConfig()


# Class mappings
CLASS_NAMES = ["REAL", "FAKE", "LOW_QUALITY"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Verdict messages
VERDICT_MESSAGES = {
    "genuine": "✅ Mostly Genuine",
    "suspected_fake": "❌ Suspected Fake",
    "low_quality": "⚠️ Low Quality / Mixed"
}


def get_detector_config() -> DetectorConfig:
    """Get detector configuration."""
    return DETECTOR_CONFIG


def get_classifier_config() -> ClassifierConfig:
    """Get classifier configuration."""
    return CLASSIFIER_CONFIG


def get_inference_config() -> InferenceConfig:
    """Get inference configuration."""
    return INFERENCE_CONFIG


def get_path_config() -> PathConfig:
    """Get path configuration and ensure directories exist."""
    PATH_CONFIG.ensure_directories()
    return PATH_CONFIG


if __name__ == "__main__":
    # Print configuration summary
    print("SeedShield Model Configuration")
    print("=" * 50)
    
    print("\nDetector Configuration:")
    print(f"  Model: {DETECTOR_CONFIG.model_name}")
    print(f"  Image Size: {DETECTOR_CONFIG.image_size}")
    print(f"  Epochs: {DETECTOR_CONFIG.epochs}")
    print(f"  Batch Size: {DETECTOR_CONFIG.batch_size}")
    
    print("\nClassifier Configuration:")
    print(f"  Model: {CLASSIFIER_CONFIG.model_name}")
    print(f"  Image Size: {CLASSIFIER_CONFIG.image_size}")
    print(f"  Classes: {CLASSIFIER_CONFIG.class_names}")
    print(f"  Epochs: {CLASSIFIER_CONFIG.epochs}")
    
    print("\nInference Configuration:")
    print(f"  Detector Confidence: {INFERENCE_CONFIG.detector_conf_threshold}")
    print(f"  Fake Threshold: {INFERENCE_CONFIG.fake_threshold}")
    print(f"  Colors: {INFERENCE_CONFIG.colors}")
    
    print("\nPath Configuration:")
    paths = get_path_config()
    print(f"  Base Dir: {paths.base_dir}")
    print(f"  Dataset: {paths.dataset_dir}")
    print(f"  Weights: {paths.weights_dir}")
