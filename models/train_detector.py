"""
SeedShield - YOLOv8 Seed Detector Training
Complete training script for seed detection model.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
from ultralytics import YOLO

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from models.model_config import (
        get_detector_config, get_path_config,
        DETECTOR_CONFIG, PATH_CONFIG
    )
except ImportError:
    from model_config import (
        get_detector_config, get_path_config,
        DETECTOR_CONFIG, PATH_CONFIG
    )


class SeedDetectorTrainer:
    """
    Training pipeline for YOLOv8 seed detector.
    Detects individual seeds in composite images.
    """
    
    def __init__(self, config=None, paths=None):
        """
        Initialize the trainer.
        
        Args:
            config: DetectorConfig instance
            paths: PathConfig instance
        """
        self.config = config or get_detector_config()
        self.paths = paths or get_path_config()
        
        # Set device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        self.results = None
    
    def load_model(self, pretrained: bool = True):
        """
        Load YOLOv8 model.
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        print(f"\nLoading model: {self.config.model_name}")
        
        if pretrained:
            self.model = YOLO(self.config.model_name)
        else:
            # Load model architecture only
            self.model = YOLO(f"{self.config.model_variant}.yaml")
        
        print(f"Model loaded: {self.config.model_variant}")
        return self.model
    
    def verify_dataset(self) -> bool:
        """Verify that the dataset exists and is properly formatted."""
        data_yaml = self.paths.data_yaml
        
        if not data_yaml.exists():
            print(f"ERROR: data.yaml not found at {data_yaml}")
            print("Please run the dataset generator first:")
            print("  python dataset/generate_synthetic_dataset.py")
            return False
        
        # Check dataset directories
        detection_dir = self.paths.detection_dataset
        for split in ['train', 'val']:
            img_dir = detection_dir / "images" / split
            label_dir = detection_dir / "labels" / split
            
            if not img_dir.exists():
                print(f"ERROR: Image directory not found: {img_dir}")
                return False
            
            if not label_dir.exists():
                print(f"ERROR: Label directory not found: {label_dir}")
                return False
            
            num_images = len(list(img_dir.glob("*.jpg")))
            num_labels = len(list(label_dir.glob("*.txt")))
            
            print(f"  {split}: {num_images} images, {num_labels} labels")
            
            if num_images == 0:
                print(f"ERROR: No images found in {img_dir}")
                return False
        
        print("Dataset verified successfully!")
        return True
    
    def train(self, 
              epochs: int = None,
              batch_size: int = None,
              resume: bool = False,
              project_name: str = "seedshield_detector"):
        """
        Train the seed detector model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            resume: Whether to resume from last checkpoint
            project_name: Name for the training project
        
        Returns:
            Training results
        """
        # Verify dataset
        if not self.verify_dataset():
            return None
        
        # Load model if not loaded
        if self.model is None:
            self.load_model()
        
        # Use config values if not specified
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        
        print(f"\nStarting training:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Image Size: {self.config.image_size}")
        print(f"  Device: {self.device}")
        
        # Training arguments
        train_args = {
            'data': str(self.paths.data_yaml),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': self.config.image_size,
            'device': self.device,
            'project': str(self.paths.detector_weights),
            'name': project_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': self.config.optimizer,
            'lr0': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'momentum': self.config.momentum,
            'patience': self.config.patience,
            'verbose': True,
            'save': True,
            'save_period': 10,
            'val': True,
            
            # Augmentation
            'augment': self.config.augment,
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'scale': self.config.scale,
            'flipud': self.config.flipud,
            'fliplr': self.config.fliplr,
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
        }
        
        if resume:
            train_args['resume'] = True
        
        # Train
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60 + "\n")
        
        self.results = self.model.train(**train_args)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        
        return self.results
    
    def evaluate(self, split: str = "test"):
        """
        Evaluate the trained model.
        
        Args:
            split: Dataset split to evaluate on ('val' or 'test')
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            print("ERROR: No model loaded. Train or load a model first.")
            return None
        
        print(f"\nEvaluating on {split} set...")
        
        metrics = self.model.val(
            data=str(self.paths.data_yaml),
            split=split,
            device=self.device,
            verbose=True
        )
        
        print("\nEvaluation Results:")
        print(f"  mAP@0.5: {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_model(self, formats: list = None):
        """
        Export the trained model to different formats.
        
        Args:
            formats: List of export formats (e.g., ['onnx', 'tflite'])
        """
        if self.model is None:
            print("ERROR: No model loaded.")
            return
        
        formats = formats or ['onnx', 'torchscript']
        
        print(f"\nExporting model to formats: {formats}")
        
        for fmt in formats:
            try:
                print(f"  Exporting to {fmt}...")
                exported = self.model.export(format=fmt)
                print(f"  Exported: {exported}")
            except Exception as e:
                print(f"  Error exporting to {fmt}: {e}")
    
    def save_best_model(self, destination: Path = None):
        """
        Copy the best model weights to a specified location.
        
        Args:
            destination: Destination path for the model
        """
        destination = destination or (self.paths.detector_weights / "best.pt")
        
        # Find the best weights from training
        best_weights = self.paths.detector_weights / "seedshield_detector" / "weights" / "best.pt"
        
        if best_weights.exists():
            shutil.copy(best_weights, destination)
            print(f"Best model saved to: {destination}")
        else:
            print(f"Best weights not found at: {best_weights}")
    
    def load_trained_model(self, weights_path: Path = None):
        """
        Load a previously trained model.
        
        Args:
            weights_path: Path to the model weights
        """
        weights_path = weights_path or (self.paths.detector_weights / "seedshield_detector" / "weights" / "best.pt")
        
        if not weights_path.exists():
            # Try alternative location
            weights_path = self.paths.detector_weights / "best.pt"
        
        if weights_path.exists():
            print(f"Loading trained model from: {weights_path}")
            self.model = YOLO(str(weights_path))
            return True
        else:
            print(f"Trained model not found at: {weights_path}")
            return False
    
    def predict(self, image_path: str, conf: float = None, save: bool = True):
        """
        Run prediction on an image.
        
        Args:
            image_path: Path to the input image
            conf: Confidence threshold
            save: Whether to save the annotated image
        
        Returns:
            Prediction results
        """
        if self.model is None:
            if not self.load_trained_model():
                print("ERROR: No model available for prediction.")
                return None
        
        conf = conf or self.config.conf_threshold
        
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=self.config.iou_threshold,
            device=self.device,
            save=save,
            verbose=False
        )
        
        return results


def main():
    """Main training entrypoint."""
    parser = argparse.ArgumentParser(description='Train SeedShield Detector')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--export', nargs='+', default=None,
                       help='Export formats (e.g., onnx tflite)')
    parser.add_argument('--predict', type=str, default=None,
                       help='Run prediction on an image')
    
    args = parser.parse_args()
    
    trainer = SeedDetectorTrainer()
    
    if args.predict:
        # Run prediction only
        results = trainer.predict(args.predict)
        if results:
            print(f"Detected {len(results[0].boxes)} seeds")
    else:
        # Train the model
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=args.resume
        )
        
        # Save best model
        trainer.save_best_model()
        
        # Evaluate if requested
        if args.evaluate:
            trainer.evaluate()
        
        # Export if requested
        if args.export:
            trainer.export_model(args.export)


if __name__ == "__main__":
    main()
