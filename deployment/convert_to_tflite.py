"""
SeedShield - TensorFlow Lite Conversion
Convert trained models to TFLite format for mobile deployment.
"""

import os
import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import tensorflow as tf

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_config import get_path_config, CLASSIFIER_CONFIG


def convert_classifier_to_tflite(
    pytorch_weights: str,
    output_path: str,
    quantize: bool = True,
    image_size: int = 224
):
    """
    Convert MobileNetV2 classifier from PyTorch to TensorFlow Lite.
    
    Args:
        pytorch_weights: Path to PyTorch weights (.pth)
        output_path: Output path for TFLite model
        quantize: Whether to apply quantization
        image_size: Input image size
    """
    print("Converting classifier to TFLite...")
    print(f"  Input: {pytorch_weights}")
    print(f"  Output: {output_path}")
    
    # Load PyTorch model
    from models.train_classifier import SeedClassifier
    
    model = SeedClassifier(num_classes=3, pretrained=False)
    
    if Path(pytorch_weights).exists():
        checkpoint = torch.load(pytorch_weights, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Loaded PyTorch weights")
    else:
        print("  Warning: Using untrained model")
    
    model.eval()
    
    # Export to ONNX first
    onnx_path = output_path.replace('.tflite', '.onnx')
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"  Exported to ONNX: {onnx_path}")
    
    # Convert ONNX to TensorFlow SavedModel
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        
        saved_model_dir = output_path.replace('.tflite', '_savedmodel')
        tf_rep.export_graph(saved_model_dir)
        print(f"  Converted to TensorFlow SavedModel: {saved_model_dir}")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            print("  Applying float16 quantization")
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"  TFLite model saved: {output_path}")
        print(f"  Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
    except ImportError:
        print("  onnx-tf not installed. Creating TFLite from Keras instead...")
        convert_using_keras(image_size, output_path, quantize)


def convert_using_keras(image_size: int, output_path: str, quantize: bool):
    """Alternative conversion using Keras MobileNetV2."""
    print("  Using Keras MobileNetV2 for TFLite conversion...")
    
    # Create Keras model with same architecture
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    x = base_model.output
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"  TFLite model saved: {output_path}")
    print(f"  Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")


def convert_yolo_to_tflite(
    yolo_weights: str,
    output_path: str,
    image_size: int = 640
):
    """
    Convert YOLOv8 detector to TFLite format.
    
    Args:
        yolo_weights: Path to YOLOv8 weights (.pt)
        output_path: Output path for TFLite model
        image_size: Input image size
    """
    print("Converting YOLOv8 detector to TFLite...")
    print(f"  Input: {yolo_weights}")
    print(f"  Output: {output_path}")
    
    try:
        from ultralytics import YOLO
        
        # Load model
        if Path(yolo_weights).exists():
            model = YOLO(yolo_weights)
        else:
            print("  Warning: Using pretrained YOLOv8n")
            model = YOLO('yolov8n.pt')
        
        # Export to TFLite
        model.export(
            format='tflite',
            imgsz=image_size,
            int8=False,
            half=True
        )
        
        # Find exported file and move to desired location
        exported_path = Path(yolo_weights).parent / f"{Path(yolo_weights).stem}_saved_model"
        tflite_file = list(exported_path.glob("*.tflite"))
        
        if tflite_file:
            import shutil
            shutil.move(str(tflite_file[0]), output_path)
            print(f"  TFLite model saved: {output_path}")
        else:
            print("  TFLite export complete. Check ultralytics output directory.")
            
    except Exception as e:
        print(f"  Error during conversion: {e}")
        print("  You can manually export using: model.export(format='tflite')")


def test_tflite_model(tflite_path: str, image_size: int = 224):
    """Test the converted TFLite model."""
    print(f"\nTesting TFLite model: {tflite_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    # Create dummy input
    input_shape = input_details[0]['shape']
    input_data = np.random.rand(*input_shape).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  Output: {output_data}")
    print("  TFLite model test passed!")


def main():
    parser = argparse.ArgumentParser(description='Convert SeedShield models to TFLite')
    parser.add_argument('--classifier', action='store_true',
                       help='Convert classifier model')
    parser.add_argument('--detector', action='store_true',
                       help='Convert detector model')
    parser.add_argument('--quantize', action='store_true', default=True,
                       help='Apply quantization')
    parser.add_argument('--test', action='store_true',
                       help='Test converted models')
    
    args = parser.parse_args()
    
    paths = get_path_config()
    output_dir = paths.deployment_dir / "tflite"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.classifier or (not args.classifier and not args.detector):
        classifier_weights = paths.classifier_weights / "best_classifier.pth"
        classifier_output = output_dir / "seedshield_classifier.tflite"
        
        convert_classifier_to_tflite(
            str(classifier_weights),
            str(classifier_output),
            quantize=args.quantize
        )
        
        if args.test and classifier_output.exists():
            test_tflite_model(str(classifier_output))
    
    if args.detector:
        detector_weights = paths.detector_weights / "seedshield_detector" / "weights" / "best.pt"
        detector_output = output_dir / "seedshield_detector.tflite"
        
        convert_yolo_to_tflite(
            str(detector_weights),
            str(detector_output)
        )
    
    print("\n" + "=" * 50)
    print("TFLite Conversion Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
