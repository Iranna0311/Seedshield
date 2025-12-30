"""
SeedShield - ONNX Conversion
Convert trained models to ONNX format for cross-platform deployment.
"""

import os
import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import onnx
import onnxruntime as ort

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_config import get_path_config


def convert_classifier_to_onnx(
    pytorch_weights: str,
    output_path: str,
    image_size: int = 224,
    opset_version: int = 11,
    simplify: bool = True
):
    """
    Convert MobileNetV2 classifier from PyTorch to ONNX.
    
    Args:
        pytorch_weights: Path to PyTorch weights (.pth)
        output_path: Output path for ONNX model
        image_size: Input image size
        opset_version: ONNX opset version
        simplify: Whether to simplify the model
    """
    print("Converting classifier to ONNX...")
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
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"  ONNX model saved: {output_path}")
    
    # Verify model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model verified successfully")
    
    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            simplified_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(simplified_model, output_path)
                print("  ONNX model simplified")
        except ImportError:
            print("  onnxsim not installed, skipping simplification")
    
    # Get model size
    model_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  Model size: {model_size:.2f} MB")


def convert_detector_to_onnx(
    yolo_weights: str,
    output_path: str,
    image_size: int = 640
):
    """
    Convert YOLOv8 detector to ONNX format.
    
    Args:
        yolo_weights: Path to YOLOv8 weights (.pt)
        output_path: Output path for ONNX model
        image_size: Input image size
    """
    print("Converting YOLOv8 detector to ONNX...")
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
        
        # Export to ONNX
        exported = model.export(
            format='onnx',
            imgsz=image_size,
            simplify=True,
            dynamic=True,
            opset=11
        )
        
        # Move to desired location
        if exported and Path(exported).exists():
            import shutil
            shutil.move(str(exported), output_path)
            print(f"  ONNX model saved: {output_path}")
        
    except Exception as e:
        print(f"  Error during conversion: {e}")


def test_onnx_model(onnx_path: str, image_size: int = 224, is_detector: bool = False):
    """Test the converted ONNX model."""
    print(f"\nTesting ONNX model: {onnx_path}")
    
    # Create inference session
    session = ort.InferenceSession(onnx_path)
    
    # Get input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"  Input name: {input_info.name}")
    print(f"  Input shape: {input_info.shape}")
    print(f"  Output name: {output_info.name}")
    print(f"  Output shape: {output_info.shape}")
    
    # Create dummy input
    if is_detector:
        input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    else:
        input_data = np.random.rand(1, 3, image_size, image_size).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_info.name: input_data})
    
    print(f"  Output shape: {outputs[0].shape}")
    
    if not is_detector:
        # For classifier, show class probabilities
        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
        class_names = ['REAL', 'FAKE', 'LOW_QUALITY']
        print(f"  Predictions:")
        for i, (name, prob) in enumerate(zip(class_names, probs[0])):
            print(f"    {name}: {prob:.4f}")
    
    print("  ONNX model test passed!")
    
    return session


def benchmark_onnx_model(onnx_path: str, num_iterations: int = 100):
    """Benchmark ONNX model inference speed."""
    import time
    
    print(f"\nBenchmarking ONNX model: {onnx_path}")
    
    session = ort.InferenceSession(onnx_path)
    input_info = session.get_inputs()[0]
    
    # Determine input shape
    shape = input_info.shape
    if None in shape or -1 in shape:
        shape = [1, 3, 224, 224]  # Default for classifier
    
    input_data = np.random.rand(*shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_info.name: input_data})
    
    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        session.run(None, {input_info.name: input_data})
    end = time.time()
    
    avg_time = (end - start) / num_iterations * 1000
    fps = 1000 / avg_time
    
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  FPS: {fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Convert SeedShield models to ONNX')
    parser.add_argument('--classifier', action='store_true',
                       help='Convert classifier model')
    parser.add_argument('--detector', action='store_true',
                       help='Convert detector model')
    parser.add_argument('--test', action='store_true',
                       help='Test converted models')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark model speed')
    
    args = parser.parse_args()
    
    paths = get_path_config()
    output_dir = paths.deployment_dir / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.classifier or (not args.classifier and not args.detector):
        classifier_weights = paths.classifier_weights / "best_classifier.pth"
        classifier_output = output_dir / "seedshield_classifier.onnx"
        
        convert_classifier_to_onnx(
            str(classifier_weights),
            str(classifier_output)
        )
        
        if args.test and classifier_output.exists():
            test_onnx_model(str(classifier_output))
        
        if args.benchmark and classifier_output.exists():
            benchmark_onnx_model(str(classifier_output))
    
    if args.detector:
        detector_weights = paths.detector_weights / "seedshield_detector" / "weights" / "best.pt"
        detector_output = output_dir / "seedshield_detector.onnx"
        
        convert_detector_to_onnx(
            str(detector_weights),
            str(detector_output)
        )
        
        if args.test and detector_output.exists():
            test_onnx_model(str(detector_output), is_detector=True)
    
    print("\n" + "=" * 50)
    print("ONNX Conversion Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
