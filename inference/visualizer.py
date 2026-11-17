"""
SeedShield - Visualization Utilities
Functions for drawing bounding boxes, labels, and annotations on seed images.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# Color definitions (BGR for OpenCV)
COLORS = {
    "REAL": (0, 255, 0),       # Green
    "FAKE": (0, 0, 255),       # Red
    "LOW_QUALITY": (0, 255, 255)  # Yellow
}

# Color definitions (RGB for PIL)
COLORS_RGB = {
    "REAL": (0, 255, 0),
    "FAKE": (255, 0, 0),
    "LOW_QUALITY": (255, 255, 0)
}

# Emoji/symbol prefixes
SYMBOLS = {
    "REAL": "✅",
    "FAKE": "❌",
    "LOW_QUALITY": "⚠️"
}


def draw_seed_detection(
    image: np.ndarray,
    detections: List[Dict],
    box_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes and labels for each detected seed.
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries with keys:
            - 'bbox': (x, y, w, h) bounding box
            - 'class_name': 'REAL', 'FAKE', or 'LOW_QUALITY'
            - 'confidence': Detection confidence
            - 'classification_confidence': Classification confidence (optional)
        box_thickness: Thickness of bounding box lines
        font_scale: Font scale for labels
        font_thickness: Thickness of font
        show_confidence: Whether to show confidence percentage
    
    Returns:
        Annotated image
    """
    result = image.copy()
    overlay = result.copy()
    
    # First pass: Draw translucent color fills for all seeds to make classes immediately visible
    for det in detections:
        bbox = det['bbox']
        class_name = det.get('class_name', 'REAL')
        x, y, w, h = [int(v) for v in bbox]
        if class_name == 'FAKE':
            # Prominent red translucent danger highlight on fake seeds
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
        elif class_name == 'REAL':
            # Subtle emerald green highlight on authentic seeds
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 220, 80), -1)
        elif class_name == 'LOW_QUALITY':
            # Amber highlight for low quality
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 255), -1)
        else:
            # Grey highlight for uncertain
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (180, 180, 180), -1)
            
    # Blend overlay with image (28% fill opacity for fake, soft blend overall)
    cv2.addWeighted(overlay, 0.28, result, 0.72, 0, result)

    # Second pass: Draw crisp borders, tactical corner brackets, and labels
    for det in detections:
        bbox = det['bbox']
        class_name = det.get('class_name', 'REAL')
        confidence = det.get('confidence', 0.0)
        class_conf = det.get('classification_confidence', confidence)
        
        # Get color
        color = COLORS.get(class_name, (128, 128, 128))
        x, y, w, h = [int(v) for v in bbox]
        
        if class_name == 'FAKE':
            # FAKE SEED: Double border & tactical corner brackets for alert effect
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), max(3, box_thickness + 1))
            corner_len = max(6, min(w, h) // 4)
            # Corner accents
            cv2.line(result, (x - 2, y - 2), (x + corner_len, y - 2), (0, 0, 255), max(3, box_thickness + 2))
            cv2.line(result, (x - 2, y - 2), (x - 2, y + corner_len), (0, 0, 255), max(3, box_thickness + 2))
            cv2.line(result, (x + w + 2, y - 2), (x + w - corner_len, y - 2), (0, 0, 255), max(3, box_thickness + 2))
            cv2.line(result, (x + w + 2, y - 2), (x + w + 2, y + corner_len), (0, 0, 255), max(3, box_thickness + 2))
            cv2.line(result, (x - 2, y + h + 2), (x + corner_len, y + h + 2), (0, 0, 255), max(3, box_thickness + 2))
            cv2.line(result, (x - 2, y + h + 2), (x - 2, y + h - corner_len), (0, 0, 255), max(3, box_thickness + 2))
            cv2.line(result, (x + w + 2, y + h + 2), (x + w - corner_len, y + h + 2), (0, 0, 255), max(3, box_thickness + 2))
            cv2.line(result, (x + w + 2, y + h + 2), (x + w + 2, y + h - corner_len), (0, 0, 255), max(3, box_thickness + 2))
            
            prefix = "[!] FAKE"
        elif class_name == 'REAL':
            # REAL SEED: Crisp clean border
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 220, 80), box_thickness)
            prefix = "[OK] REAL"
        elif class_name == 'LOW_QUALITY':
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 220, 255), box_thickness)
            prefix = "[*] LOW"
        else:
            cv2.rectangle(result, (x, y), (x + w, y + h), (160, 160, 160), box_thickness)
            prefix = "[?] UNCERTAIN"
        
        # Prepare label text
        if show_confidence:
            label = f"{prefix}: {class_conf * 100:.1f}%"
        else:
            label = prefix
        
        # Calculate label position
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        label_x = max(0, x)
        label_y = y - 6 if y > 22 else y + h + label_size[1] + 8
        
        # Draw label background
        cv2.rectangle(
            result,
            (label_x, label_y - label_size[1] - 4),
            (label_x + label_size[0] + 6, label_y + 4),
            color,
            -1
        )
        
        # Draw label text
        text_color = (0, 0, 0) if class_name in ("LOW_QUALITY", "UNCERTAIN") else (255, 255, 255)
        cv2.putText(
            result,
            label,
            (label_x + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness
        )
    
    return result


def draw_verdict_overlay(
    image: np.ndarray,
    verdict: str,
    stats: Dict[str, int],
    position: str = "top"
) -> np.ndarray:
    """
    Draw verdict and statistics overlay on the image.
    
    Args:
        image: Input image
        verdict: Verdict string ('genuine', 'suspected_fake', 'low_quality')
        stats: Dictionary with counts {'REAL': n, 'FAKE': n, 'LOW_QUALITY': n}
        position: Position of overlay ('top' or 'bottom')
    
    Returns:
        Image with overlay
    """
    result = image.copy()
    h, w = result.shape[:2]
    
    # Use shorter text for smaller images
    if w < 400:
        verdict_messages = {
            "genuine": ("OK - REAL", (0, 255, 0)),
            "suspected_fake": ("FAKE!", (0, 0, 255)),
            "low_quality": ("LOW QUAL", (0, 255, 255))
        }
    else:
        verdict_messages = {
            "genuine": ("GENUINE", (0, 255, 0)),
            "suspected_fake": ("FAKE DETECTED", (0, 0, 255)),
            "low_quality": ("LOW QUALITY", (0, 255, 255))
        }
    
    verdict_text, verdict_color = verdict_messages.get(verdict, ("?", (128, 128, 128)))
    
    # Calculate overlay height based on image size
    overlay_height = max(50, min(80, h // 5))
    if position == "top":
        overlay_y = 0
    else:
        overlay_y = h - overlay_height
    
    # Dark solid background
    overlay = result.copy()
    cv2.rectangle(overlay, (0, overlay_y), (w, overlay_y + overlay_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.9, result, 0.1, 0, result)
    
    # Calculate font scale to fit text within image width
    # Test different scales and pick the one that fits
    test_scale = 1.0
    while test_scale > 0.3:
        text_size = cv2.getTextSize(verdict_text, cv2.FONT_HERSHEY_SIMPLEX, test_scale, 2)[0]
        if text_size[0] < w - 20:  # Leave 10px margin on each side
            break
        test_scale -= 0.1
    
    font_scale = max(0.4, test_scale)
    font_thickness = max(1, int(font_scale * 2))
    
    # Draw verdict text
    text_y = overlay_y + int(overlay_height * 0.5)
    cv2.putText(
        result,
        verdict_text,
        (10, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        verdict_color,
        font_thickness
    )
    
    # Draw statistics on second line (smaller)
    total = sum(stats.values())
    if w < 300:
        stats_text = f"R:{stats.get('REAL', 0)} F:{stats.get('FAKE', 0)} L:{stats.get('LOW_QUALITY', 0)}"
    else:
        stats_text = f"Total:{total} R:{stats.get('REAL', 0)} F:{stats.get('FAKE', 0)} L:{stats.get('LOW_QUALITY', 0)}"
    
    stats_scale = max(0.3, font_scale * 0.5)
    stats_y = overlay_y + int(overlay_height * 0.85)
    cv2.putText(
        result,
        stats_text,
        (10, stats_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        stats_scale,
        (255, 255, 255),
        max(1, font_thickness - 1)
    )
    
    return result



def create_result_visualization(
    image: np.ndarray,
    detections: List[Dict],
    verdict: str,
    stats: Dict[str, int],
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Create complete visualization with detections and verdict.
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries
        verdict: Verdict string
        stats: Statistics dictionary
        output_size: Optional output size (width, height)
    
    Returns:
        Complete annotated image
    """
    # Draw detections
    result = draw_seed_detection(image, detections)
    
    # Draw verdict overlay
    result = draw_verdict_overlay(result, verdict, stats, position="top")
    
    # Resize if needed
    if output_size is not None:
        result = cv2.resize(result, output_size)
    
    return result


def create_side_by_side(
    original: np.ndarray,
    annotated: np.ndarray,
    spacing: int = 20
) -> np.ndarray:
    """
    Create side-by-side comparison of original and annotated images.
    
    Args:
        original: Original image
        annotated: Annotated image
        spacing: Spacing between images
    
    Returns:
        Combined image
    """
    # Ensure same height
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]
    
    max_h = max(h1, h2)
    
    if h1 != max_h:
        scale = max_h / h1
        original = cv2.resize(original, (int(w1 * scale), max_h))
        w1 = int(w1 * scale)
    
    if h2 != max_h:
        scale = max_h / h2
        annotated = cv2.resize(annotated, (int(w2 * scale), max_h))
        w2 = int(w2 * scale)
    
    # Create combined image
    combined = np.ones((max_h, w1 + spacing + w2, 3), dtype=np.uint8) * 255
    combined[:, :w1] = original
    combined[:, w1 + spacing:] = annotated
    
    return combined


def create_grid_visualization(
    seeds: List[np.ndarray],
    labels: List[str],
    confidences: List[float],
    grid_size: Tuple[int, int] = (4, 4),
    cell_size: int = 100
) -> np.ndarray:
    """
    Create a grid visualization of individual seeds.
    
    Args:
        seeds: List of seed images
        labels: List of class labels
        confidences: List of confidence scores
        grid_size: (rows, cols) of the grid
        cell_size: Size of each cell
    
    Returns:
        Grid image
    """
    rows, cols = grid_size
    padding = 5
    
    # Create canvas
    canvas_h = rows * (cell_size + padding) + padding
    canvas_w = cols * (cell_size + padding) + padding
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240
    
    for idx, (seed, label, conf) in enumerate(zip(seeds, labels, confidences)):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        
        x = col * (cell_size + padding) + padding
        y = row * (cell_size + padding) + padding
        
        # Resize seed
        seed_resized = cv2.resize(seed, (cell_size - 10, cell_size - 30))
        
        # Get color
        color = COLORS.get(label, (128, 128, 128))
        
        # Draw cell background
        cv2.rectangle(canvas, (x, y), (x + cell_size - 10, y + cell_size - 10), color, 2)
        
        # Place seed
        canvas[y + 5:y + 5 + cell_size - 30, x + 5:x + 5 + cell_size - 10] = seed_resized
        
        # Add label
        label_text = f"{label[:4]}: {conf*100:.0f}%"
        cv2.putText(
            canvas,
            label_text,
            (x + 5, y + cell_size - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1
        )
    
    return canvas


def draw_legend(
    image: np.ndarray,
    position: str = "bottom-right"
) -> np.ndarray:
    """
    Add a color legend to the image.
    
    Args:
        image: Input image
        position: Legend position
    
    Returns:
        Image with legend
    """
    result = image.copy()
    h, w = result.shape[:2]
    
    legend_items = [
        ("REAL", COLORS["REAL"]),
        ("FAKE", COLORS["FAKE"]),
        ("LOW QUALITY", COLORS["LOW_QUALITY"])
    ]
    
    legend_w = 120
    legend_h = len(legend_items) * 25 + 10
    
    if position == "bottom-right":
        x = w - legend_w - 10
        y = h - legend_h - 10
    elif position == "top-right":
        x = w - legend_w - 10
        y = 10
    elif position == "bottom-left":
        x = 10
        y = h - legend_h - 10
    else:  # top-left
        x = 10
        y = 10
    
    # Draw legend background
    overlay = result.copy()
    cv2.rectangle(overlay, (x, y), (x + legend_w, y + legend_h), (255, 255, 255), -1)
    cv2.rectangle(overlay, (x, y), (x + legend_w, y + legend_h), (100, 100, 100), 1)
    cv2.addWeighted(overlay, 0.9, result, 0.1, 0, result)
    
    # Draw legend items
    for idx, (name, color) in enumerate(legend_items):
        item_y = y + 20 + idx * 25
        cv2.rectangle(result, (x + 5, item_y - 10), (x + 25, item_y + 5), color, -1)
        cv2.putText(result, name, (x + 30, item_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return result


def save_annotated_image(
    image: np.ndarray,
    output_path: str,
    quality: int = 95
):
    """
    Save annotated image to file.
    
    Args:
        image: Image to save
        output_path: Output file path
        quality: JPEG quality (1-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), image)
    
    print(f"Image saved to: {output_path}")


def numpy_to_base64(image: np.ndarray, format: str = 'jpeg') -> str:
    """
    Convert numpy image to base64 string for API responses.
    
    Args:
        image: Image in BGR format
        format: Output format ('jpeg' or 'png')
    
    Returns:
        Base64 encoded string
    """
    import base64
    
    if format == 'jpeg':
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        _, buffer = cv2.imencode('.png', image)
    
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_numpy(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to numpy image.
    
    Args:
        base64_string: Base64 encoded image
    
    Returns:
        Image in BGR format
    """
    import base64
    
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image


if __name__ == "__main__":
    # Test visualization functions
    print("Testing SeedShield Visualization Utilities")
    
    # Create test image
    test_img = np.ones((640, 640, 3), dtype=np.uint8) * 200
    
    # Test detections
    test_detections = [
        {'bbox': (50, 50, 80, 80), 'class_name': 'REAL', 'confidence': 0.95},
        {'bbox': (200, 100, 70, 70), 'class_name': 'FAKE', 'confidence': 0.87},
        {'bbox': (350, 200, 90, 90), 'class_name': 'LOW_QUALITY', 'confidence': 0.72},
    ]
    
    # Draw detections
    annotated = draw_seed_detection(test_img, test_detections)
    print(f"Drew {len(test_detections)} detections")
    
    # Add verdict
    stats = {'REAL': 5, 'FAKE': 2, 'LOW_QUALITY': 1}
    annotated = draw_verdict_overlay(annotated, 'genuine', stats)
    print("Added verdict overlay")
    
    # Add legend
    annotated = draw_legend(annotated)
    print("Added legend")
    
    # Test base64 conversion
    b64 = numpy_to_base64(annotated)
    recovered = base64_to_numpy(b64)
    print(f"Base64 conversion test passed: {recovered.shape == annotated.shape}")
    
    print("\nVisualization utilities test completed!")
