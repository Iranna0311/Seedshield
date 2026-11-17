"""
SeedShield - Inference Pipeline
Complete pipeline for seed detection and classification from a single image.
Uses OpenCV for detection and scikit-learn for classification.
No GPU or deep learning framework required.
"""

import os
import sys
import json
import uuid
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    low = max(0.0, centre - spread)
    high = min(1.0, centre + spread)
    return (low, high)


@dataclass
class SeedDetection:
    """Data class for a single seed detection."""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    class_name: str
    class_id: int
    detection_confidence: float
    classification_confidence: float
    bbox_norm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    probs: Dict[str, float] = field(default_factory=lambda: {"REAL": 0.0, "FAKE": 0.0, "LOW_QUALITY": 0.0})

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
    uncertain_count: int = 0
    uncertain_percentage: float = 0.0
    ci95: List[float] = field(default_factory=lambda: [0.0, 0.0])
    model_version: str = "1.0.0-sklearn"
    scan_id: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.annotated_image_base64 and len(self.annotated_image_base64) > 1000000:
            result['annotated_image_base64'] = None
            result['image_available'] = True
        return result


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_seed_features(roi: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Extract a 20-dimensional feature vector from a single seed crop + contour.

    Features (20 total):
      0  circularity         – 4π·area / perimeter²
      1  aspect_ratio        – width / height of bounding rect
      2  solidity            – contour area / convex hull area
      3  extent              – contour area / bounding rect area
      4  equiv_diameter_norm – normalised equivalent circular diameter
      5  mean_hue            – mean hue in [0,1]
      6  mean_sat            – mean saturation in [0,1]
      7  mean_val            – mean value/brightness in [0,1]
      8  std_hue
      9  std_sat
      10 std_val
      11 mean_r_norm
      12 mean_g_norm
      13 mean_b_norm
      14 std_r_norm
      15 std_g_norm
      16 std_b_norm
      17 color_uniformity    – 1 – (std_val / (mean_val+1e-6))
      18 lbp_uniformity      – fraction of uniform LBP patterns
      19 convexity           – convex hull perimeter / contour perimeter
    """
    feats = np.zeros(20, dtype=np.float32)

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, bw, bh = cv2.boundingRect(contour)

    # Shape features
    feats[0] = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    feats[1] = float(bw) / float(bh) if bh > 0 else 1.0
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    feats[2] = area / hull_area if hull_area > 0 else 1.0
    feats[3] = area / (bw * bh) if (bw * bh) > 0 else 1.0
    feats[4] = np.sqrt(4 * area / np.pi) / max(bw, bh) if max(bw, bh) > 0 else 1.0

    if roi.size == 0:
        return feats

    # Resize to fixed size for consistent statistics
    roi_resized = cv2.resize(roi, (32, 32))
    hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] /= 180.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0

    feats[5] = float(np.mean(hsv[:, :, 0]))
    feats[6] = float(np.mean(hsv[:, :, 1]))
    feats[7] = float(np.mean(hsv[:, :, 2]))
    feats[8] = float(np.std(hsv[:, :, 0]))
    feats[9] = float(np.std(hsv[:, :, 1]))
    feats[10] = float(np.std(hsv[:, :, 2]))

    bgr = roi_resized.astype(np.float32) / 255.0
    feats[11] = float(np.mean(bgr[:, :, 2]))  # R
    feats[12] = float(np.mean(bgr[:, :, 1]))  # G
    feats[13] = float(np.mean(bgr[:, :, 0]))  # B
    feats[14] = float(np.std(bgr[:, :, 2]))
    feats[15] = float(np.std(bgr[:, :, 1]))
    feats[16] = float(np.std(bgr[:, :, 0]))

    feats[17] = 1.0 - feats[10] / (feats[7] + 1e-6)

    # Simplified LBP-like uniformity via gradient magnitude
    gray32 = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    feats[18] = 1.0 - float(np.mean(grad_mag)) / 255.0  # high = smooth = uniform

    hull_perim = float(cv2.arcLength(hull, True))
    feats[19] = hull_perim / perimeter if perimeter > 0 else 1.0

    return feats


def _generate_synthetic_training_data(n_per_class: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic feature vectors for REAL, FAKE, LOW_QUALITY seeds.

    Characteristics are grounded in agronomic observations:
    - REAL: uniform oval/round shape, warm brown/tan color, even surface
    - FAKE: unusual shape or color (too saturated, wrong hue), irregular edges
    - LOW_QUALITY: damaged shape, shriveled (low solidity), discolored
    """
    rng = np.random.default_rng(seed)
    feats_list = []
    labels = []

    def clip(arr, lo=0.0, hi=1.0):
        return np.clip(arr, lo, hi)

    n = n_per_class

    # ---- REAL seeds (class 0) ----
    real_circ = rng.beta(8, 2, n) * 0.45 + 0.55       # 0.55–1.0
    real_ar   = rng.normal(1.0, 0.15, n)               # ~1.0 aspect ratio
    real_sol  = rng.beta(9, 1.5, n) * 0.2 + 0.80      # 0.80–1.00
    real_ext  = rng.beta(7, 2, n) * 0.3 + 0.65        # 0.65–0.95
    real_eqd  = rng.normal(0.88, 0.06, n)
    real_hue  = rng.normal(0.10, 0.03, n)              # warm reddish-brown hue
    real_sat  = rng.beta(4, 3, n) * 0.5 + 0.25        # 0.25–0.75
    real_val  = rng.normal(0.62, 0.07, n)              # medium brightness
    real_hs   = rng.uniform(0.02, 0.07, n)
    real_ss   = rng.uniform(0.03, 0.10, n)
    real_vs   = rng.uniform(0.04, 0.10, n)
    real_r    = rng.normal(0.60, 0.07, n)
    real_g    = rng.normal(0.48, 0.06, n)
    real_b    = rng.normal(0.32, 0.05, n)
    real_sr   = rng.uniform(0.03, 0.09, n)
    real_sg   = rng.uniform(0.03, 0.08, n)
    real_sb   = rng.uniform(0.02, 0.07, n)
    real_cu   = rng.normal(0.82, 0.07, n)              # high color uniformity
    real_lbp  = rng.normal(0.72, 0.08, n)              # fairly smooth surface
    real_conv = rng.beta(9, 1.5, n) * 0.15 + 0.85     # high convexity

    real = np.stack([
        clip(real_circ), clip(real_ar, 0.5, 2.5), clip(real_sol),
        clip(real_ext), clip(real_eqd), clip(real_hue),
        clip(real_sat), clip(real_val), clip(real_hs),
        clip(real_ss), clip(real_vs), clip(real_r),
        clip(real_g), clip(real_b), clip(real_sr),
        clip(real_sg), clip(real_sb), clip(real_cu),
        clip(real_lbp), clip(real_conv)
    ], axis=1)

    # ---- FAKE seeds (class 1) ----
    fake_circ = rng.beta(2, 4, n) * 0.55 + 0.1        # 0.1–0.65 irregular
    _ar_opts = np.stack([rng.normal(0.5, 0.2, n), rng.normal(2.0, 0.3, n)], axis=1)
    fake_ar   = _ar_opts[np.arange(n), rng.integers(0, 2, n)]  # extreme aspect ratios
    fake_sol  = rng.beta(3, 3, n) * 0.4 + 0.35        # 0.35–0.75
    fake_ext  = rng.beta(3, 3, n) * 0.4 + 0.30
    fake_eqd  = rng.normal(0.65, 0.15, n)
    _hue_opts = np.stack([rng.normal(0.35, 0.08, n), rng.normal(0.65, 0.08, n)], axis=1)
    fake_hue  = _hue_opts[np.arange(n), rng.integers(0, 2, n)]
    fake_sat  = rng.beta(2, 1.5, n) * 0.5 + 0.50      # high saturation – unnatural
    _val_opts = np.stack([rng.normal(0.25, 0.08, n), rng.normal(0.90, 0.05, n)], axis=1)
    fake_val  = _val_opts[np.arange(n), rng.integers(0, 2, n)]  # too dark or too bright
    fake_hs   = rng.uniform(0.10, 0.25, n)
    fake_ss   = rng.uniform(0.12, 0.28, n)
    fake_vs   = rng.uniform(0.12, 0.30, n)
    fake_r    = rng.normal(0.40, 0.15, n)
    fake_g    = rng.normal(0.55, 0.15, n)              # greenish
    fake_b    = rng.normal(0.30, 0.10, n)
    fake_sr   = rng.uniform(0.10, 0.22, n)
    fake_sg   = rng.uniform(0.10, 0.22, n)
    fake_sb   = rng.uniform(0.08, 0.18, n)
    fake_cu   = rng.normal(0.45, 0.12, n)              # non-uniform color
    fake_lbp  = rng.normal(0.45, 0.12, n)              # rough / inconsistent surface
    fake_conv = rng.beta(2, 4, n) * 0.4 + 0.30        # low convexity – jagged

    fake = np.stack([
        clip(fake_circ), clip(fake_ar, 0.2, 4.0), clip(fake_sol),
        clip(fake_ext), clip(fake_eqd), clip(fake_hue),
        clip(fake_sat), clip(fake_val), clip(fake_hs),
        clip(fake_ss), clip(fake_vs), clip(fake_r),
        clip(fake_g), clip(fake_b), clip(fake_sr),
        clip(fake_sg), clip(fake_sb), clip(fake_cu),
        clip(fake_lbp), clip(fake_conv)
    ], axis=1)

    # ---- LOW_QUALITY seeds (class 2) ----
    lq_circ = rng.beta(3, 3, n) * 0.4 + 0.30          # 0.30–0.70
    lq_ar   = rng.normal(1.2, 0.35, n)                 # slightly off
    lq_sol  = rng.beta(3, 4, n) * 0.35 + 0.40         # 0.40–0.75 – shriveled
    lq_ext  = rng.beta(3, 4, n) * 0.35 + 0.35
    lq_eqd  = rng.normal(0.72, 0.12, n)
    lq_hue  = rng.normal(0.08, 0.06, n)                # similar hue but faded
    lq_sat  = rng.beta(2, 4, n) * 0.30 + 0.08         # low saturation – pale
    lq_val  = rng.normal(0.45, 0.12, n)                # darker / duller
    lq_hs   = rng.uniform(0.04, 0.14, n)
    lq_ss   = rng.uniform(0.05, 0.15, n)
    lq_vs   = rng.uniform(0.08, 0.18, n)
    lq_r    = rng.normal(0.50, 0.08, n)
    lq_g    = rng.normal(0.43, 0.07, n)
    lq_b    = rng.normal(0.28, 0.06, n)
    lq_sr   = rng.uniform(0.06, 0.14, n)
    lq_sg   = rng.uniform(0.05, 0.13, n)
    lq_sb   = rng.uniform(0.04, 0.11, n)
    lq_cu   = rng.normal(0.62, 0.10, n)
    lq_lbp  = rng.normal(0.55, 0.10, n)               # somewhat rough (damaged)
    lq_conv = rng.beta(4, 3, n) * 0.3 + 0.55          # medium convexity

    lq = np.stack([
        clip(lq_circ), clip(lq_ar, 0.3, 3.0), clip(lq_sol),
        clip(lq_ext), clip(lq_eqd), clip(lq_hue),
        clip(lq_sat), clip(lq_val), clip(lq_hs),
        clip(lq_ss), clip(lq_vs), clip(lq_r),
        clip(lq_g), clip(lq_b), clip(lq_sr),
        clip(lq_sg), clip(lq_sb), clip(lq_cu),
        clip(lq_lbp), clip(lq_conv)
    ], axis=1)

    X = np.vstack([real, fake, lq]).astype(np.float32)
    y = np.array([0] * n + [1] * n + [2] * n, dtype=np.int32)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _train_and_save_classifier(save_path: Path) -> Pipeline:
    """Train a Random Forest classifier on synthetic seed features and save it."""
    print("Training seed classifier on synthetic data…")
    X, y = _generate_synthetic_training_data(n_per_class=2000)
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    clf.fit(X, y)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"Classifier saved to {save_path}")
    return clf


def _load_or_train_classifier(save_path: Path) -> Pipeline:
    """Load classifier from disk, or train a new one if not found."""
    if save_path.exists():
        try:
            clf = joblib.load(save_path)
            print(f"Classifier loaded from {save_path}")
            return clf
        except Exception as e:
            print(f"Could not load classifier ({e}), retraining…")
    return _train_and_save_classifier(save_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class SeedShieldPipeline:
    """
    Complete inference pipeline for SeedShield.

    Uses OpenCV contour detection for seed segmentation and a pre-trained
    Random Forest classifier (scikit-learn) for REAL / FAKE / LOW_QUALITY
    classification.  No GPU or deep-learning framework is required.

    Pipeline steps
    1. Load image
    2. Detect individual seeds via adaptive thresholding + contour analysis
    3. Extract a 20-dimensional feature vector per seed
    4. Classify each seed with the Random Forest model
    5. Draw bounding boxes with class-specific colours
    6. Calculate overall verdict
    7. Return annotated image and JSON results
    """

    #: Path where the trained sklearn model is stored
    _DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "weights" / "classifier" / "seed_classifier.joblib"

    def __init__(self,
                 classifier_weights: str = None,
                 device: str = "cpu"):
        self.config = get_inference_config()
        self.paths = get_path_config()
        self.device = "cpu"   # sklearn runs on CPU

        model_path = Path(classifier_weights) if classifier_weights else self._DEFAULT_MODEL_PATH
        self._classifier = _load_or_train_classifier(model_path)
        print("SeedShield Pipeline ready (sklearn / OpenCV backend)")

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, image_source: Union[str, np.ndarray, "Image.Image"]) -> np.ndarray:
        """Load image from path, numpy array, or PIL Image.  Returns BGR ndarray."""
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"Could not load image: {image_source}")
        elif isinstance(image_source, Image.Image):
            img = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        elif isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                img = cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
            elif image_source.shape[2] == 4:
                img = cv2.cvtColor(image_source, cv2.COLOR_RGBA2BGR)
            else:
                img = image_source.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")
        return img

    # ------------------------------------------------------------------
    # Seed detection (OpenCV-based)
    # ------------------------------------------------------------------

    def _detect_contours(self, image: np.ndarray) -> List:
        """Return a list of valid seed contours from the image."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bilateral filter preserves edges while reducing noise
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # Otsu + adaptive threshold combined
        _, th_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th_adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        thresh = cv2.bitwise_or(th_otsu, th_adapt)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(60, h * w * 0.0005)
        max_area = h * w * 0.25
        valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

        if not valid:
            # Fallback: Canny edges
            edges = cv2.Canny(blurred, 30, 100)
            edges = cv2.dilate(edges, kernel, iterations=2)
            contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours2 if min_area < cv2.contourArea(c) < max_area]

        # Limit to 100 seeds
        valid.sort(key=cv2.contourArea, reverse=True)
        return valid[:self.config.max_detections]

    def detect_seeds(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all seeds in image and return raw detection dicts
        (no classification yet).
        """
        h, w = image.shape[:2]
        contours = self._detect_contours(image)

        detections = []
        for i, cnt in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            # Detection confidence from circularity
            perim = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.5
            det_conf = float(np.clip(0.5 + circ * 0.45, 0.3, 0.99))
            detections.append({
                'id': i,
                'bbox': (x, y, bw, bh),
                'contour': cnt,
                'detection_confidence': det_conf,
            })

        return detections

    # ------------------------------------------------------------------
    # Seed classification
    # ------------------------------------------------------------------

    def classify_seed(self, roi: np.ndarray, contour: np.ndarray) -> Tuple[str, int, float, np.ndarray]:
        """Classify a single seed.  Returns (class_name, class_id, confidence, proba_arr)."""
        feats = _extract_seed_features(roi, contour).reshape(1, -1)
        proba = self._classifier.predict_proba(feats)[0]
        if max(proba) < 0.6:
            class_name = "UNCERTAIN"
            class_id = 3
            confidence = float(max(proba))
        else:
            class_id = int(np.argmax(proba))
            confidence = float(proba[class_id])
            class_name = CLASS_NAMES[class_id]
        return class_name, class_id, confidence, proba

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    def calculate_verdict(self, detections: List[Dict]) -> Tuple[str, str]:
        if not detections:
            return "genuine", VERDICT_MESSAGES["genuine"]

        total = len(detections)
        fake_n = sum(1 for d in detections if d.get('class_name') == 'FAKE')
        lq_n   = sum(1 for d in detections if d.get('class_name') == 'LOW_QUALITY')

        fake_r = fake_n / total
        lq_r   = lq_n / total

        if fake_r > self.config.fake_threshold:
            return "suspected_fake", VERDICT_MESSAGES["suspected_fake"]
        if lq_r > self.config.low_quality_threshold:
            return "low_quality", VERDICT_MESSAGES["low_quality"]
        if (fake_r + lq_r) > 0.5:
            return "low_quality", VERDICT_MESSAGES["low_quality"]
        return "genuine", VERDICT_MESSAGES["genuine"]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_image(self,
                      image_source: Union[str, np.ndarray, "Image.Image"],
                      return_annotated: bool = True,
                      save_path: Optional[str] = None) -> InferenceResult:
        """Run the complete pipeline on a single image."""
        image = self.load_image(image_source)
        h, w = image.shape[:2]

        # Detect
        detections = self.detect_seeds(image)

        # Classify
        for det in detections:
            x, y, bw, bh = det['bbox']
            roi = image[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
            if roi.size > 0:
                cls_name, cls_id, conf, proba = self.classify_seed(roi, det['contour'])
            else:
                cls_name, cls_id, conf = 'REAL', 0, 0.5
                proba = np.array([0.5, 0.25, 0.25])
            det['class_name'] = cls_name
            det['class_id'] = cls_id
            det['classification_confidence'] = round(conf, 4)
            det['confidence'] = round(conf, 4)
            # Normalized bbox
            det['bbox_norm'] = [
                x / w, y / h, bw / w, bh / h
            ]
            # Per-class probabilities (proba has 3 entries: REAL, FAKE, LOW_QUALITY)
            det['probs'] = {
                "REAL": float(proba[0]),
                "FAKE": float(proba[1]),
                "LOW_QUALITY": float(proba[2]),
            }

        # Statistics
        total = len(detections)
        if total == 0:
            raise ValueError("No seeds detected in this image. Only seed photos are accepted. Please place seeds on a plain background and try again.")
        real_n = sum(1 for d in detections if d['class_name'] == 'REAL')
        fake_n = sum(1 for d in detections if d['class_name'] == 'FAKE')
        lq_n   = sum(1 for d in detections if d['class_name'] == 'LOW_QUALITY')
        uncertain_n = sum(1 for d in detections if d['class_name'] == 'UNCERTAIN')

        r_pct  = real_n / total * 100 if total else 0.0
        f_pct  = fake_n / total * 100 if total else 0.0
        lq_pct = lq_n  / total * 100 if total else 0.0
        u_pct  = uncertain_n / total * 100 if total else 0.0

        ci_low, ci_high = wilson(fake_n, total)

        verdict_key, verdict_message = self.calculate_verdict(detections)

        # Visualisation
        annotated_b64 = None
        if return_annotated or save_path:
            stats_d = {'REAL': real_n, 'FAKE': fake_n, 'LOW_QUALITY': lq_n}
            annotated = create_result_visualization(image, detections, verdict_key, stats_d)
            annotated = draw_legend(annotated)
            if save_path:
                save_annotated_image(annotated, save_path)
            if return_annotated:
                annotated_b64 = numpy_to_base64(annotated)

        result = InferenceResult(
            total_seeds=total,
            real_count=real_n,
            fake_count=fake_n,
            low_quality_count=lq_n,
            real_percentage=round(r_pct, 2),
            fake_percentage=round(f_pct, 2),
            low_quality_percentage=round(lq_pct, 2),
            verdict=verdict_key,
            verdict_message=verdict_message,
            detections=[SeedDetection(
                id=d['id'],
                bbox=d['bbox'],
                class_name=d['class_name'],
                class_id=d['class_id'],
                detection_confidence=d['detection_confidence'],
                classification_confidence=d['classification_confidence'],
                bbox_norm=d['bbox_norm'],
                probs=d['probs'],
            ).to_dict() for d in detections],
            annotated_image_base64=annotated_b64,
            uncertain_count=uncertain_n,
            uncertain_percentage=round(u_pct, 2),
            ci95=[round(ci_low, 6), round(ci_high, 6)],
            model_version="1.0.0-sklearn",
            scan_id=str(uuid.uuid4()),
        )
        return result

    def get_annotated_image(self,
                            image_source: Union[str, np.ndarray, "Image.Image"],
                            result: Optional[InferenceResult] = None) -> np.ndarray:
        if result is None:
            result = self.process_image(image_source, return_annotated=False)
        image = self.load_image(image_source)
        stats_d = {'REAL': result.real_count, 'FAKE': result.fake_count, 'LOW_QUALITY': result.low_quality_count}
        dets = [{'bbox': d['bbox'], 'class_name': d['class_name'], 'confidence': d['classification_confidence']}
                for d in result.detections]
        annotated = create_result_visualization(image, dets, result.verdict, stats_d)
        return draw_legend(annotated)


# ---------------------------------------------------------------------------
# Helpers used by the FastAPI demo mode
# ---------------------------------------------------------------------------

def create_demo_prediction() -> List[Dict]:
    """Return a small list of fake detection dicts (for unit-testing only)."""
    rng = np.random.default_rng(0)
    detections = []
    for i in range(8):
        cls = rng.choice(['REAL', 'FAKE', 'LOW_QUALITY'], p=[0.6, 0.2, 0.2])
        cls = str(cls)
        x = int(rng.integers(50, 400))
        y = int(rng.integers(50, 400))
        bw = int(rng.integers(40, 90))
        bh = int(rng.integers(40, 90))
        # Synthetic probs consistent with class
        if cls == 'REAL':
            probs = {"REAL": 0.80, "FAKE": 0.10, "LOW_QUALITY": 0.10}
        elif cls == 'FAKE':
            probs = {"REAL": 0.10, "FAKE": 0.80, "LOW_QUALITY": 0.10}
        else:
            probs = {"REAL": 0.10, "FAKE": 0.10, "LOW_QUALITY": 0.80}
        detections.append({
            'id': i,
            'seed_id': f"demo-{i}",
            'bbox': (x, y, bw, bh),
            'bbox_norm': [x / 640.0, y / 640.0, bw / 640.0, bh / 640.0],
            'class_name': cls,
            'class_id': CLASS_NAMES.index(cls),
            'detection_confidence': float(rng.uniform(0.70, 0.99)),
            'classification_confidence': float(rng.uniform(0.70, 0.99)),
            'probs': probs,
        })
    return detections


def main():
    """CLI entry-point for testing the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='SeedShield Inference Pipeline')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default=None, help='Output annotated image path')
    parser.add_argument('--json-output', type=str, default=None, help='Path to save JSON results')
    args = parser.parse_args()

    pipeline = SeedShieldPipeline()
    result = pipeline.process_image(args.image, return_annotated=True, save_path=args.output)

    print("\n" + "=" * 50)
    print("SEEDSHIELD ANALYSIS RESULTS")
    print("=" * 50)
    print(f"\nTotal Seeds: {result.total_seeds}")
    print(f"  REAL:        {result.real_count} ({result.real_percentage:.1f}%)")
    print(f"  FAKE:        {result.fake_count} ({result.fake_percentage:.1f}%)")
    print(f"  LOW QUALITY: {result.low_quality_count} ({result.low_quality_percentage:.1f}%)")
    print(f"\nVerdict: {result.verdict_message}")

    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nJSON results saved to: {args.json_output}")


if __name__ == "__main__":
    main()
