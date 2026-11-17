# SeedShield 🌱🛡️

## AI-Powered Seed Authenticity Detection

SeedShield is a working prototype that detects **FAKE vs REAL seeds** from a single smartphone image. Built for farmers to combat counterfeit seed fraud.

TODO: ![Demo](docs/demo.gif)
Demo video: <link>

---

## ✨ Features

- **Multi-Seed Detection**: Detect all individual seeds in a single image
- **3-Class Classification**: Classify each seed as REAL, FAKE, or LOW_QUALITY
- **Visual Results**: Color-coded bounding boxes (🟢 Green = Real, 🔴 Red = Fake, 🟡 Yellow = Low Quality)
- **Confidence Scores**: Per-seed confidence percentages
- **Final Verdict**: Overall batch assessment
- **Mobile-Ready**: Optimized for low-end smartphones
- **API + Web App**: Complete deployment solution

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SEEDSHIELD ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐ │
│  │   FRONTEND   │     │   BACKEND    │     │        AI PIPELINE           │ │
│  │   (React)    │────▶│  (FastAPI)   │────▶│                              │ │
│  │              │     │              │     │  ┌──────────┐  ┌──────────┐  │ │
│  │  • Upload    │     │  • /predict  │     │  │ Stage 1  │  │ Stage 2  │  │ │
│  │  • Display   │◀────│  • Image     │◀────│  │ OpenCV   │─▶│  Random  │  │ │
│  │  • Verdict   │     │    Return    │     │  │ Contours │  │  Forest  │  │ │
│  └──────────────┘     └──────────────┘     │  └──────────┘  └──────────┘  │ │
│                                             └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
seedshield/
├── dataset/
│   ├── data_augmentation.py       # Augmentation & synthetic fake seed generation
│   ├── generate_synthetic_dataset.py  # Complete dataset generator
│   ├── data.yaml                  # YOLO dataset configuration
│   └── processed/                 # Generated datasets
│       ├── detection/             # YOLO format (images + labels)
│       └── classification/        # Image folders by class
├── models/
│   ├── model_config.py            # Configuration for all models
│   ├── train_detector.py          # YOLOv8 training script
│   ├── train_classifier.py        # MobileNetV2 training script
│   └── weights/                   # Saved model weights
├── inference/
│   ├── pipeline.py                # Complete inference pipeline
│   └── visualizer.py              # Visualization utilities
├── api/
│   ├── main.py                    # FastAPI backend
│   ├── schemas.py                 # Pydantic models
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js                 # Main React component
│   │   └── App.css                # Styling
│   └── package.json               # Node dependencies
├── deployment/
│   ├── convert_to_tflite.py       # TFLite conversion
│   └── convert_to_onnx.py         # ONNX conversion
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- CUDA (optional, for GPU training)

### 1. Clone and Install

```bash
# Clone the repository
cd seedshield

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r api/requirements.txt
```

### 2. Generate Dataset

```bash
# Generate synthetic training data
python dataset/generate_synthetic_dataset.py
```

This creates:
- 300 detection images (multi-seed composites)
- 600 classification samples per class

### 3. Train Models

```bash
# Train YOLOv8 detector
python models/train_detector.py --epochs 100

# Train MobileNetV2 classifier
python models/train_classifier.py --epochs 50 --evaluate
```

### 4. Run API Server

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`

### 5. Run Frontend

```bash
cd frontend
npm install
npm start
```

Frontend will be available at: `http://localhost:3000`

---

## 🔧 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Analyze seed image |
| `/predict/image` | POST | Get annotated image only |
| `/stats` | GET | Prediction statistics |
| `/config` | GET | Current configuration |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@seeds.jpg"
```

### Example Response

```json
{
  "success": true,
  "total_seeds": 15,
  "real_count": 10,
  "fake_count": 3,
  "low_quality_count": 2,
  "real_percentage": 66.67,
  "fake_percentage": 20.0,
  "low_quality_percentage": 13.33,
  "verdict": "suspected_fake",
  "verdict_message": "❌ Suspected Fake",
  "detections": [...],
  "annotated_image_base64": "..."
}
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t seedshield .

# Run container
docker run -p 8000:8000 seedshield
```

### Using Docker Compose

```bash
# Development mode
docker-compose up

# Production mode
docker-compose --profile production up
```

---

## ☁️ Cloud Deployment

### Render

1. Create new Web Service
2. Connect your GitHub repository
3. Set build command: `pip install -r api/requirements.txt`
4. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `SEEDSHIELD_DEMO_MODE=true`

### Railway

1. New project from GitHub
2. Add Python service
3. Set start command in Procfile:
   ```
   web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```

### AWS (EC2)

```bash
# SSH to instance
ssh -i key.pem ec2-user@your-ip

# Install dependencies
sudo yum update -y
sudo yum install python3-pip docker -y

# Clone and run
git clone <repo-url>
cd seedshield
docker-compose up -d
```

---

## 📱 Mobile Deployment (Android)

### Convert Models

```bash
# Convert to TFLite
python deployment/convert_to_tflite.py --classifier --detector

# Convert to ONNX
python deployment/convert_to_onnx.py --classifier --detector
```

### Android Integration

1. Copy `.tflite` files to `app/src/main/assets/`
2. Add TensorFlow Lite dependency:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

3. Load and run model:

```kotlin
// Load classifier
val classifier = Interpreter(loadModelFile("seedshield_classifier.tflite"))

// Preprocess image (224x224, normalized)
val inputBuffer = preprocessImage(bitmap)

// Run inference
val outputBuffer = Array(1) { FloatArray(3) }
classifier.run(inputBuffer, outputBuffer)

// Get prediction
val classes = arrayOf("REAL", "FAKE", "LOW_QUALITY")
val prediction = classes[outputBuffer[0].indices.maxByOrNull { outputBuffer[0][it] } ?: 0]
```

---

## 🌾 Extending to New Crops

### 1. Collect Seed Images

- Gather images of the new seed type (wheat, corn, rice, etc.)
- Ensure variety in lighting, backgrounds, and camera angles

### 2. Update Dataset Generator

```python
# In dataset/generate_synthetic_dataset.py
SEED_TYPES = ['wheat', 'rice', 'corn', 'soybean', 'your_new_seed']
```

### 3. Regenerate and Retrain

```bash
python dataset/generate_synthetic_dataset.py
python models/train_detector.py --epochs 100
python models/train_classifier.py --epochs 50
```

---

## How it works

Two-stage pipeline: OpenCV contour detection finds seed regions (no YOLO model loaded at runtime — detection uses classical computer vision). Each crop is passed to a Random Forest classifier trained on 20 hand-crafted shape/color/texture features. This avoids needing a GPU and keeps the binary under 50 MB.

---

## Explainability (Grad-CAM)

Not available in this release. The classifier is a Random Forest over hand-crafted features, not a convolutional network, so Grad-CAM does not apply. Listed in Roadmap.

---

## Uncertainty handling

When the classifier's maximum class probability is below 0.6, the seed is labelled UNCERTAIN rather than forced into REAL/FAKE/LOW_QUALITY. UNCERTAIN seeds are excluded from the fake-rate numerator. The batch fake rate is reported with a 95% Wilson confidence interval.

---

## Limitations

- The fake class is synthetically generated by perturbing feature distributions. The model has not been validated on real counterfeit seeds.
- Accuracy is sensitive to lighting, background uniformity, and seed type (only tuned for soybean-shaped features).
- Small or clustered seeds may be missed by the contour detector.
- No real counterfeit data has been collected yet.

---

## Roadmap

- Real-photo evaluation set with verified counterfeit samples
- One-class anomaly scoring (train only on genuine seeds)
- CLIP zero-shot baseline for new crop types
- ONNX Runtime Web for offline PWA on mobile
- CI pipeline with automated test runs

---

## ⚠️ Common Issues & Fixes

### Issue: "Model not found"

**Solution**: Run in demo mode or train models first:
```bash
export SEEDSHIELD_DEMO_MODE=true
python api/main.py
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size:
```bash
python models/train_detector.py --batch-size 8
python models/train_classifier.py --batch-size 16
```

### Issue: Low accuracy

**Solutions**:
1. Increase training epochs
2. Add more training data
3. Apply stronger augmentation
4. Fine-tune hyperparameters

### Issue: Slow inference

**Solutions**:
1. Use TFLite/ONNX for mobile
2. Enable GPU acceleration
3. Reduce image size
4. Use quantized models

---

## Results on synthetic validation set

> Note: the fake class is synthetically generated by perturbing real seed features. These numbers do not reflect real counterfeit detection accuracy.

| Model | Size | Accuracy (synthetic val set only) | Inference Time |
|-------|------|-----------------------------------|----------------|
| OpenCV Contour (Detector) | — | — | ~5ms (CPU) |
| Random Forest (Classifier) | <50 MB | ~90% | ~2ms (CPU) |

---

## 🔐 Security Considerations

- Validate all uploaded files
- Rate limit API endpoints
- Use HTTPS in production
- Sanitize file names
- Set CORS appropriately

---

## 📜 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/)
- [OpenCV](https://opencv.org/)
- [React](https://reactjs.org/)

---

## 📞 Support

For issues and feature requests, please open a GitHub issue.

---

**Built with ❤️ to protect farmers from seed fraud**
