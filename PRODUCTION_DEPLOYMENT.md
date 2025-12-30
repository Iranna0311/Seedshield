# SeedShield Production Deployment Guide

This guide covers the next steps for deploying SeedShield to production.

---

## 1. Train Models with Real Seed Data

### Collect Real Seed Images

For best results, collect images that represent:
- **Real seeds**: Certified, healthy seeds from trusted sources
- **Fake seeds**: Known counterfeit or adulterated seeds (if available)
- **Low quality seeds**: Damaged, broken, or defective seeds

**Image requirements:**
- Smartphone camera (simulates real use case)
- Multiple seeds per image (5-30 seeds)
- Various lighting conditions
- Different backgrounds (paper, cloth, table)

### Organize Dataset

```
dataset/
├── real_seeds/
│   ├── wheat/
│   ├── rice/
│   └── corn/
├── fake_seeds/
│   └── known_counterfeits/
└── low_quality/
    ├── broken/
    ├── discolored/
    └── damaged/
```

### Retrain Models

```powershell
# 1. Update dataset configuration
# Edit dataset/generate_synthetic_dataset.py to load your real images

# 2. Train detector with more epochs
python models/train_detector.py --epochs 200 --batch-size 16

# 3. Train classifier
python models/train_classifier.py --epochs 100 --evaluate

# 4. Export for mobile
python deployment/convert_to_tflite.py --classifier --detector
python deployment/convert_to_onnx.py --classifier --detector
```

---

## 2. Deploy to Cloud

### Option A: Render (Recommended for Startups)

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial SeedShield deployment"
git remote add origin https://github.com/yourusername/seedshield.git
git push -u origin main
```

2. **Create Render Web Service**
- Go to [render.com](https://render.com)
- New → Web Service → Connect GitHub repo
- Configure:
  - **Name**: seedshield-api
  - **Region**: Choose closest to users
  - **Build Command**: `pip install -r api/requirements.txt`
  - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
  - **Plan**: Free or Starter ($7/month)

3. **Environment Variables**
```
SEEDSHIELD_DEMO_MODE=false
PYTHONPATH=/opt/render/project/src
```

4. **Deploy Frontend**
- Create another Web Service for frontend
- Or use Render Static Sites for built React app

### Option B: Railway

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
railway login
```

2. **Deploy**
```bash
railway init
railway up
```

3. **Add environment variables in Railway dashboard**

### Option C: AWS EC2

1. **Launch EC2 Instance**
- AMI: Amazon Linux 2 or Ubuntu 22.04
- Instance type: t3.medium (for inference)
- Storage: 20GB

2. **SSH and Setup**
```bash
ssh -i your-key.pem ec2-user@your-ip

# Install dependencies
sudo yum update -y
sudo yum install python3-pip docker git -y

# Clone and run
git clone https://github.com/yourusername/seedshield.git
cd seedshield
docker-compose up -d
```

3. **Configure Security Group**
- Inbound: Port 80, 443 from anywhere
- Outbound: All traffic

4. **Add HTTPS with Let's Encrypt**
```bash
sudo yum install certbot
sudo certbot --nginx -d yourdomain.com
```

---

## 3. Build Android App

### Quick Start

1. **Open Android Studio**
2. **Create New Project** → Empty Compose Activity
3. **Copy files from** `deployment/android/INTEGRATION_GUIDE.md`
4. **Add TFLite models** to `app/src/main/assets/`
5. **Build and test** on physical device

### Key Files to Create

```
app/
├── src/main/
│   ├── assets/
│   │   ├── seedshield_detector.tflite
│   │   └── seedshield_classifier.tflite
│   ├── java/com/seedshield/
│   │   ├── MainActivity.kt
│   │   └── analyzer/
│   │       └── SeedShieldAnalyzer.kt
│   └── res/layout/
│       └── activity_main.xml
├── build.gradle
└── AndroidManifest.xml
```

### Testing

1. Use Android Emulator for basic tests
2. Test on physical device for accurate performance
3. Test with various seed images
4. Measure inference time (should be <100ms)

---

## 4. Collect Farmer Feedback

### Feedback Collection Methods

1. **In-App Feedback Button**
   - Add to frontend: "Was this helpful? 👍 👎"
   - Track click events

2. **Analytics Dashboard**
   - Track predictions per day
   - Monitor verdict distribution
   - Log error rates

3. **User Surveys**
   - Send follow-up surveys
   - Ask about accuracy
   - Gather feature requests

### Key Metrics to Track

| Metric | Target | Action if Below |
|--------|--------|-----------------|
| Accuracy | >90% | Collect more training data |
| False Positive Rate | <5% | Adjust confidence thresholds |
| Inference Time | <2s | Optimize model or use GPU |
| User Retention | >60% | Improve UX |

---

## 5. Iterate and Improve

### Continuous Improvement Cycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Collect   │────▶│   Retrain   │────▶│   Deploy    │
│  Feedback   │     │   Models    │     │   Update    │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                       │
       └───────────────────────────────────────┘
```

### Version Control for Models

```bash
# Tag model versions
git tag -a v1.0.0-model -m "Initial trained model"

# Store model weights separately
aws s3 cp models/weights/ s3://seedshield-models/v1.0.0/ --recursive
```

### A/B Testing

For major model updates:
1. Deploy new model to 10% of users
2. Compare accuracy metrics
3. Gradually increase if performance improves

---

## Cost Estimates

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Render | 750 hrs/month | $7-25/month |
| Railway | $5 credit | $5-20/month |
| AWS EC2 | 750 hrs (t2.micro) | $15-50/month |
| Domain | - | $10-15/year |

---

## Security Checklist

- [ ] Use HTTPS everywhere
- [ ] Set proper CORS origins
- [ ] Rate limit API endpoints
- [ ] Validate all file uploads
- [ ] Store secrets in environment variables
- [ ] Regular security updates
- [ ] Monitor for unusual activity

---

## Support Contacts

For technical issues:
- GitHub Issues
- Email: support@seedshield.app

For dataset contributions:
- Email seed images to: data@seedshield.app
