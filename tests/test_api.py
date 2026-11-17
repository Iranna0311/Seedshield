import sys
import os
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["SEEDSHIELD_DEMO_MODE"] = "true"

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_valid():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 200), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 90, 100], fill=(140, 80, 30))
    draw.ellipse([120, 90, 170, 150], fill=(160, 90, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    for key in (
        "success",
        "total_seeds",
        "real_count",
        "fake_count",
        "low_quality_count",
        "verdict",
        "verdict_message",
        "detections",
    ):
        assert key in data, f"Missing key: {key}"


def test_predict_non_seed_rejected():
    from PIL import Image

    # Solid image without any seeds
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("plain.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 400
    msg = response.json().get("error", "") or response.json().get("detail", "")
    assert "No seeds detected" in msg


def test_predict_invalid_file():
    buf = io.BytesIO(b"not an image")
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 400


def test_predict_too_large():
    large_data = b"x" * (8 * 1024 * 1024 + 1)
    buf = io.BytesIO(large_data)
    response = client.post(
        "/predict",
        files={"file": ("big.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 413
