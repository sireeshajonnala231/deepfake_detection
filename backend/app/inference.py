"""Model loading and inference for the image deepfake detector."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from facenet_pytorch import MTCNN


logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnet_b0_stage2_best.pth")

# Pin the artifact evaluated for this service. Override only when deploying a
# deliberately replaced and re-evaluated model.
EXPECTED_MODEL_SHA256 = os.getenv(
    "MODEL_SHA256",
    "e7656a670847762bcedd26758362baa28fc8adbffbc9a2d5c20b8ace2eb99360",
).lower()

CLASS_NAMES = ("REAL", "FAKE")
REAL_INDEX = 0
FAKE_INDEX = 1
FINAL_THRESHOLD = 0.40
DEVICE = torch.device(os.getenv("DEEPFAKE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint() -> Mapping[str, object]:
    """Load the trusted checkpoint without enabling arbitrary pickle objects."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    actual_hash = _sha256(MODEL_PATH)
    if actual_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Model checkpoint hash does not match MODEL_SHA256.")

    # The existing checkpoint stores NumPy float64 metadata. These are the only
    # non-tensor globals it requires; arbitrary pickle loading remains disabled.
    from numpy._core.multiarray import scalar as numpy_scalar

    safe_globals = [
        (numpy_scalar, "numpy._core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Model checkpoint has an invalid format.")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise RuntimeError("Model checkpoint does not contain model_state_dict.")
    return checkpoint


mtcnn = MTCNN(
    image_size=224,
    margin=30,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    device=DEVICE,
)

transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
checkpoint = _load_checkpoint()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
del checkpoint
model.to(DEVICE)
model.eval()

logger.info("Deepfake detector loaded: architecture=EfficientNet-B0 device=%s threshold=%.2f", DEVICE, FINAL_THRESHOLD)


def predict_image(image: Image.Image) -> dict[str, float | str]:
    """Classify one detected face from a decoded, orientation-corrected image."""
    if image is None:
        raise ValueError("No image was provided.")

    image = image.convert("RGB")
    if min(image.size) < 40:
        raise ValueError("The image is too small to detect a face. Upload an image at least 40 pixels wide and high.")

    face = mtcnn(image)
    if face is None:
        raise ValueError("No face detected in the uploaded image.")
    if face.ndim != 3 or not torch.isfinite(face).all():
        raise ValueError("The detected face could not be processed.")

    face = transform(face.float().div(255.0)).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(face), dim=1)[0]

    real_probability = probabilities[REAL_INDEX].item()
    fake_probability = probabilities[FAKE_INDEX].item()
    is_fake = fake_probability >= FINAL_THRESHOLD

    return {
        "label": CLASS_NAMES[FAKE_INDEX if is_fake else REAL_INDEX],
        "decision_score": fake_probability if is_fake else real_probability,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "decision_threshold": FINAL_THRESHOLD,
    }
