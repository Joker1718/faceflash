"""
Face anti-spoofing (liveness detection).

Single-frame liveness check using MiniFASNetV2 — a lightweight (~600KB) CNN
that classifies an 80x80 face crop as live, print-attack, or replay-attack.

Returns a liveness score (0-1) and a label. Intended to gate identification
flows against printed-photo and screen-replay attacks. Not foolproof against
3D masks or high-quality deepfakes — treat it as a first-line filter.

Model: MiniFASNetV2 (2.7_80x80) from Silent-Face-Anti-Spoofing (minivision-ai).
ONNX export from HuggingFace: garciafido/minifasnet-v2-anti-spoofing-onnx
"""

import logging
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/"
    "resolve/main/minifasnet_v2.onnx"
)
MODEL_DIR = Path.home() / ".faceflash" / "models"
MODEL_PATH = MODEL_DIR / "minifasnet_v2.onnx"

# Class labels from the model
_LABELS = ["live", "print_attack", "replay_attack"]


def _ensure_model() -> Path:
    """Download the anti-spoofing model if not cached."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading anti-spoofing model (~600KB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Saved anti-spoofing model to %s", MODEL_PATH)
    return MODEL_PATH


class LivenessDetector:
    """Single-frame face anti-spoofing using MiniFASNetV2.

    Usage:
        detector = LivenessDetector()
        score, label = detector.check(face_crop)
        # score: 0.0 (fake) to 1.0 (live)
        # label: "live", "print_attack", or "replay_attack"
    """

    def __init__(self):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for liveness detection. "
                "Install: pip install faceflash[cpu]"
            ) from e

        model_path = _ensure_model()
        from faceflash.runtime import make_session
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Resize to 80x80, normalize, convert to NCHW float32.

        Input: RGB face crop (any size, HWC uint8 or float).
        Output: (1, 3, 80, 80) float32 tensor normalized to [0, 1].
        """
        from PIL import Image

        if isinstance(face_img, np.ndarray):
            if face_img.dtype != np.uint8:
                face_img = (face_img * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(face_img)
        else:
            img = face_img

        # Resize to 80x80
        img = img.resize((80, 80), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0

        # RGB → BGR (model trained on BGR)
        arr = arr[:, :, ::-1].copy()

        # HWC → CHW → NCHW
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def check(self, face_img: np.ndarray) -> Tuple[float, str]:
        """Check if a face crop is live or spoofed.

        Args:
            face_img: RGB face crop as numpy array (HWC). Can be any size,
                      will be resized to 80x80. For best results, pass the
                      detected face region (not the full image).

        Returns:
            (liveness_score, label)
            - liveness_score: 0.0 (definitely fake) to 1.0 (definitely live)
            - label: "live", "print_attack", or "replay_attack"
        """
        inp = self.preprocess(face_img)
        outputs = self.session.run(None, {self.input_name: inp})

        # Output is logits (3 classes) — softmax to get probabilities
        logits = outputs[0][0]
        probs = _softmax(logits)

        # Class 0 = live, class 1 = print, class 2 = replay
        liveness_score = float(probs[0])
        predicted_class = int(np.argmax(probs))
        label = _LABELS[predicted_class]

        return liveness_score, label

    def check_batch(self, face_images: list) -> list:
        """Check liveness for multiple face crops.

        Returns list of (liveness_score, label) tuples.
        """
        return [self.check(img) for img in face_images]


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
