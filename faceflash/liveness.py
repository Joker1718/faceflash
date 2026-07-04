"""
Face anti-spoofing (liveness detection).

Single-frame liveness check using MiniFASNetV2 — a lightweight (~600KB) CNN
that classifies an 80x80 face crop as live or spoof (Silent-Face's real
class is index 1; the two spoof classes are reported together as "spoof").

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

# Silent-Face-Anti-Spoofing convention: the MiniFASNet head outputs 3 classes
# where index 1 is the REAL/live class; 0 and 2 are spoof types (2D print /
# 3D replay in the upstream repo, but we don't reliably distinguish them from
# this single model, so both report as "spoof").
_LIVE_CLASS = 1
_LABELS = ["spoof", "live", "spoof"]


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
        # label: "live" or "spoof"
    """

    def __init__(self):
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
            - label: "live" or "spoof"
        """
        inp = self.preprocess(face_img)
        outputs = self.session.run(None, {self.input_name: inp})

        # Output is logits (3 classes) — softmax to get probabilities
        logits = outputs[0][0]
        probs = _softmax(logits)

        # Silent-Face convention: class 1 is the live class. Score is P(live);
        # label is consistent — "live" iff class 1 wins.
        liveness_score = float(probs[_LIVE_CLASS])
        label = _LABELS[int(np.argmax(probs))]
        return liveness_score, label

    def check_batch(self, face_images: list) -> list:
        """Check liveness for multiple face crops.

        Returns list of (liveness_score, label) tuples.
        """
        return [self.check(img) for img in face_images]

    def check_frame(self, image_rgb: np.ndarray, box, bbox_scale: float = 2.7):
        """Correct liveness path: crop the face region from the FULL frame at
        the scale the model was trained on, not a tight aligned crop.

        MiniFASNet '2.7_80x80' expects a crop that is ~2.7x the face box —
        the face plus surrounding context. Passing a tight, aligned face
        (the ArcFace crop) is out of distribution and flags live faces as
        spoof. `box` is [x1, y1, x2, y2] in image pixels.
        """
        crop = _crop_with_scale(image_rgb, box, bbox_scale)
        return self.check(crop)


def _crop_with_scale(image, box, scale):
    """Expand a face box by `scale` around its center (clamped to the image)
    and return that region — mirrors Silent-Face-Anti-Spoofing's CropImage,
    which is how the MiniFASNet weights were trained."""
    src_h, src_w = image.shape[:2]
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    # don't ask for more context than the image has
    scale = min(scale, (src_h - 1) / h, (src_w - 1) / w)
    new_w, new_h = w * scale, h * scale
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    left = int(round(cx - new_w / 2.0))
    top = int(round(cy - new_h / 2.0))
    right = int(round(left + new_w))
    bottom = int(round(top + new_h))
    left, top = max(0, left), max(0, top)
    right, bottom = min(src_w, right), min(src_h, bottom)
    return image[top:bottom, left:right]


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
