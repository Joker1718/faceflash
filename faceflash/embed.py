"""ArcFace ONNX embedding. Downloads model on first use (~166MB)."""

import logging
import urllib.request
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".faceflash" / "models"

# Two embedding models, one storage format (both output 512-d):
#   accurate — ArcFace w600k_r50, the model every benchmark was measured with.
#   light    — MobileFaceNet (buffalo_s). ~8x faster on plain CPU
#              (measured 6.1ms vs 50.9ms per face); slightly lower accuracy.
# The two produce INCOMPATIBLE embedding spaces: a gallery enrolled with one
# cannot be searched with the other. The index records its profile and the
# engine refuses to mix them.
PROFILES = {
    "accurate": {
        "url": ("https://huggingface.co/public-data/insightface/resolve/main/"
                "models/buffalo_l/w600k_r50.onnx"),
        "file": "w600k_r50.onnx",
        "size_mb": 166,
    },
    "light": {
        "url": ("https://huggingface.co/immich-app/buffalo_s/resolve/main/"
                "recognition/model.onnx"),
        "file": "w600k_mbf.onnx",
        "size_mb": 13,
    },
}
# Older installs (and the RunPod script) cached w600k_r50 under a misleading
# name; reuse it instead of re-downloading 166MB.
_LEGACY_MODEL_PATH = MODEL_DIR / "arcface_r100.onnx"

# Backward-compat module attributes
MODEL_URL = PROFILES["accurate"]["url"]
MODEL_PATH = MODEL_DIR / PROFILES["accurate"]["file"]


def ensure_model(profile: str = "accurate"):
    """Download the profile's embedding model if missing."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; use one of {list(PROFILES)}")
    spec = PROFILES[profile]
    path = MODEL_DIR / spec["file"]
    if path.exists():
        return path
    if profile == "accurate" and _LEGACY_MODEL_PATH.exists():
        return _LEGACY_MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s embedding model (~%dMB)...", profile, spec["size_mb"])
    urllib.request.urlretrieve(spec["url"], path)
    logger.info("Saved model to %s", path)
    return path


class FaceEmbedder:
    """ArcFace ONNX inference. Picks GPU if available."""

    def __init__(self, profile: str = "accurate"):
        self.profile = profile
        model_path = ensure_model(profile)
        from faceflash.runtime import make_session
        self.session = make_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 112, 112]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize to 112x112, normalize to [-1,1], HWC→NCHW."""
        from PIL import Image

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Resize to 112x112
        img = img.resize((112, 112), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)

        # Normalize to [-1, 1]
        img = (img - 127.5) / 127.5

        # HWC -> CHW -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def embed(self, face_img: np.ndarray) -> np.ndarray:
        """512-dim L2-normalized embedding from an aligned face."""
        inp = self.preprocess(face_img)
        outputs = self.session.run(None, {self.input_name: inp})
        embedding = outputs[0][0]
        # L2 normalize (guard against degenerate zero-norm)
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            raise ValueError("Face embedding has zero norm (degenerate face detection)")
        embedding = embedding / norm
        return embedding

    def embed_batch(self, face_images: list) -> np.ndarray:
        """Embed many aligned faces in one ONNX call.

        Stacks preprocessed images into one (N, 3, 112, 112) tensor. Falls
        back to per-image inference if the model's batch axis is fixed at 1.
        """
        if not face_images:
            return np.empty((0, 512), dtype=np.float32)

        batch_dim = self.input_shape[0]
        if isinstance(batch_dim, int) and batch_dim == 1:
            return np.array([self.embed(img) for img in face_images])

        # CoreML compiles per input shape, so varying batch sizes trigger
        # recompilation and stacked inference comes out ~5x SLOWER than the
        # loop (measured on M-series). Batch only on CPU/CUDA providers.
        if self.session.get_providers()[0] == "CoreMLExecutionProvider":
            return np.array([self.embed(img) for img in face_images])

        inp = np.concatenate([self.preprocess(img) for img in face_images], axis=0)
        try:
            out = self.session.run(None, {self.input_name: inp})[0]
        except Exception:
            # Some exports declare a symbolic batch dim but only accept N=1
            return np.array([self.embed(img) for img in face_images])

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        if np.any(norms < 1e-10):
            raise ValueError("Face embedding has zero norm (degenerate face detection)")
        return out / norms
