"""
Face Flashing — active liveness detection via light reflection analysis.

Based on: "Face Flashing: a Secure Liveness Detection Protocol based on
Light Reflections" (Tang et al., 2018, arxiv.org/abs/1801.01949).

Displays random colored light patterns on screen, captures how the face
reflects them via camera. A real 3D face reflects unevenly; a flat photo
or screen produces uniform/incorrect reflections.

Requires: opencv-python, numpy. Needs a screen + camera.
"""

import time
import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_N_FLASHES = 8
DEFAULT_FLASH_DURATION = 0.15
DEFAULT_CAPTURE_DELAY = 0.05
DEFAULT_CONSISTENCY_THRESHOLD = 0.6


def _generate_flash_patterns(n: int, seed: Optional[int] = None) -> List[np.ndarray]:
    """Generate random solid-color flash patterns."""
    rng = np.random.default_rng(seed)
    return [rng.integers(50, 256, size=3).astype(np.uint8) for _ in range(n)]


def _extract_face_region(frame: np.ndarray) -> Optional[np.ndarray]:
    """Detect and crop the face from a frame."""
    try:
        from faceflash.detect import detect_and_align_info
        face, method = detect_and_align_info(frame)
        # A center crop is NOT a detected face — reflections off a wall
        # must not count toward the liveness decision.
        return None if method == "center_crop" else face
    except Exception:
        return None


def _compute_reflection_features(face_crop: np.ndarray, flash_color: np.ndarray) -> np.ndarray:
    """Extract per-zone reflection features from a face under a flash color.

    Divides face into 5 zones (forehead, left cheek, right cheek, nose, chin),
    measures intensity response per zone relative to the flash. Real faces have
    varying responses (3D geometry); flat surfaces respond uniformly.
    """
    h, w = face_crop.shape[:2]
    if h < 10 or w < 10:
        return np.zeros(15)

    zones = [
        face_crop[:h // 3, :],
        face_crop[h // 3:2 * h // 3, :w // 2],
        face_crop[h // 3:2 * h // 3, w // 2:],
        face_crop[h // 4:3 * h // 4, w // 3:2 * w // 3],
        face_crop[2 * h // 3:, :],
    ]

    features = []
    for zone in zones:
        if zone.size == 0:
            features.extend([0.0, 0.0, 0.0])
            continue
        mean_response = zone.mean(axis=(0, 1)).astype(np.float32)
        flash_f = flash_color.astype(np.float32) + 1e-6
        features.extend((mean_response / flash_f).tolist())

    return np.array(features, dtype=np.float32)


def _analyze_consistency(all_features: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
    """Analyze reflection consistency across flash patterns.

    Real face: high inter-zone variance, stable ratios, proportional response.
    Flat spoof: uniform zones, no/wrong response to flash changes.
    """
    if len(all_features) < 3:
        return 0.0, {"reason": "too_few_frames"}

    features = np.array(all_features)
    zone_means = features.reshape(len(features), 5, 3).mean(axis=2)

    # Zone variance (3D geometry → different angles per zone)
    inter_zone_var = zone_means.var(axis=1).mean()
    zone_score = min(1.0, inter_zone_var / 0.05)

    # Temporal consistency (same face shape → stable zone ratios across flashes)
    zone_ratios = zone_means / (zone_means.mean(axis=1, keepdims=True) + 1e-6)
    ratio_stability = 1.0 - min(1.0, zone_ratios.std(axis=0).mean() / 0.3)

    # Flash response (brightness changes with flash color)
    frame_brightness = features.mean(axis=1)
    brightness_var = frame_brightness.std() / (frame_brightness.mean() + 1e-6)
    response_score = min(1.0, brightness_var / 0.15)

    confidence = 0.4 * zone_score + 0.3 * ratio_stability + 0.3 * response_score

    details = {
        "zone_variance_score": round(zone_score, 4),
        "temporal_consistency": round(ratio_stability, 4),
        "flash_response_score": round(response_score, 4),
    }
    return float(confidence), details


class FaceFlashingChallenge:
    """Active liveness via screen light reflection analysis.

    Flashes random colors on screen, captures face reflections, analyzes
    consistency to distinguish real faces from photos/screens.

    Usage:
        challenge = FaceFlashingChallenge()
        result = challenge.run(camera_index=0)
        # {"is_live": True, "confidence": 0.94, ...}
    """

    def __init__(self, n_flashes: int = DEFAULT_N_FLASHES,
                 flash_duration: float = DEFAULT_FLASH_DURATION,
                 capture_delay: float = DEFAULT_CAPTURE_DELAY,
                 threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
                 window_name: str = "FaceFlash Liveness"):
        self.n_flashes = n_flashes
        self.flash_duration = flash_duration
        self.capture_delay = capture_delay
        self.threshold = threshold
        self.window_name = window_name

    def run(self, camera_index: int = 0,
            screen_size: Tuple[int, int] = (640, 480)) -> Dict[str, Any]:
        """Run the face flashing liveness challenge.

        Opens camera, displays flash patterns, captures reflections,
        returns liveness verdict.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required: pip install opencv-python")

        start = time.perf_counter()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        seed = int(time.time() * 1000) % (2 ** 31)
        patterns = _generate_flash_patterns(self.n_flashes, seed=seed)

        # Let camera stabilize
        time.sleep(0.2)
        ret, baseline = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Cannot read from camera")

        baseline_rgb = cv2.cvtColor(baseline, cv2.COLOR_BGR2RGB)
        if _extract_face_region(baseline_rgb) is None:
            cap.release()
            return {"is_live": False, "confidence": 0.0,
                    "reason": "no_face_detected",
                    "time_ms": round((time.perf_counter() - start) * 1000, 1)}

        # Flash and capture
        all_features = []
        frames_captured = 0

        for color in patterns:
            flash_img = np.full((screen_size[1], screen_size[0], 3),
                                color[::-1], dtype=np.uint8)
            cv2.imshow(self.window_name, flash_img)
            cv2.waitKey(int(self.flash_duration * 1000))
            time.sleep(self.capture_delay)

            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = _extract_face_region(frame_rgb)
            if face is None:
                continue

            features = _compute_reflection_features(face, color)
            all_features.append(features)
            frames_captured += 1

        cap.release()
        cv2.destroyWindow(self.window_name)

        if frames_captured < 3:
            return {"is_live": False, "confidence": 0.0,
                    "reason": "insufficient_captures",
                    "n_frames_captured": frames_captured,
                    "time_ms": round((time.perf_counter() - start) * 1000, 1)}

        confidence, details = _analyze_consistency(all_features)
        return {
            "is_live": confidence >= self.threshold,
            "confidence": round(confidence, 4),
            "n_frames_captured": frames_captured,
            "details": details,
            "time_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    def run_headless(self, frames: List[np.ndarray],
                     flash_colors: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze pre-captured frames (for testing without live camera)."""
        start = time.perf_counter()
        all_features = [_compute_reflection_features(f, c)
                        for f, c in zip(frames, flash_colors)]

        if len(all_features) < 3:
            return {"is_live": False, "confidence": 0.0,
                    "reason": "insufficient_frames",
                    "time_ms": round((time.perf_counter() - start) * 1000, 1)}

        confidence, details = _analyze_consistency(all_features)
        return {
            "is_live": confidence >= self.threshold,
            "confidence": round(confidence, 4),
            "n_frames_captured": len(all_features),
            "details": details,
            "time_ms": round((time.perf_counter() - start) * 1000, 1),
        }
