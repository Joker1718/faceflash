"""Face detection. SCRFD 5-point alignment preferred, Haar cascade fallback."""
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def detect_and_align(img: np.ndarray) -> np.ndarray:
    """Largest face → 112x112 aligned crop. SCRFD if available, else Haar."""
    face, _ = detect_and_align_info(img)
    return face


def detect_and_align_info(img: np.ndarray):
    """Like detect_and_align, but also reports HOW the crop was produced.

    Returns (face_crop, method) where method is one of:
      "scrfd"       — a face was detected and 5-point aligned (trustworthy)
      "haar"        — Haar cascade found a face (no landmarks, cruder crop)
      "center_crop" — NO face was found anywhere; this is just the middle of
                      the image. Enrolling it indexes noise — callers should
                      refuse it unless explicitly told otherwise.
    """
    try:
        from faceflash.align import detect_and_align as _rf_align
        aligned = _rf_align(img)
        if aligned is not None:
            return aligned, "scrfd"
    except Exception as e:
        logger.debug("SCRFD alignment failed; falling back to Haar/center crop: %s",
                     e, exc_info=True)
    return _detect_haar_fallback_info(img)


def detect_faces(img: np.ndarray) -> list:
    """All faces in image → list of 112x112 aligned crops."""
    return [f["crop"] for f in detect_faces_boxes(img)]


def detect_faces_boxes(img: np.ndarray) -> list:
    """All faces in image, with locations.

    Returns a list of dicts, one per face:
      {"crop":   112x112 aligned RGB array,
       "box":    [x1, y1, x2, y2] in image pixel coords,
       "det_score": detector confidence (0-1; 0.0 for fallback paths),
       "method": "scrfd" | "haar" | "center_crop"}
    """
    try:
        from faceflash.align import get_detector, align_face
        det = get_detector()
        if det is not None:
            boxes, kps = det.detect(img)
            if len(boxes):
                out = []
                for i in range(len(boxes)):
                    crop = align_face(img, kps[i])
                    if crop is None:
                        continue
                    x1, y1, x2, y2 = (int(round(v)) for v in boxes[i][:4])
                    out.append({"crop": crop,
                                "box": [x1, y1, x2, y2],
                                "det_score": round(float(boxes[i][4]), 4),
                                "method": "scrfd"})
                if out:
                    return out
    except Exception as e:
        logger.debug("SCRFD face detection failed; falling back to Haar/center crop: %s",
                     e, exc_info=True)
    crop, method, box = _detect_haar_fallback_box(img)
    return [{"crop": crop, "box": box, "det_score": 0.0, "method": method}]


def _detect_haar_fallback(img: np.ndarray) -> np.ndarray:
    """Haar + center crop when SCRFD isn't available."""
    return _detect_haar_fallback_box(img)[0]


def _detect_haar_fallback_info(img: np.ndarray):
    """Haar crop (method='haar') or center crop (method='center_crop')."""
    crop, method, _ = _detect_haar_fallback_box(img)
    return crop, method


def _detect_haar_fallback_box(img: np.ndarray):
    """Returns (crop, method, [x1, y1, x2, y2])."""
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            margin = int(max(fw, fh) * 0.2)
            h, w = img.shape[:2]
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + fw + margin)
            y2 = min(h, y + fh + margin)
            return img[y1:y2, x1:x2], "haar", [int(x1), int(y1), int(x2), int(y2)]
    except Exception as e:
        # cv2 missing OR broken (partial installs lack CascadeClassifier) —
        # degrade to the center crop rather than crashing register().
        logger.debug("Haar fallback failed; using center crop: %s", e, exc_info=True)

    # Final fallback: center crop — NOT a detected face
    h, w = img.shape[:2]
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return (img[top:top + size, left:left + size], "center_crop",
            [left, top, left + size, top + size])
