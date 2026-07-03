"""High-level FaceFlash API. Wraps detection, embedding, and search."""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
import time

from faceflash.detect import (load_image, detect_and_align,
                              detect_and_align_info, detect_faces_boxes)
from faceflash.embed import FaceEmbedder
from faceflash.index import FaceIndex
from faceflash.attributes import FaceAttributes

logger = logging.getLogger(__name__)


class NoFaceFoundError(ValueError):
    """Raised when register() finds no face in the image (detection fell all
    the way back to a center crop). Pass allow_fallback=True to enroll the
    crop anyway — knowing it indexes whatever happens to be mid-frame."""


# Fallback FAR->threshold anchors measured on LFW (15K genuine / 200K impostor
# pairs, benchmarks/bench_far_frr.py). Used until calibrate() is run on the
# actual gallery — error rates shift with demographics and photo quality.
_LFW_FAR_CURVE = [
    # (false_accept_rate, cosine threshold), descending FAR
    (0.01,    0.169),
    (0.00363, 0.20),
    (0.001,   0.237),
    (0.00068, 0.25),
    (0.00012, 0.30),
    (0.0001,  0.308),
    (0.00002, 0.35),
]


def _threshold_from_curve(curve, max_far: float) -> float:
    """Interpolate a cosine threshold for a target FAR (log-FAR domain).
    Below the calibrated range, returns the most conservative threshold."""
    pts = sorted(curve, key=lambda p: -p[0])  # descending FAR
    if max_far >= pts[0][0]:
        return pts[0][1]
    if max_far <= pts[-1][0]:
        return pts[-1][1]
    for (f_hi, t_lo), (f_lo, t_hi) in zip(pts, pts[1:]):
        if f_lo <= max_far <= f_hi:
            span = np.log10(f_hi) - np.log10(f_lo)
            w = (np.log10(f_hi) - np.log10(max_far)) / span if span else 0.0
            return float(t_lo + w * (t_hi - t_lo))
    return pts[-1][1]


class FaceFlash:
    """Face search engine. Register faces, search by image, verify pairs."""

    def __init__(self, index_path: Optional[str] = None,
                 n_bits: int = 512, n_candidates: int = 100):
        """
        index_path: a .db file opens (or creates) the write-through SQLite
            backend — every register() persists immediately, no save() needed.
            A directory path loads a legacy .npy snapshot if it exists.
        n_bits: code length. 512 = full recall, 256 = half memory / ~99% recall.
        n_candidates: how many to shortlist before float rerank. Tradeoff knob.
        """
        self._embedder: Optional[FaceEmbedder] = None
        self._attributes: Optional[FaceAttributes] = None
        self._liveness = None
        self.n_candidates = n_candidates
        if index_path and str(index_path).endswith(".db"):
            self.index = FaceIndex(n_bits=n_bits, db_path=str(index_path))
        else:
            self.index = FaceIndex(n_bits=n_bits)
            if index_path and Path(index_path).exists():
                self.index.load(index_path)

    @property
    def embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            self._embedder = FaceEmbedder()
        return self._embedder

    @property
    def attributes(self) -> FaceAttributes:
        if self._attributes is None:
            self._attributes = FaceAttributes()
        return self._attributes

    @property
    def liveness(self) -> "LivenessDetector":
        """Lazy-loaded anti-spoofing detector (~600KB model, downloads on first use)."""
        if self._liveness is None:
            from faceflash.liveness import LivenessDetector
            self._liveness = LivenessDetector()
        return self._liveness

    def register(self, name: str, image_path: str,
                 allow_fallback: bool = False,
                 check_duplicate: bool = True) -> dict:
        """Add a face to the gallery.

        Raises NoFaceFoundError if no face was detected anywhere in the image
        (enrolling a blind center crop indexes noise and silently corrupts the
        gallery). Pass allow_fallback=True to enroll the crop anyway.

        With check_duplicate (default), the new face is searched against the
        existing gallery first; a strong match under a DIFFERENT name is
        reported in the result and logged — that's how mislabeled galleries
        get caught at the door instead of at search time.
        """
        img = load_image(image_path)
        face, method = detect_and_align_info(img)
        if method == "center_crop" and not allow_fallback:
            raise NoFaceFoundError(
                f"No face detected in {image_path!r}. Enrolling the raw "
                f"center crop would index noise — fix the photo, or pass "
                f"allow_fallback=True to enroll it anyway.")

        embedding = self.embedder.embed(face)

        result = {"name": name, "registered": True, "detection": method}
        if check_duplicate and self.index.count > 0:
            hits = self.index.search(embedding, k=1,
                                     n_candidates=self.n_candidates)
            if hits and hits[0][1] >= 0.85 and hits[0][0] != name:
                other, sim = hits[0][0], hits[0][1]
                logger.warning(
                    "register(%r): this face matches existing person %r "
                    "(similarity %.2f) — possible mislabel or duplicate",
                    name, other, sim)
                result["duplicate_of"] = {"name": other,
                                          "confidence": round(float(sim), 4)}

        self.index.add(embedding, name, image_path)
        result["index_size"] = self.index.count
        return result

    def register_folder(self, folder_path: str, progress: bool = True) -> dict:
        """Bulk register. Expects folder/person_name/photo.jpg structure."""
        folder = Path(folder_path)
        registered = 0
        errors = 0

        items = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png")) + list(folder.rglob("*.jpeg"))

        if progress:
            from tqdm import tqdm
            items = tqdm(items, desc="Indexing faces")

        for img_path in items:
            # Derive name from parent folder or filename
            if img_path.parent != folder:
                name = img_path.parent.name
            else:
                name = img_path.stem.rsplit("_", 1)[0]

            try:
                self.register(name, str(img_path))
                registered += 1
            except Exception as e:
                errors += 1
                logger.warning("Failed to register %s: %s", img_path, e)

        return {"registered": registered, "errors": errors, "total": self.index.count}

    def analyze(self, image_path: str) -> dict:
        """Detect a face and return age, gender, emotion — no search needed."""
        img = load_image(image_path)
        face = detect_and_align(img)
        attrs = self.attributes.analyze(face)
        return attrs

    def check_liveness(self, image_path: str) -> dict:
        """Anti-spoofing check on the main face in an image.

        Returns {"live": bool, "liveness_score": 0-1, "label":
        "live"|"print_attack"|"replay_attack", "detection": how the face
        was found}. Single-frame passive check (MiniFASNetV2) — a first-line
        filter against printed photos and screen replays, not 3D masks.
        """
        img = load_image(image_path)
        face, method = detect_and_align_info(img)
        score, label = self.liveness.check(face)
        return {"live": label == "live",
                "liveness_score": round(float(score), 4),
                "label": label,
                "detection": method}

    def calibrate(self, n_pairs: int = 100_000, seed: int = 0) -> dict:
        """Calibrate what a cosine threshold means in false-accept rate on
        THIS gallery (impostor pairs sampled from the enrolled faces).

        After calling this, `search(..., max_far=0.001)` uses thresholds
        derived from your own population instead of the shipped LFW anchors.
        The calibration persists with the index (SQLite meta table / legacy
        metadata.json). Needs at least 2 distinct enrolled people.
        """
        sims = self.index.sample_impostor_sims(n_pairs=n_pairs, seed=seed)
        fars = [f for f in (0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)
                if f * len(sims) >= 10]     # need enough tail mass to estimate
        curve = [(f, float(np.quantile(sims, 1.0 - f))) for f in fars]
        calibration = {
            "curve": curve,
            "n_pairs": int(len(sims)),
            "source": "gallery",
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.index.set_calibration(calibration)
        return calibration

    def _threshold_for_far(self, max_far: float) -> float:
        cal = self.index.get_calibration()
        if cal:
            return _threshold_from_curve(cal["curve"], max_far)
        logger.info("max_far=%g using shipped LFW anchors — run calibrate() "
                    "to measure thresholds on your own gallery", max_far)
        return _threshold_from_curve(_LFW_FAR_CURVE, max_far)

    def search(self, image_path: str, k: int = 1, threshold: float = 0.4,
               n_candidates: Optional[int] = None,
               max_far: Optional[float] = None,
               check_liveness: bool = False,
               with_attributes: bool = False,
               age_filter: Optional[str] = None,
               min_age: Optional[int] = None,
               max_age: Optional[int] = None) -> dict:
        """Identify a face. Returns matches above threshold with timing info.

        max_far: set the decision threshold by target false-accept rate
        instead of a raw cosine value — e.g. max_far=0.001 means "at most
        1 in 1,000 impostor queries may match". Overrides `threshold`.
        Run calibrate() once to derive thresholds from your own gallery.
        """
        start = time.perf_counter()
        if max_far is not None:
            threshold = self._threshold_for_far(max_far)

        img = load_image(image_path)
        face = detect_and_align(img)

        if check_liveness:
            score, label = self.liveness.check(face)
            if label != "live":
                return {"matches": [], "filtered": f"liveness: {label}",
                        "liveness_score": round(float(score), 4),
                        "time_ms": round((time.perf_counter() - start) * 1000, 2)}

        embedding = self.embedder.embed(face)

        # Age/gender filtering on the query face
        query_attrs = None
        if with_attributes or age_filter or min_age is not None or max_age is not None:
            query_attrs = self.attributes.analyze(face)
            # Apply age filters — skip search entirely if query doesn't pass
            if age_filter == "adult" and query_attrs["age"] < 18:
                return {"matches": [], "filtered": "query is child", "attributes": query_attrs,
                        "time_ms": round((time.perf_counter() - start) * 1000, 2)}
            if age_filter == "child" and query_attrs["age"] >= 18:
                return {"matches": [], "filtered": "query is adult", "attributes": query_attrs,
                        "time_ms": round((time.perf_counter() - start) * 1000, 2)}
            if min_age is not None and query_attrs["age"] < min_age:
                return {"matches": [], "filtered": f"age {query_attrs['age']} below min_age {min_age}",
                        "attributes": query_attrs, "time_ms": round((time.perf_counter() - start) * 1000, 2)}
            if max_age is not None and query_attrs["age"] > max_age:
                return {"matches": [], "filtered": f"age {query_attrs['age']} above max_age {max_age}",
                        "attributes": query_attrs, "time_ms": round((time.perf_counter() - start) * 1000, 2)}

        embed_time = time.perf_counter() - start

        search_start = time.perf_counter()
        results = self.index.search(embedding, k=k, n_candidates=n_candidates or self.n_candidates)
        search_time = time.perf_counter() - search_start

        total_time = time.perf_counter() - start

        matches = []
        for label, similarity, idx in results:
            if similarity >= threshold:
                match_info = {
                    "name": label,
                    "confidence": round(float(similarity), 4),
                    "index": idx,
                }
                matches.append(match_info)

        result = {
            "matches": matches,
            "threshold": round(float(threshold), 4),
            "time_ms": round(total_time * 1000, 2),
            "embed_time_ms": round(embed_time * 1000, 2),
            "search_time_ms": round(search_time * 1000, 2),
            "index_size": self.index.count,
        }
        if max_far is not None:
            result["max_far"] = max_far
        if query_attrs:
            result["attributes"] = query_attrs
        return result

    def verify(self, image1: str, image2: str,
               threshold: float = 0.4,
               max_far: Optional[float] = None) -> dict:
        """Check if two images are the same person.

        max_far sets the decision threshold by false-accept rate (see
        search()); it overrides `threshold`.
        """
        start = time.perf_counter()
        if max_far is not None:
            threshold = self._threshold_for_far(max_far)

        img1 = load_image(image1)
        face1 = detect_and_align(img1)
        emb1 = self.embedder.embed(face1)

        img2 = load_image(image2)
        face2 = detect_and_align(img2)
        emb2 = self.embedder.embed(face2)

        similarity = float(np.dot(emb1, emb2))
        total_time = time.perf_counter() - start

        out = {
            "match": similarity >= threshold,
            "confidence": round(similarity, 4),
            "threshold": round(float(threshold), 4),
            "time_ms": round(total_time * 1000, 2),
        }
        if max_far is not None:
            out["max_far"] = max_far
        return out

    def search_all(self, image_path: str, k: int = 1, threshold: float = 0.4,
                   n_candidates: Optional[int] = None,
                   max_far: Optional[float] = None) -> dict:
        """Identify EVERY face in an image (group photos, camera frames).

        Returns one entry per detected face, in detection order:
        {"faces": [{"face": 0, "box": [x1, y1, x2, y2], "matches": [...]}, ...]}
        """
        start = time.perf_counter()
        if max_far is not None:
            threshold = self._threshold_for_far(max_far)

        img = load_image(image_path)
        detections = detect_faces_boxes(img)
        faces = []
        for fi, d in enumerate(detections):
            embedding = self.embedder.embed(d["crop"])
            results = self.index.search(
                embedding, k=k, n_candidates=n_candidates or self.n_candidates)
            matches = [{"name": lbl, "confidence": round(float(sim), 4), "index": idx}
                       for lbl, sim, idx in results if sim >= threshold]
            faces.append({"face": fi, "box": d["box"],
                          "det_score": d["det_score"], "matches": matches})

        out = {
            "faces": faces,
            "n_faces": len(faces),
            "threshold": round(float(threshold), 4),
            "time_ms": round((time.perf_counter() - start) * 1000, 2),
        }
        if max_far is not None:
            out["max_far"] = max_far
        return out

    def save(self, path: str):
        self.index.save(path)

    def load(self, path: str):
        self.index.load(path)

    def stats(self) -> dict:
        return self.index.stats()

    def names(self) -> list:
        return self.index.names()

    def remove(self, name: str) -> int:
        """Remove all faces for a person. Returns count removed."""
        return self.index.remove(name)

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def __repr__(self) -> str:
        return f"FaceFlash(faces={len(self.index)}, people={len(self.index.names())})"
