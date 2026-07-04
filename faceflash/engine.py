"""High-level FaceFlash API. Wraps detection, embedding, and search."""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from faceflash.liveness import LivenessDetector

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

# Same protocol, light profile (MobileFaceNet). Its score distribution sits
# lower, so the same FAR needs a different threshold (results/bench_far_frr_lfw_light.json).
_LFW_FAR_CURVE_LIGHT = [
    (0.01,    0.240),
    (0.00183, 0.30),
    (0.001,   0.320),
    (0.00038, 0.35),
    (0.0001,  0.386),
    (0.00006, 0.40),
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


_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _schedule_allows(rule: str, when) -> bool:
    """Evaluate "Mon-Fri 9:00-18:00" / "Sat,Sun 10:00-14:00" style rules."""
    try:
        parts = rule.strip().split()
        days_part = parts[0].lower()
        allowed_days = set()
        for token in days_part.split(","):
            if "-" in token:
                a, b = token.split("-")
                ia, ib = _DAYS.index(a[:3]), _DAYS.index(b[:3])
                allowed_days.update(range(ia, ib + 1))
            else:
                allowed_days.add(_DAYS.index(token[:3]))
        if when.weekday() not in allowed_days:
            return False
        if len(parts) > 1:
            start_s, end_s = parts[1].split("-")
            h1, m1 = (list(map(int, start_s.split(":"))) + [0])[:2]
            h2, m2 = (list(map(int, end_s.split(":"))) + [0])[:2]
            minutes = when.hour * 60 + when.minute
            return h1 * 60 + m1 <= minutes <= h2 * 60 + m2
        return True
    except Exception:
        logger.warning("unparseable schedule rule %r — denying", rule)
        return False


def _schedule_parseable(rule: str) -> bool:
    """True if the rule's day/time syntax is valid (independent of 'now')."""
    try:
        parts = rule.strip().split()
        for token in parts[0].lower().split(","):
            for d in (token.split("-") if "-" in token else [token]):
                _DAYS.index(d[:3])
        if len(parts) > 1:
            start_s, end_s = parts[1].split("-")
            for t in (start_s, end_s):
                h, m = (list(map(int, t.split(":"))) + [0])[:2]
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    return False
        return True
    except Exception:
        return False


def _consent_expired(expires: Optional[str]) -> bool:
    """expires may be a date (YYYY-MM-DD, valid through end of that day) or a
    full ISO datetime (YYYY-MM-DDTHH:MM:SS, valid to the second)."""
    if not expires:
        return False
    import datetime as _dt
    try:
        if "T" in expires or " " in expires:
            return _dt.datetime.now() > _dt.datetime.fromisoformat(expires)
        return _dt.date.today() > _dt.date.fromisoformat(expires)
    except ValueError:
        logger.warning("bad consent expiry %r — treating as expired", expires)
        return True


class FaceFlash:
    """Face search engine. Register faces, search by image, verify pairs."""

    def __init__(self, index_path: Optional[str] = None,
                 n_bits: int = 512, n_candidates: int = 100,
                 profile: Optional[str] = None):
        """
        index_path: a .db file opens (or creates) the write-through SQLite
            backend — every register() persists immediately, no save() needed.
            A directory path loads a legacy .npy snapshot if it exists.
        n_bits: code length. 512 = full recall, 256 = half memory / ~99% recall.
        n_candidates: how many to shortlist before float rerank. Tradeoff knob.
        profile: embedding model — "accurate" (ArcFace r50, default) or
            "light" (MobileFaceNet, ~8x faster on plain CPU, slightly lower
            accuracy). The two are incompatible embedding spaces: a gallery
            remembers which profile enrolled it, and opening it with the
            other raises. Omit to adopt the gallery's stored profile.
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

        stored = self.index.get_meta("embed_profile")
        if stored and profile and profile != stored:
            raise ValueError(
                f"This gallery was enrolled with the {stored!r} embedding "
                f"model; opening it with profile={profile!r} would compare "
                f"incompatible embeddings. Re-enroll to switch profiles.")
        self.profile = profile or stored or "accurate"
        if not stored:
            self.index.set_meta("embed_profile", self.profile)

    @property
    def embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            self._embedder = FaceEmbedder(profile=self.profile)
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
                 check_duplicate: bool = True,
                 consent: Optional[str] = None,
                 consent_expires: Optional[str] = None) -> dict:
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
        if consent is not None:
            self.index.set_person_meta(name, "consent", consent)
        if consent_expires is not None:
            self.index.set_person_meta(name, "consent_expires", consent_expires)
            result["consent_expires"] = consent_expires
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

    def set_consent(self, name: str, consent: str,
                    expires: Optional[str] = None):
        """Record consent for an enrolled person. After `expires`
        (YYYY-MM-DD) they are silently excluded from search results —
        still stored, just never matched — until re-consented."""
        self.index.set_person_meta(name, "consent", consent)
        self.index.set_person_meta(name, "consent_expires", expires)

    def set_schedule(self, name: str, allowed_hours: str):
        """Per-person access schedule, e.g. "Mon-Fri 9:00-18:00".
        Checked by search(..., check_access=True): a face can match and
        still be denied outside its hours."""
        if not _schedule_parseable(allowed_hours):
            raise ValueError(
                f"unparseable schedule {allowed_hours!r} — use e.g. "
                f"'Mon-Fri 9:00-18:00' or 'Sat,Sun 10:00-14:00'")
        self.index.set_person_meta(name, "schedule", allowed_hours)

    def _filter_matches(self, matches, within=None, check_access=False):
        """Apply consent expiry (always), scope, and schedule.

        Returns (allowed, denied). Access-denied matches are moved OUT of the
        allowed list so a caller acting on matches[0] can never open a door
        for someone outside their hours; denied entries are reported
        separately for logging/UX.
        """
        allowed, denied = [], []
        scope = set(within) if within else None
        import datetime as _dt
        now = _dt.datetime.now()
        for m in matches:
            name = m["name"]
            if _consent_expired(self.index.get_person_meta(name, "consent_expires")):
                continue                     # silently unmatchable
            if scope is not None and name not in scope:
                continue
            if check_access:
                rule = self.index.get_person_meta(name, "schedule")
                if rule and not _schedule_allows(rule, now):
                    m["access"] = "denied"
                    m["reason"] = f"outside scheduled hours ({rule})"
                    denied.append(m)
                    continue
                m["access"] = "allowed"
            allowed.append(m)
        return allowed, denied

    def find_duplicates(self, threshold: float = 0.85) -> list:
        """Different names, same face — mislabels and double enrollments."""
        return self.index.find_duplicates(threshold=threshold)

    def audit(self, name: str) -> dict:
        """Everything the system knows about one person — enrollment,
        consent state, schedule, and every persisted stream event that
        references them. The raw material for a GDPR data subject
        access request."""
        self.index._flush_pending()
        out = {"name": name,
               "enrolled_faces": sum(1 for l in self.index.labels if l == name),
               "consent": self.index.get_person_meta(name, "consent"),
               "consent_expires": self.index.get_person_meta(name, "consent_expires"),
               "consent_expired": _consent_expired(
                   self.index.get_person_meta(name, "consent_expires")),
               "schedule": self.index.get_person_meta(name, "schedule"),
               "events": []}
        db = getattr(self.index, "_db", None)
        if db is not None:
            try:
                from faceflash.events import EventStore
                store = EventStore(db, lock=getattr(self.index, "_db_lock", None))
                out["events"] = list(reversed(store.query(
                    data_value=name, limit=10_000)))
            except Exception as e:
                logger.warning("audit event query failed: %s", e)
        return out

    def check_liveness(self, image_path: str) -> dict:
        """Anti-spoofing check on the main face in an image.

        Returns {"live": bool, "liveness_score": 0-1, "label":
        "live"|"spoof"|"no_face", "detection": how the face
        was found}. Single-frame passive check (MiniFASNetV2) — a first-line
        filter against printed photos and screen replays, not 3D masks.
        """
        img = load_image(image_path)
        dets = detect_faces_boxes(img)
        scrfd = [d for d in dets if d["method"] == "scrfd"]
        if scrfd:
            d = max(scrfd, key=lambda d: (d["box"][2] - d["box"][0]) *
                    (d["box"][3] - d["box"][1]))
            score, label = self.liveness.check_frame(img, d["box"])
            method = "scrfd"
        else:
            # no real detection (only a fallback crop) — can't do a valid
            # context crop; report unknown rather than a bogus verdict
            return {"live": False, "liveness_score": 0.0, "label": "no_face",
                    "detection": dets[0]["method"] if dets else "none"}
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
        logger.info("max_far=%g using shipped LFW anchors for the %r profile — "
                    "run calibrate() to measure thresholds on your own gallery",
                    max_far, self.profile)
        curve = _LFW_FAR_CURVE_LIGHT if self.profile == "light" else _LFW_FAR_CURVE
        return _threshold_from_curve(curve, max_far)

    def search(self, image_path: str, k: int = 1, threshold: float = 0.4,
               n_candidates: Optional[int] = None,
               max_far: Optional[float] = None,
               check_liveness: bool = False,
               within: Optional[list] = None,
               check_access: bool = False,
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

        if check_liveness:
            # liveness needs the context crop from the box; get box + aligned
            # face in one detect pass, use the box for liveness, crop for embed
            dets = [d for d in detect_faces_boxes(img) if d["method"] == "scrfd"]
            if not dets:
                return {"matches": [], "filtered": "liveness: no_face",
                        "time_ms": round((time.perf_counter() - start) * 1000, 2)}
            d = max(dets, key=lambda d: (d["box"][2] - d["box"][0]) *
                    (d["box"][3] - d["box"][1]))
            face = d["crop"]
            score, label = self.liveness.check_frame(img, d["box"])
            if label != "live":
                return {"matches": [], "filtered": f"liveness: {label}",
                        "liveness_score": round(float(score), 4),
                        "time_ms": round((time.perf_counter() - start) * 1000, 2)}
        else:
            face, method = detect_and_align_info(img)
            if method == "center_crop":
                return {"matches": [], "filtered": "no_face",
                        "detection": method,
                        "threshold": round(float(threshold), 4),
                        "time_ms": round((time.perf_counter() - start) * 1000, 2),
                        "index_size": self.index.count}

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
        # scoped search / consent filtering may drop candidates — fetch deeper
        k_search = k if within is None else max(k * 10, 50)
        results = self.index.search(embedding, k=k_search,
                                    n_candidates=n_candidates or self.n_candidates)
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
        matches, denied = self._filter_matches(matches, within=within,
                                               check_access=check_access)
        matches = matches[:k]

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
        if check_access and denied:
            result["access_denied"] = denied
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
        face1, method1 = detect_and_align_info(img1)
        if method1 == "center_crop":
            return {"match": False,
                    "confidence": 0.0,
                    "threshold": round(float(threshold), 4),
                    "filtered": "no_face",
                    "image": "image1",
                    "detection": method1,
                    "time_ms": round((time.perf_counter() - start) * 1000, 2)}
        emb1 = self.embedder.embed(face1)

        img2 = load_image(image2)
        face2, method2 = detect_and_align_info(img2)
        if method2 == "center_crop":
            return {"match": False,
                    "confidence": 0.0,
                    "threshold": round(float(threshold), 4),
                    "filtered": "no_face",
                    "image": "image2",
                    "detection": method2,
                    "time_ms": round((time.perf_counter() - start) * 1000, 2)}
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
        detections = [d for d in detect_faces_boxes(img)
                      if d["method"] != "center_crop"]
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
