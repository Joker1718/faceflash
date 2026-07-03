"""Environment diagnostics for FaceFlash installs."""

from __future__ import annotations

import importlib
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Optional


def _check_import(module: str, package: Optional[str] = None) -> dict:
    package = package or module
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", None)
        return {"ok": True, "module": module, "version": version}
    except Exception as e:
        return {
            "ok": False,
            "module": module,
            "error": f"{type(e).__name__}: {e}",
            "fix": f"pip install {package}",
        }


def _check_onnxruntime() -> dict:
    out = _check_import("onnxruntime", "onnxruntime")
    if not out["ok"]:
        out["fix"] = "pip install faceflash[cpu]"
        return out
    import onnxruntime as ort
    out["providers"] = ort.get_available_providers()
    try:
        from faceflash.runtime import select_providers
        out["selected_embedder"] = select_providers(dynamic_shapes=False)
        out["selected_detector"] = select_providers(dynamic_shapes=True)
    except Exception as e:
        out["provider_error"] = f"{type(e).__name__}: {e}"
    return out


def _check_rust_backend() -> dict:
    from faceflash.pca_quantize import backend_info
    import faceflash.index as index_module
    return {
        "ok": bool(index_module._HAS_RUST),
        "backend": backend_info(),
        "fix": None if index_module._HAS_RUST
        else "Install a wheel with the Rust extension, or run `maturin develop`.",
    }


def _check_model_cache() -> dict:
    from faceflash.align import DET_500M_PATH, DET_PATH
    from faceflash.embed import PROFILES, MODEL_DIR
    from faceflash.liveness import MODEL_PATH as LIVENESS_PATH

    embed = {
        name: str(MODEL_DIR / spec["file"])
        for name, spec in PROFILES.items()
    }
    files = {
        "embedder": {k: {"path": v, "exists": Path(v).exists()}
                     for k, v in embed.items()},
        "detector": {
            "10g": {"path": str(DET_PATH), "exists": DET_PATH.exists()},
            "500m": {"path": str(DET_500M_PATH), "exists": DET_500M_PATH.exists()},
        },
        "liveness": {
            "path": str(LIVENESS_PATH),
            "exists": LIVENESS_PATH.exists(),
        },
    }
    return {
        "ok": True,
        "cache_dir": str(MODEL_DIR),
        "files": files,
        "note": "Missing models are downloaded on first use.",
    }


def _check_sqlite() -> dict:
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE t(x)")
        con.execute("INSERT INTO t VALUES (1)")
        value = con.execute("SELECT x FROM t").fetchone()[0]
        con.close()
        return {"ok": value == 1, "version": sqlite3.sqlite_version}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _check_camera(camera: Optional[int]) -> dict:
    if camera is None:
        return {"ok": None, "checked": False,
                "note": "Pass `--camera 0` to test webcam access."}
    try:
        import cv2
        cap = cv2.VideoCapture(int(camera))
        ok = bool(cap.isOpened())
        ret, frame = cap.read() if ok else (False, None)
        cap.release()
        shape = list(frame.shape) if ret and frame is not None else None
        return {"ok": ok and ret, "checked": True, "camera": int(camera),
                "frame_shape": shape,
                "fix": None if ok and ret else
                "Check camera permissions, close other apps using the camera, or try --camera 1."}
    except Exception as e:
        return {"ok": False, "checked": True, "camera": camera,
                "error": f"{type(e).__name__}: {e}",
                "fix": "pip install opencv-python"}


def run_doctor(camera: Optional[int] = None) -> dict:
    """Return a JSON-serializable diagnostic report."""
    checks = {
        "python": {
            "ok": sys.version_info >= (3, 10),
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "fix": None if sys.version_info >= (3, 10)
            else "Use Python 3.10 or newer.",
        },
        "numpy": _check_import("numpy", "numpy"),
        "pillow": _check_import("PIL", "Pillow"),
        "opencv": _check_import("cv2", "opencv-python"),
        "onnxruntime": _check_onnxruntime(),
        "sqlite": _check_sqlite(),
        "rust_backend": _check_rust_backend(),
        "model_cache": _check_model_cache(),
        "camera": _check_camera(camera),
        "browser": {
            "ok": True,
            "note": "Browser camera works best on localhost or HTTPS. "
                    "Use Chrome/Edge first; Safari may require extra permissions.",
        },
    }
    blocking = [name for name, c in checks.items()
                if c.get("ok") is False and name not in {"camera"}]
    warnings = [name for name, c in checks.items()
                if c.get("ok") is False and name in {"camera", "rust_backend"}]
    return {
        "ok": not blocking,
        "blocking": blocking,
        "warnings": warnings,
        "checks": checks,
    }


def format_doctor(report: dict) -> str:
    """Human-readable doctor output."""
    lines = []
    status = "OK" if report["ok"] else "NEEDS ATTENTION"
    lines.append(f"FaceFlash doctor: {status}")
    for name, check in report["checks"].items():
        ok = check.get("ok")
        icon = "OK" if ok is True else ("SKIP" if ok is None else "FAIL")
        detail = check.get("version") or check.get("backend") or check.get("note") or ""
        lines.append(f"- {name}: {icon}" + (f" ({detail})" if detail else ""))
        if ok is False and check.get("fix"):
            lines.append(f"  fix: {check['fix']}")
        if name == "onnxruntime" and check.get("providers"):
            lines.append(f"  providers: {', '.join(check['providers'])}")
            lines.append(f"  selected detector: {', '.join(check.get('selected_detector', []))}")
        if name == "model_cache":
            missing = []
            for group_name, group in check.get("files", {}).items():
                if isinstance(group, dict) and "exists" in group:
                    group = {group_name: group}
                for label, item in getattr(group, "items", lambda: [])():
                    if not isinstance(item, dict):
                        continue
                    if not item.get("exists"):
                        missing.append(label)
            if missing:
                lines.append(f"  missing cached models: {', '.join(missing)}")
    if report.get("blocking"):
        lines.append("Blocking checks: " + ", ".join(report["blocking"]))
    return "\n".join(lines)
