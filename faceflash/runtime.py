"""Shared ONNX Runtime setup — one place that decides execution providers.

FaceFlash is CPU-first: the default is the plain CPU provider, and everything
in the benchmarks was measured there. But if a faster provider is present on
the machine, use it automatically — none of these are dependencies, they're
opt-in extras you install for your hardware:

  GPU:    pip install faceflash[gpu]      -> onnxruntime-gpu   (CUDA)
  Intel:  pip install faceflash[intel]    -> onnxruntime-openvino
  Apple:  (CoreML ships with onnxruntime on macOS)
  ARM:    XNNPACK ships with some onnxruntime builds

Set FACEFLASH_PROVIDERS to force an ordered list (e.g.
"CPUExecutionProvider") to pin behavior for a benchmark or a flaky provider.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Preference order when auto-detecting. First available wins (CPU always the
# guaranteed tail). CoreML/XNNPACK help small CNNs; OpenVINO helps on Intel;
# CUDA when a GPU is present.
_PREFERENCE = [
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "XnnpackExecutionProvider",
    "CPUExecutionProvider",
]

# Providers that specialize on STATIC input shapes. They accelerate the
# fixed-size embedder (always 1x3x112x112) but MISCOMPILE the detector, which
# runs at variable input sizes (320/480/640) and dynamic output shapes —
# CoreML raises a rank-mismatch at runtime. Only CUDA and CPU handle the
# detector's dynamic shapes reliably.
_STATIC_SHAPE_ONLY = {
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "XnnpackExecutionProvider",
}


def select_providers(dynamic_shapes: bool = False):
    """Return an ordered provider list ending in CPUExecutionProvider.

    dynamic_shapes=True (the detector) excludes static-shape-only providers,
    which crash on variable input sizes.

    Honors FACEFLASH_PROVIDERS (comma-separated) if set; otherwise picks the
    single best available provider with CPU as fallback.
    """
    import onnxruntime as ort
    available = set(ort.get_available_providers())

    forced = os.environ.get("FACEFLASH_PROVIDERS")
    if forced:
        want = [p.strip() for p in forced.split(",") if p.strip()]
        chosen = [p for p in want
                  if p in available
                  and not (dynamic_shapes and p in _STATIC_SHAPE_ONLY)]
        if "CPUExecutionProvider" not in chosen:
            chosen.append("CPUExecutionProvider")
        return chosen

    for p in _PREFERENCE:
        if p == "CPUExecutionProvider":
            break
        if dynamic_shapes and p in _STATIC_SHAPE_ONLY:
            continue
        if p in available:
            return [p, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session(model_path, quiet: bool = True, dynamic_shapes: bool = False):
    """Build an InferenceSession with the selected providers and sane options.

    dynamic_shapes=True (detector) restricts to providers that handle variable
    input sizes (CUDA/CPU). Graph optimization defaults to ALL (already ORT's
    default); CPU thread count is left at ORT's default — measured optimal on
    a 12-core CPU, manual thread caps were slower. quiet silences per-inference
    shape warnings (the detector runs at non-metadata input sizes).
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if quiet:
        so.log_severity_level = 3
    providers = select_providers(dynamic_shapes=dynamic_shapes)
    try:
        sess = ort.InferenceSession(str(model_path), sess_options=so,
                                    providers=providers)
    except Exception as e:
        # A provider can be installed but fail to init (missing driver, bad
        # OpenVINO device). Fall back to pure CPU rather than crash.
        if providers != ["CPUExecutionProvider"]:
            logger.warning("provider init failed (%s) — falling back to CPU", e)
            sess = ort.InferenceSession(str(model_path), sess_options=so,
                                        providers=["CPUExecutionProvider"])
        else:
            raise
    active = sess.get_providers()[0]
    if active != "CPUExecutionProvider":
        logger.info("onnxruntime using %s", active)
    return sess
