"""Liveness regressions that do not need the ONNX model file."""

import numpy as np

from faceflash.liveness import LivenessDetector, _crop_with_scale


class _FakeSession:
    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype=np.float32)

    def run(self, *_args, **_kwargs):
        return [self.logits]


def _detector_with_logits(logits):
    det = LivenessDetector.__new__(LivenessDetector)
    det.session = _FakeSession(logits)
    det.input_name = "input"
    return det


def test_liveness_class_one_is_live():
    det = _detector_with_logits([0.0, 6.0, 0.0])
    score, label = det.check(np.zeros((80, 80, 3), dtype=np.uint8))

    assert label == "live"
    assert score > 0.99


def test_liveness_non_live_classes_are_spoof():
    for logits in ([6.0, 0.0, 0.0], [0.0, 0.0, 6.0]):
        det = _detector_with_logits(logits)
        score, label = det.check(np.zeros((80, 80, 3), dtype=np.uint8))
        assert label == "spoof"
        assert score < 0.01


def test_liveness_frame_crop_expands_face_box_with_context():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    crop = _crop_with_scale(frame, (100, 60, 140, 100), 2.7)

    assert crop.shape[0] > 40
    assert crop.shape[1] > 40
    assert crop.shape[0] <= frame.shape[0]
    assert crop.shape[1] <= frame.shape[1]


def test_liveness_frame_crop_clamps_near_image_edge():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = _crop_with_scale(frame, (0, 0, 30, 30), 2.7)

    assert crop.size > 0
    assert crop.shape[0] <= frame.shape[0]
    assert crop.shape[1] <= frame.shape[1]
