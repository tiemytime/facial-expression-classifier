"""Contract test for the pretrained expression classifier wrapper.

This checks the plumbing (a real forward pass through the pinned model
returns a well-formed result) rather than correctness on a real face - we
deliberately don't commit a real face photo to this repo (see README.md
"Known limitations"), so there is no labeled fixture to assert a specific
emotion against here. The classifier's actual accuracy is inherited from
its upstream model card, not something this test suite re-validates.
"""
import numpy as np
import pytest

from app.models.expression_classifier import ExpressionClassifier

KNOWN_LABELS = {"sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"}


@pytest.fixture(scope="module")
def classifier():
    return ExpressionClassifier()


def test_predict_returns_a_known_label_with_valid_confidence(classifier):
    fake_face = np.random.default_rng(seed=0).integers(0, 255, size=(224, 224, 3), dtype=np.uint8)

    result = classifier.predict(fake_face)

    assert result.label in KNOWN_LABELS
    assert 0.0 <= result.confidence <= 1.0
    assert result.inference_time_ms >= 0
