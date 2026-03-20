"""
Tests for Heart Disease Prediction API
Run: pytest tests/ -v
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_PATIENT = {
    "age": 54, "sex": 1, "cp": 0, "trestbps": 122, "chol": 286,
    "fbs": 0, "restecg": 0, "thalach": 116, "exang": 1,
    "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 2,
}

FEATURES = list(SAMPLE_PATIENT.keys())

MOCK_METRICS = {
    "accuracy": 0.84, "precision": 0.81,
    "recall": 0.79, "f1": 0.80, "roc_auc": 0.88,
}

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """Fake sklearn pipeline that returns controllable predictions."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([1])
    pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])
    pipeline.named_steps = {"clf": MagicMock(__class__=MagicMock(__name__="RandomForestClassifier"))}
    return pipeline


@pytest.fixture
def app_client(mock_pipeline, tmp_path):
    """TestClient with model files mocked — no real model.pkl needed."""
    # Write temporary features.json
    feat_file = tmp_path / "features.json"
    feat_file.write_text(json.dumps({"features": FEATURES, "metrics": MOCK_METRICS}))

    model_file = tmp_path / "model.pkl"
    model_file.write_bytes(b"fake")  # content doesn't matter; load is mocked

    import main
    main.MODEL.clear()
    main.MODEL["pipeline"] = mock_pipeline
    main.MODEL["features"]  = FEATURES
    main.MODEL["metrics"]   = MOCK_METRICS

    from fastapi.testclient import TestClient
    with TestClient(main.app, raise_server_exceptions=True) as client:
        yield client

    main.MODEL.clear()

# ── Tests: root ───────────────────────────────────────────────────────────────

def test_root(app_client):
    r = app_client.get("/")
    assert r.status_code == 200
    assert "Heart Disease" in r.json()["message"]

# ── Tests: health ─────────────────────────────────────────────────────────────

def test_health_ok(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["features"] == len(FEATURES)
    assert "accuracy" in body["metrics"]

# ── Tests: predict ────────────────────────────────────────────────────────────

def test_predict_disease(app_client):
    r = app_client.post("/predict", json=SAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1
    assert body["label"] == "Heart disease detected"
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["confidence"] in ("High", "Medium", "Low")


def test_predict_no_disease(app_client, mock_pipeline):
    mock_pipeline.predict.return_value = np.array([0])
    mock_pipeline.predict_proba.return_value = np.array([[0.85, 0.15]])

    r = app_client.post("/predict", json=SAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 0
    assert body["label"] == "No heart disease detected"
    assert body["confidence"] == "High"


def test_predict_missing_field(app_client):
    bad = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
    r = app_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_out_of_range(app_client):
    bad = {**SAMPLE_PATIENT, "age": 999}
    r = app_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_wrong_type(app_client):
    bad = {**SAMPLE_PATIENT, "sex": "male"}
    r = app_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_latency_header(app_client):
    r = app_client.post("/predict", json=SAMPLE_PATIENT)
    assert "x-latency-ms" in r.headers


# ── Tests: confidence bands ───────────────────────────────────────────────────

@pytest.mark.parametrize("proba,expected", [
    (0.9,  "High"),
    (0.1,  "High"),
    (0.65, "Medium"),
    (0.35, "Medium"),
    (0.52, "Low"),
])
def test_confidence_bands(app_client, mock_pipeline, proba, expected):
    mock_pipeline.predict_proba.return_value = np.array([[1 - proba, proba]])
    mock_pipeline.predict.return_value = np.array([int(proba >= 0.5)])
    r = app_client.post("/predict", json=SAMPLE_PATIENT)
    assert r.json()["confidence"] == expected
