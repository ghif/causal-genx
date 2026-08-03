import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import ClientRateLimiter, create_app
from backend.app.service import RequestValidationError, preprocess_image


class _FakeRegistry:
    ready = True
    error = None

    def __init__(self):
        self.calls = []

    def generate(self, image_data, intervention_name, intervention_value, seed):
        self.calls.append((image_data, intervention_name, intervention_value, seed))
        return {
            "model_version": "gs://bucket/checkpoints/15795",
            "seed": 7 if seed is None else seed,
            "intervention": {"name": intervention_name, "value_physical": intervention_value},
            "factual_parents": {"normalized": {"digit": 1}, "physical": {"digit": 1}},
            "counterfactual_parents": {"normalized": {"digit": 2}, "physical": {"digit": 2}},
            "image_png_base64": "ZmFrZQ==",
            "latency_ms": 1.0,
        }

    def generate_from_sliders(self, digit, thickness, intensity, style_seed):
        self.calls.append(("generate", digit, thickness, intensity, style_seed))
        return {
            "style_seed": 7 if style_seed is None else style_seed,
            "generated_parents": {"physical": {"digit": digit, "thickness": thickness, "intensity": intensity}},
            "image_png_base64": "ZmFrZQ==",
            "latency_ms": 1.0,
        }

    def predict_parents(self, image_data):
        self.calls.append(("predict", image_data))
        return {
            "factual_parents": {"physical": {"digit": 3, "thickness": 3.5, "intensity": 160.0}},
            "latency_ms": 1.0,
        }

    def linked_intensity(self, thickness, image_data):
        self.calls.append(("linked", thickness, image_data))
        return {"intensity_physical": 150.0, "latency_ms": 1.0}

    def render_counterfactual(self, image_data, digit, thickness, intensity, seed):
        self.calls.append(("render", image_data, digit, thickness, intensity, seed))
        parents = {"physical": {"digit": digit, "thickness": thickness, "intensity": intensity}}
        return {
            "factual_parents": parents,
            "counterfactual_parents": parents,
            "seed_image_png_base64": "ZmFrZQ==",
            "image_png_base64": "ZmFrZQ==",
            "latency_ms": 1.0,
        }


def _png(size=(28, 28)):
    image = Image.fromarray(np.zeros(size, dtype=np.uint8), mode="L")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_preprocess_image_pads_morphomnist_input():
    image = np.asarray(preprocess_image(_png()))
    assert image.shape == (1, 32, 32, 1)
    assert image.dtype == np.float32


def test_preprocess_image_rejects_unsupported_size():
    with np.testing.assert_raises_regex(RequestValidationError, "28x28 or 32x32"):
        preprocess_image(_png((30, 30)))


def test_counterfactual_endpoint_forwards_multipart_request():
    registry = _FakeRegistry()
    client = TestClient(create_app(registry))
    response = client.post(
        "/v1/counterfactual",
        files={"image": ("digit.png", _png(), "image/png")},
        data={"intervention_name": "thickness", "intervention_value": "3.0", "seed": "11"},
    )
    assert response.status_code == 200
    assert response.json()["seed"] == 11
    assert registry.calls[0][1:] == ("thickness", 3.0, 11)


def test_counterfactual_endpoint_rejects_non_image_upload():
    client = TestClient(create_app(_FakeRegistry()))
    response = client.post(
        "/v1/counterfactual",
        files={"image": ("input.txt", b"not an image", "text/plain")},
        data={"intervention_name": "digit", "intervention_value": "2"},
    )
    assert response.status_code == 422


def test_readyz_reports_unready_registry():
    registry = _FakeRegistry()
    registry.ready = False
    registry.error = "checkpoint unavailable"
    client = TestClient(create_app(registry))
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["detail"] == "checkpoint unavailable"


def test_visualizer_endpoints_forward_expected_controls():
    registry = _FakeRegistry()
    client = TestClient(create_app(registry))

    generated = client.post(
        "/v1/generate",
        data={"digit": "4", "thickness": "3.0", "intensity": "170.0", "style_seed": "12"},
    )
    predicted = client.post("/v1/predict-parents", files={"image": ("digit.png", _png(), "image/png")})
    linked = client.post("/v1/linked-intensity", data={"thickness": "3.0"})
    rendered = client.post(
        "/v1/render-counterfactual",
        files={"image": ("digit.png", _png(), "image/png")},
        data={"digit": "4", "thickness": "3.0", "intensity": "170.0", "seed": "12"},
    )

    assert generated.status_code == predicted.status_code == linked.status_code == rendered.status_code == 200
    assert registry.calls[0] == ("generate", 4, 3.0, 170.0, 12)
    assert registry.calls[1][0] == "predict"
    assert registry.calls[2] == ("linked", 3.0, None)
    assert registry.calls[3][0] == "render"


def test_local_frontend_origin_is_allowed_by_cors():
    client = TestClient(create_app(_FakeRegistry()))
    response = client.options(
        "/v1/generate",
        headers={
            "Origin": "http://127.0.0.1:8000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:8000"


def test_github_pages_origin_is_allowed_when_configured(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://ghif.github.io")
    client = TestClient(create_app(_FakeRegistry()))
    response = client.options(
        "/v1/generate",
        headers={
            "Origin": "https://ghif.github.io",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://ghif.github.io"


def test_rate_limiter_rejects_excess_inference_requests():
    client = TestClient(create_app(_FakeRegistry(), rate_limiter=ClientRateLimiter(limit=1, window_seconds=60)))
    first = client.post(
        "/v1/generate",
        data={"digit": "4", "thickness": "3.0", "intensity": "170.0", "style_seed": "12"},
        headers={"x-forwarded-for": "198.51.100.1"},
    )
    second = client.post(
        "/v1/generate",
        data={"digit": "4", "thickness": "3.0", "intensity": "170.0", "style_seed": "12"},
        headers={"x-forwarded-for": "198.51.100.2"},
    )
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"


def test_default_rate_limit_is_thirty_requests_per_minute(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_REQUESTS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_WINDOW_SECONDS", raising=False)
    client = TestClient(create_app(_FakeRegistry()))

    for _ in range(30):
        response = client.post(
            "/v1/generate",
            data={"digit": "4", "thickness": "3.0", "intensity": "170.0", "style_seed": "12"},
            headers={"x-forwarded-for": "198.51.100.1"},
        )
        assert response.status_code == 200

    rejected = client.post(
        "/v1/generate",
        data={"digit": "4", "thickness": "3.0", "intensity": "170.0", "style_seed": "12"},
        headers={"x-forwarded-for": "198.51.100.1"},
    )
    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == "60"
