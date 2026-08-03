"""Private HTTP API for MorphoMNIST counterfactual generation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections import defaultdict, deque
import logging
import os
import threading
import time
from typing import Annotated

from runtime import configure_backend

# This must execute before importing service.py, which imports JAX.
configure_backend("cpu")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .service import CounterfactualModels, ModelSettings, RequestValidationError, ServiceUnavailableError


logger = logging.getLogger(__name__)


class ClientRateLimiter:
    """Fixed-window request limiter for the unauthenticated demo API."""

    def __init__(self, *, limit: int, window_seconds: float):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client_id: str) -> bool:
        now = time.monotonic()
        with self._lock:
            entries = self._requests[client_id]
            cutoff = now - self.window_seconds
            while entries and entries[0] <= cutoff:
                entries.popleft()
            if len(entries) >= self.limit:
                return False
            entries.append(now)
            return True


def _client_id(request: Request) -> str:
    """Use Cloud Run's forwarded client address, falling back to the socket peer."""
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(
    registry: CounterfactualModels | None = None,
    rate_limiter: ClientRateLimiter | None = None,
) -> FastAPI:
    models = registry or CounterfactualModels(ModelSettings.from_environment())
    limiter = rate_limiter or ClientRateLimiter(
        limit=int(os.getenv("RATE_LIMIT_REQUESTS", "12")),
        window_seconds=float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if registry is None:
            try:
                models.load()
            except Exception:
                # /readyz exposes the startup error without making liveness fail.
                logger.exception("Counterfactual model startup failed")
        yield

    app = FastAPI(title="Causal-GenX Counterfactual API", version="1.0.0", lifespan=lifespan)
    app.state.models = models
    allowed_origins = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def limit_inference_requests(request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith("/v1/"):
            if not limiter.allow(_client_id(request)):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded; retry in one minute"},
                    headers={"Retry-After": str(int(limiter.window_seconds))},
                )
        return await call_next(request)

    async def image_payload(image: UploadFile | None, *, required: bool = True) -> bytes | None:
        if image is None:
            if required:
                raise HTTPException(status_code=422, detail="image is required")
            return None
        if image.content_type not in {"image/png", "image/jpeg"}:
            raise HTTPException(status_code=422, detail="image must be PNG or JPEG")
        payload = await image.read()
        if not payload or len(payload) > 1_000_000:
            raise HTTPException(status_code=422, detail="image must be non-empty and at most 1 MB")
        return payload

    def execute(action):
        try:
            return action()
        except RequestValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ServiceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/livez")
    def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        if not models.ready:
            raise HTTPException(status_code=503, detail=models.error or "models are loading")
        return {"status": "ready"}

    @app.post("/v1/counterfactual")
    async def counterfactual(
        image: Annotated[UploadFile, File(...)],
        intervention_name: Annotated[str, Form(...)],
        intervention_value: Annotated[float, Form(...)],
        seed: Annotated[int | None, Form()] = None,
    ) -> dict:
        payload = await image_payload(image)
        return execute(lambda: models.generate(payload, intervention_name, intervention_value, seed))

    @app.post("/v1/generate")
    def generate(
        digit: Annotated[int, Form(...)],
        thickness: Annotated[float, Form(...)],
        intensity: Annotated[float, Form(...)],
        style_seed: Annotated[int | None, Form()] = None,
    ) -> dict:
        return execute(lambda: models.generate_from_sliders(digit, thickness, intensity, style_seed))

    @app.post("/v1/predict-parents")
    async def predict_parents(image: Annotated[UploadFile, File(...)]) -> dict:
        payload = await image_payload(image)
        return execute(lambda: models.predict_parents(payload))

    @app.post("/v1/linked-intensity")
    async def linked_intensity(
        thickness: Annotated[float, Form(...)],
        image: Annotated[UploadFile | None, File()] = None,
    ) -> dict:
        payload = await image_payload(image, required=False)
        return execute(lambda: models.linked_intensity(thickness, payload))

    @app.post("/v1/render-counterfactual")
    async def render_counterfactual(
        image: Annotated[UploadFile, File(...)],
        digit: Annotated[int, Form(...)],
        thickness: Annotated[float, Form(...)],
        intensity: Annotated[float, Form(...)],
        seed: Annotated[int | None, Form()] = None,
    ) -> dict:
        payload = await image_payload(image)
        return execute(lambda: models.render_counterfactual(payload, digit, thickness, intensity, seed))

    return app


app = create_app()
