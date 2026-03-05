"""FastAPI app factory shared by all backend servers."""

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import FastAPI, HTTPException

from .config import DEFAULT_PARAMS
from .schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)


def create_app(model_name: str, generate_fn: Callable[[GenerateRequest], GenerateResponse]) -> FastAPI:
    """Create a FastAPI application for a given model.

    Args:
        model_name: Name of the model (e.g. "momask").
        generate_fn: Async or sync callable that accepts a ``GenerateRequest``
            and returns a ``GenerateResponse``.  The function will be called
            inside the route handler, which already runs in a thread pool for
            sync callables via FastAPI's default behaviour.

    Returns:
        A configured FastAPI application instance.
    """
    app = FastAPI(title=f"Motion API – {model_name}", version="1.0.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model=model_name)

    @app.get("/v1/models", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(
            id=model_name,
            default_params=DEFAULT_PARAMS.get(model_name, {}),
        )

    @app.post("/v1/motion/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        try:
            return generate_fn(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def make_response_id() -> str:
    return "gen_" + uuid.uuid4().hex[:12]


def now_ts() -> int:
    return int(time.time())
