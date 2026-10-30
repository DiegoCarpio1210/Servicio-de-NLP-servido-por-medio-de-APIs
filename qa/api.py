"""One model, one worker; loading never claims readiness prematurely."""
import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=5000)
    question: str = Field(min_length=1, max_length=500)


def create_app(loader=None):
    token = os.getenv("QA_API_TOKEN", "")
    team = os.getenv("TEAM_ID", "Pimientero")
    def default_loader():
        from qa.inference import Engine
        if not token or not team or any(c.isspace() for c in team):
            raise ValueError("Set QA_API_TOKEN and a registered TEAM_ID without spaces")
        engine = Engine(os.getenv("MODEL_DIR", "models/quality40k"))
        engine.predict("Warmup context.", "What is this?")
        return engine

    @asynccontextmanager
    async def lifespan(app):
        async def load():
            try:
                app.state.engine = await asyncio.to_thread(loader or default_loader)
            except Exception:
                logging.error("Model initialization failed; check configuration and local checkpoint files")
        task = asyncio.create_task(load())
        yield
        await task

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.engine = None

    @app.middleware("http")
    async def guard(request, call_next):
        if request.url.path in ("/info", "/predict"):
            supplied = request.headers.get("authorization", "")
            if not token or not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
                return JSONResponse({"error": "unauthorized"}, 401, headers={"WWW-Authenticate": "Bearer"})
        try:
            return await call_next(request)
        except Exception:
            return JSONResponse({"error": "internal_error"}, 500)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        # Never echo request values, headers or secrets in validation details.
        return JSONResponse({"error": "bad_request", "detail": "Invalid JSON or request fields"}, 422)

    @app.get("/health")
    def health():
        ready = app.state.engine is not None
        return JSONResponse({"status": "ok" if ready else "loading"}, 200 if ready else 503)

    @app.get("/info")
    def info():
        engine = app.state.engine
        if engine is None:
            return JSONResponse({"error": "internal_error"}, 500)
        return {"team": team, "model": engine.metadata["base_model"], "params": engine.params,
                "max_seq_length": engine.max_length, "doc_stride": engine.stride,
                "checkpoint_sha256": getattr(engine,"weights_sha256",None),
                "training_steps": engine.metadata.get("selected_optimizer_steps",engine.metadata.get("optimizer_steps")),
                "training_run_steps": engine.metadata.get("optimizer_steps"),
                "no_answer_threshold": getattr(engine,"threshold",None),
                "extraction_policy": getattr(engine,"policy",None),
                "calibration_objective": getattr(engine,"calibration",{}).get("objective","balanced_f1")}

    @app.post("/predict")
    def predict(body: PredictRequest):
        if app.state.engine is None:
            return JSONResponse({"error": "internal_error"}, 500)
        return {"id": body.id, **app.state.engine.predict(body.context, body.question)}

    return app


app = create_app()
