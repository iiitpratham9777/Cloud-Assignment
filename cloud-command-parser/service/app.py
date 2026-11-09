"""Command-parser inference service.

Two ways to get the model (see .env.example):
  * local mode (default): load a model folder from MODEL_DIR (default model_out/v1);
    no Docker, Kubernetes or object storage needed.
  * cluster mode: if MODEL_URI is set (s3://bucket/prefix/), download it from S3_ENDPOINT at start-up.
Other environment: MODEL_VERSION, MAX_NEW_TOKENS, MODEL_CACHE_DIR (+ FAULT_INJECT for testing).
"""
import os
import socket
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import checks, settings
from .validator import InvalidModelOutput, validate_output

settings.load_dotenv()  # optional .env in the repository root; real environment variables win

app = FastAPI(title="command-parser")
state = {"generator": None, "ready": False, "error": None, "detail": None}


class ParseRequest(BaseModel):
    text: str


def _headers():
    return {
        "X-Model-Version": os.environ.get("MODEL_VERSION", "unknown"),
        # Kubernetes sets HOSTNAME = Pod name; Windows does not set it, so fall back to the host name.
        "X-Served-By": os.environ.get("HOSTNAME") or socket.gethostname(),
    }


def _load():
    try:
        from .model_loader import Generator, download_prefix

        uri = os.environ.get("MODEL_URI")
        if uri:  # cluster mode: credentials come from AWS_* variables / the Kubernetes Secret
            if not os.environ.get("S3_ENDPOINT"):
                raise checks.MissingArtifactError("MODEL_URI is set but S3_ENDPOINT is not.")
            path = download_prefix(uri, os.environ["S3_ENDPOINT"], settings.model_cache_dir())
        else:  # local mode
            path = settings.model_dir()
            checks.require_model_dir(path)
        state["generator"] = Generator(path, int(os.environ.get("MAX_NEW_TOKENS", "64")))
        state["ready"] = True
        print("model loaded; service ready", flush=True)
    except checks.MissingArtifactError as exc:  # curated, user-facing message (no secrets)
        state["error"], state["detail"] = repr(exc), str(exc)
        print(f"model not available:\n{exc}", flush=True)
    except Exception as exc:  # keep the process alive so /healthz stays 200 and logs show why
        state["error"], state["detail"] = repr(exc), "model load failed; see the server log"
        print(f"model load failed: {exc!r}", flush=True)


@app.on_event("startup")
def startup():
    if os.environ.get("SKIP_MODEL_LOAD") != "1":
        threading.Thread(target=_load, daemon=True).start()


@app.get("/healthz")
def healthz():
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    if state["ready"]:
        return {"status": "ready"}
    body = {"status": "loading"}
    if state["detail"]:  # loading failed: tell the developer why instead of waiting forever
        body["detail"] = state["detail"]
    return JSONResponse(body, status_code=503)


@app.post("/parse")
def parse(req: ParseRequest):
    if not state["ready"]:
        return JSONResponse({"error": "model_not_ready"}, status_code=503, headers=_headers())
    raw = state["generator"](req.text)
    if os.environ.get("FAULT_INJECT") == "1":
        raw = "not json"
    try:
        result = validate_output(raw)
    except InvalidModelOutput:
        return JSONResponse({"error": "invalid_model_output"}, status_code=500, headers=_headers())
    return JSONResponse(result, status_code=200, headers=_headers())
