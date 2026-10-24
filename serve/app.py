"""HTTP API for velocity humanization (Hugging Face Space / local)."""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from serve.inference import HumanizeError, humanize_bytes, load_model

DEFAULT_ORIGINS = [
    "https://krishivsinghal.github.io",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

extra = os.environ.get("CORS_ORIGINS", "")
ORIGINS = DEFAULT_ORIGINS + [o.strip() for o in extra.split(",") if o.strip()]

app = FastAPI(title="Velocity humanize", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm() -> None:
    load_model()


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "model": "velocity-diffusion-asap-v1"}


@app.post("/humanize")
async def humanize(file: UploadFile = File(...), seed: int = 2):
    if seed < 0 or seed > 2_147_483_647:
        raise HTTPException(status_code=400, detail="seed out of range")
    try:
        data = await file.read()
        out, meta = humanize_bytes(data, seed=seed, filename=file.filename)
    except HumanizeError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed.") from exc
    headers = {
        "X-Humanize-Notes": str(meta["notes"]),
        "X-Humanize-Seconds": str(meta["seconds"]),
        "X-Humanize-Seed": str(meta["seed"]),
    }
    return Response(
        content=out,
        media_type="audio/midi",
        headers=headers,
        status_code=200,
    )
