"""Load the velocity diffusion model and humanize a MIDI byte string."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from corrected_align import write_excerpt  # noqa: E402
from train_transformer import get_device  # noqa: E402
from velocity_models import VelocityDiffusion  # noqa: E402
from velocity_run import (  # noqa: E402
    as_midi_velocities,
    deadpan_from_midi,
    generate_diffusion,
)

MAX_BYTES = 512_000
MAX_NOTES = 4_000
ALLOWED_SUFFIX = {".mid", ".midi"}
MIDI_MAGIC = b"MThd"

_model = None
_mu = 0.0
_sigma = 1.0
_device = None


class HumanizeError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _checkpoint_path() -> Path:
    env = os.environ.get("HUMANIZE_CHECKPOINT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "checkpoint.pt"


def load_model() -> None:
    global _model, _mu, _sigma, _device
    if _model is not None:
        return
    path = _checkpoint_path()
    if not path.is_file():
        raise RuntimeError(f"checkpoint missing: {path}")
    _device = get_device()
    ckpt = torch.load(path, map_location=_device, weights_only=False)
    _model = VelocityDiffusion().to(_device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()
    _mu = float(ckpt["mu_train"])
    _sigma = float(ckpt["sigma_train"])


def validate_upload(filename: str | None, data: bytes) -> None:
    if not data:
        raise HumanizeError("Empty upload.")
    if len(data) > MAX_BYTES:
        raise HumanizeError(
            f"File is too large ({len(data) // 1024} KB). Maximum is {MAX_BYTES // 1024} KB."
        )
    if not data.startswith(MIDI_MAGIC):
        raise HumanizeError("Not a Standard MIDI File (expected MThd header).")
    suffix = Path(filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_SUFFIX:
        raise HumanizeError("Only .mid and .midi files are accepted.")


@torch.no_grad()
def humanize_bytes(data: bytes, *, seed: int = 2, filename: str | None = None) -> tuple[bytes, dict]:
    load_model()
    validate_upload(filename or "upload.mid", data)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.mid"
        dst = Path(tmp) / "out.mid"
        src.write_bytes(data)
        try:
            notes, arrays = deadpan_from_midi(src)
        except Exception as exc:
            raise HumanizeError(f"Could not read MIDI: {exc}") from exc
        n = len(notes)
        if n == 0:
            raise HumanizeError("The file has no piano notes (drum tracks are ignored).")
        if n > MAX_NOTES:
            raise HumanizeError(
                f"Too many notes ({n}). Maximum is {MAX_NOTES}. Try a shorter excerpt."
            )
        raw, seconds = generate_diffusion(_model, arrays, _mu, _sigma, int(seed), _device)
        full = np.full(n, 80, dtype=np.float64)
        full[:] = as_midi_velocities(raw)
        write_excerpt(src, dst, notes, full.astype(np.int64), 0, n)
        out = dst.read_bytes()
    vel = full.astype(np.int64)
    meta = {
        "notes": n,
        "seed": int(seed),
        "seconds": round(float(seconds), 2),
        "velocity_mean": round(float(vel.mean()), 1),
        "velocity_std": round(float(vel.std()), 1),
    }
    return out, meta
