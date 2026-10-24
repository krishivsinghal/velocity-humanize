"""Corrected-label velocity run: diffusion and an autoregressive decoder.

One process alternates the two models on MPS. Old checkpoints are not opened
for writing. The cache is corrected-asap-v1 only.

  source venv_ml/bin/activate
  export PYTHONPATH=src
  python -u src/velocity_run.py
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import signal
import time
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from corrected_align import (
    ASAP,
    CROP_MAX,
    CROP_MIN,
    ROOT,
    VERSION,
    apply_note_velocities,
    build_cache,
    collect_notes,
    excerpt_bounds,
    group_spans,
    load_meta,
    load_perf,
    prefix_crop,
    sample_crop,
    select_panel_and_excerpts,
    spot_check,
    window_spans,
    write_excerpt,
)
from features import N_TIMING_STRUCT, assemble_timing, note_features
from train_transformer import get_device
from velocity_models import (
    TIMESTEPS,
    VelocityDiffusion,
    VelocityScoreDecoder,
    decoder_loss,
    diffusion_loss,
    masked_epsilon_mse,
    param_count,
    rollout_decoder,
    sample_diffusion,
    sample_diffusion_ddim,
)

DEMO = ROOT / "artifacts" / "velocity_demo"
CKPT = ROOT / "data" / "processed" / "velocity_corrected" / "checkpoints"
STATE_PATH = ROOT / "data" / "processed" / "velocity_corrected" / "run_state.json"
PERPETUITY = ROOT / "data" / "test" / "Perpetuity-Piano.mid"
TOTAL_STEPS = 20000
WARMUP = 250
LR = 2e-4
LR_MIN = 2e-5
BATCH = 16
TRAIN_BUDGET_SEC = 5.5 * 3600
PACK_BUDGET_SEC = 6.0 * 3600
PANEL_T = (20, 60, 100, 140, 180)
OLD_CHECKPOINTS = [
    ROOT / "data" / "processed" / "velocity_transformer.pt",
    ROOT / "data" / "processed" / "velocity_diffusion.pt",
    ROOT / "data" / "processed" / "velocity_offset_transformer.pt",
    ROOT / "data" / "processed" / "velocity_diffusion_struct.pt",
    ROOT / "data" / "processed" / "velocity_diffusion_struct_last.pt",
    ROOT / "data" / "processed" / "phrase_seq2seq.pt",
]

STOP = {"flag": False}


def _handle_stop(signum, _frame):
    STOP["flag"] = True
    print(f"signal {signum}: will save at this step", flush=True)


class WarmupCosine:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0

    def lr_at(self, step):
        if step < WARMUP:
            return LR * float(step + 1) / float(WARMUP)
        span = max(TOTAL_STEPS - WARMUP, 1)
        progress = min(max((step - WARMUP) / span, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN + (LR - LR_MIN) * cosine

    def step(self):
        lr = self.lr_at(self.step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.step_num += 1
        return lr

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, state):
        self.step_num = int(state["step_num"])


def build_plan():
    plan = [("diffusion", 500), ("encdec", 500)]
    done = {"diffusion": 500, "encdec": 500}
    while done["diffusion"] < TOTAL_STEPS or done["encdec"] < TOTAL_STEPS:
        for name in ("diffusion", "encdec"):
            take = min(2500, TOTAL_STEPS - done[name])
            if take > 0:
                plan.append((name, take))
                done[name] += take
    return plan


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng():
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def snapshot_old():
    out = {}
    for path in OLD_CHECKPOINTS:
        out[str(path)] = (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
    return out


def assert_old_unchanged(before):
    after = snapshot_old()
    if after != before:
        raise RuntimeError("an old checkpoint changed during this run")


def atomic_torch(path, payload):
    path = Path(path)
    if path.name in {p.name for p in OLD_CHECKPOINTS} and path.parent == OLD_CHECKPOINTS[0].parent:
        raise RuntimeError(f"refusing to write historical checkpoint {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=_json_default))
    os.replace(tmp, STATE_PATH)


def elapsed(state):
    return time.time() - float(state["started_unix"])


def corpus_from(meta, split):
    by_comp = defaultdict(list)
    for record in meta["performances"]:
        if record["split"] != split:
            continue
        arrays = load_perf(record)
        by_comp[(record["composer"], record["title"])].append((record, arrays))
    return by_comp


def slice_arrays(arrays, start, end):
    start, end = int(start), int(end)
    return {
        "pitch": np.asarray(arrays["pitch"][start:end], dtype=np.int64),
        "timing": np.asarray(arrays["timing"][start:end], dtype=np.float32),
        "voice": np.asarray(arrays["voice"][start:end], dtype=np.int64),
        "velocity": np.asarray(arrays["velocity"][start:end], dtype=np.float32),
        "available": np.asarray(arrays["available"][start:end], dtype=bool),
        "onset_id": np.asarray(arrays["onset_id"][start:end], dtype=np.int64),
        "start": np.asarray(arrays["start"][start:end], dtype=np.float64),
    }


def collate(items):
    width = int(items[0]["timing"].shape[-1])
    if width != N_TIMING_STRUCT:
        raise RuntimeError(f"timing width {width} != {N_TIMING_STRUCT}")
    length = max(item["pitch"].shape[0] for item in items)
    count = len(items)
    pitch = torch.zeros(count, length, dtype=torch.long)
    timing = torch.zeros(count, length, width, dtype=torch.float32)
    voice = torch.zeros(count, length, dtype=torch.long)
    velocity = torch.zeros(count, length, dtype=torch.float32)
    available = torch.zeros(count, length, dtype=torch.bool)
    pad_mask = torch.zeros(count, length, dtype=torch.bool)
    for i, item in enumerate(items):
        n = item["pitch"].shape[0]
        pitch[i, :n] = torch.from_numpy(np.clip(item["pitch"], 0, 127).astype(np.int64))
        timing[i, :n] = torch.from_numpy(item["timing"].astype(np.float32))
        voice[i, :n] = torch.from_numpy(item["voice"].astype(np.int64))
        velocity[i, :n] = torch.from_numpy(item["velocity"].astype(np.float32))
        available[i, :n] = torch.from_numpy(item["available"].astype(bool))
        pad_mask[i, :n] = True
    return {
        "pitch": pitch,
        "timing": timing,
        "voice": voice,
        "velocity": velocity,
        "available": available,
        "pad_mask": pad_mask,
    }


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def sample_batch(by_comp, size):
    comps = list(by_comp)
    items = []
    guard = 0
    while len(items) < size:
        guard += 1
        if guard > size * 50:
            raise RuntimeError("could not draw enough crops that contain a velocity target")
        perfs = by_comp[comps[int(np.random.randint(0, len(comps)))]]
        _record, arrays = perfs[int(np.random.randint(0, len(perfs)))]
        start, end = sample_crop(arrays["onset_id"])
        if end <= start:
            continue
        _assert_crop(arrays["onset_id"], start, end)
        item = slice_arrays(arrays, start, end)
        if not item["available"].any():
            continue
        items.append(item)
    return collate(items)


def _assert_crop(onset_id, start, end):
    n = len(onset_id)
    if not (0 <= start < end <= n):
        raise AssertionError("crop outside the piece")
    if end - start > CROP_MAX and n > CROP_MAX:
        raise AssertionError("crop exceeds 256")
    spans = group_spans(onset_id)
    if n <= CROP_MAX:
        if (start, end) != (0, n):
            raise AssertionError("short piece was cropped")
        return
    if not any(span[0] == start for span in spans):
        raise AssertionError("crop splits the opening chord")
    if not any(span[1] == end for span in spans):
        raise AssertionError("crop splits the closing chord")


def deadpan_from_midi(path):
    _pm, notes = collect_notes(path)
    pitch = np.array([note["pitch"] for note in notes], dtype=np.int64)
    start = np.array([note["start"] for note in notes], dtype=np.float64)
    duration = np.array([max(float(note["duration"]), 0.0) for note in notes], dtype=np.float64)
    built = note_features(pitch, start, duration, snap=False)
    timing, voice = assemble_timing(built, None)
    arrays = {
        "pitch": pitch.astype(np.int16),
        "start": start,
        "duration": duration.astype(np.float32),
        "timing": timing.astype(np.float32),
        "voice": voice.astype(np.int16),
        "onset_id": np.asarray(built["onset_id"], dtype=np.int32),
        "velocity": np.zeros(len(notes), dtype=np.float32),
        "available": np.zeros(len(notes), dtype=bool),
        "src_track": np.array([note["track"] for note in notes], dtype=np.int16),
        "src_on": np.array([note["on_msg"] for note in notes], dtype=np.int32),
        "src_off": np.array([note["off_msg"] for note in notes], dtype=np.int32),
    }
    return notes, arrays


def notes_for_excerpt(arrays):
    count = len(arrays["pitch"])
    return [
        {
            "track": int(arrays["src_track"][i]),
            "on_msg": int(arrays["src_on"][i]),
            "off_msg": int(arrays["src_off"][i]),
            "pitch": int(arrays["pitch"][i]),
            "start": float(arrays["start"][i]),
        }
        for i in range(count)
    ]


def make_models(device):
    diffusion = VelocityDiffusion().to(device)
    encdec = VelocityScoreDecoder().to(device)
    return {"diffusion": diffusion, "encdec": encdec}


def make_opt(model):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01)
    sched = WarmupCosine(opt)
    return opt, sched


def payload(model, opt, sched, kind, step, state, panel_score):
    return {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": int(step),
        "rng": capture_rng(),
        "mu_train": float(state["mu"]),
        "sigma_train": float(state["sigma"]),
        "fingerprint": state["fingerprint"],
        "panel_score": panel_score,
        "kind": kind,
        "config": {
            "version": VERSION,
            "d_model": 96,
            "heads": 4,
            "swiglu": 384,
            "dropout": 0.1,
            "encoder_blocks": 4,
            "decoder_blocks": 2 if kind == "encdec" else 0,
            "diffusion_blocks": 4 if kind == "diffusion" else 0,
            "timesteps": TIMESTEPS,
            "lr": LR,
            "lr_min": LR_MIN,
            "warmup": WARMUP,
            "total_steps": TOTAL_STEPS,
            "batch": BATCH,
            "microbatch": int(state["microbatch"]),
            "seed": 42,
            "parameters": param_count(model),
        },
    }


def save_checkpoint(kind, tag, model, opt, sched, state, panel_score=None):
    path = CKPT / kind / f"{tag}.pt"
    atomic_torch(path, payload(model, opt, sched, kind, state["steps"][kind], state, panel_score))
    print(f"saved {path} step {state['steps'][kind]}", flush=True)
    return path


def load_checkpoint(kind, tag, model, opt, sched, device, restore_random=False):
    path = CKPT / kind / f"{tag}.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    if restore_random:
        restore_rng(ckpt["rng"])
    return ckpt


def check_crops():
    onset = np.repeat(np.arange(120, dtype=np.int64), 3)
    assert len(onset) == 360
    for _ in range(40):
        start, end = sample_crop(onset)
        _assert_crop(onset, start, end)
        if end - start < CROP_MIN:
            raise AssertionError(f"crop shorter than 64: {end - start}")
    short = np.arange(20)
    if sample_crop(short) != (0, 20):
        raise AssertionError("short piece was not kept whole")
    start, end = prefix_crop(onset, 256)
    _assert_crop(onset, start, end)
    if end - start > 256:
        raise AssertionError("prefix crop too long")
    print("crop check passed", flush=True)


def check_shift(model, device):
    model.eval()
    batch, seq = 2, 16
    pitch = torch.randint(0, 128, (batch, seq), device=device)
    timing = torch.randn(batch, seq, 12, device=device)
    voice = torch.zeros(batch, seq, dtype=torch.long, device=device)
    velocity = torch.randint(1, 128, (batch, seq), device=device).float()
    available = torch.ones(batch, seq, dtype=torch.bool, device=device)
    available[:, 3] = False
    velocity[:, 3] = -12345
    pad = torch.ones(batch, seq, dtype=torch.bool, device=device)
    first = model(pitch, timing, voice, velocity, available, pad)
    index = 7
    tweaked = velocity.clone()
    tweaked[:, index] = torch.where(tweaked[:, index] < 60, tweaked[:, index] + 30, tweaked[:, index] - 30)
    second = model(pitch, timing, voice, tweaked, available, pad)
    if not torch.allclose(first[:, index], second[:, index], atol=1e-5):
        raise AssertionError("changing target v_i changed the logits for v_i")
    if torch.allclose(first[:, index + 1], second[:, index + 1], atol=1e-4):
        raise AssertionError("shifted history did not affect the next note")
    poison = velocity.clone()
    poison[:, 3] = 99999
    third = model(pitch, timing, voice, poison, available, pad)
    if not torch.allclose(first[:, 3], third[:, 3], atol=1e-5):
        raise AssertionError("missing velocity leaked into its own logits")
    if not torch.allclose(first[:, 4], third[:, 4], atol=1e-5):
        raise AssertionError("unknown history still depends on the missing value")
    print("decoder shift check passed", flush=True)


def check_chain(model, mu, sigma, device):
    model.eval()
    if model.schedule.timesteps != 200:
        raise AssertionError("diffusion schedule is not T=200")
    pitch = torch.randint(0, 128, (1, 4), device=device)
    timing = torch.zeros(1, 4, 12, device=device)
    voice = torch.zeros(1, 4, dtype=torch.long, device=device)
    pad = torch.ones(1, 4, dtype=torch.bool, device=device)
    held = capture_rng()
    out = sample_diffusion(model, pitch, timing, voice, pad, mu, sigma)
    if out.shape != (1, 4) or not torch.isfinite(out).all():
        raise AssertionError("full DDPM chain did not return finite velocities")
    fast = sample_diffusion_ddim(model, pitch, timing, voice, pad, mu, sigma, steps=50)
    if fast.shape != (1, 4) or not torch.isfinite(fast).all():
        raise AssertionError("DDIM-50 did not return finite velocities")
    restore_rng(held)
    print(
        f"diffusion chain passed full200 mean {float(out.mean()):.2f} ddim50 mean {float(fast.mean()):.2f}",
        flush=True,
    )


def check_grads(models, train, device, mu, sigma):
    batch = to_device(sample_batch(train, 2), device)
    for kind, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        if kind == "diffusion":
            loss = diffusion_loss(model, batch, mu, sigma)
        else:
            loss = decoder_loss(model, batch)
        if loss is None or not torch.isfinite(loss):
            raise AssertionError(f"{kind} loss is not finite: {loss}")
        loss.backward()
        total = 0.0
        saw = False
        for param in model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                raise AssertionError(f"{kind} has a non-finite gradient")
            total += float(param.grad.detach().float().pow(2).sum().cpu())
            saw = True
        if not saw or total <= 0:
            raise AssertionError(f"{kind} gradient norm is zero")
        model.zero_grad(set_to_none=True)
        print(f"grad check {kind} loss {float(loss.detach().cpu()):.4f}", flush=True)


def tiny_fit(kind, model, train, device, mu, sigma, steps=200):
    """Fit a handful of repeated crops. Reports the loss change, then discards nothing itself."""
    opt, sched = make_opt(model)
    model.train()
    crops = []
    guard = 0
    comps = list(train)
    while len(crops) < 4 and guard < 40:
        guard += 1
        perfs = train[comps[guard % len(comps)]]
        _record, arrays = perfs[0]
        start, end = prefix_crop(arrays["onset_id"], 96)
        item = slice_arrays(arrays, start, end)
        if item["available"].any():
            crops.append(item)
    if len(crops) < 2:
        raise RuntimeError("tiny set has fewer than 2 supervised crops")
    losses = []
    for _ in range(steps):
        batch = to_device(collate([crops[len(losses) % len(crops)]]), device)
        opt.zero_grad(set_to_none=True)
        sched.step()
        loss = diffusion_loss(model, batch, mu, sigma) if kind == "diffusion" else decoder_loss(model, batch)
        if loss is None or not torch.isfinite(loss):
            raise AssertionError(f"tiny fit {kind} produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    first = float(np.mean(losses[:20]))
    last = float(np.mean(losses[-20:]))
    print(
        f"tiny fit {kind} steps {steps} loss start {losses[0]:.4f} end {losses[-1]:.4f} "
        f"first20 {first:.4f} last20 {last:.4f} delta {last - first:.4f}",
        flush=True,
    )
    return {"start": losses[0], "end": losses[-1], "first20": first, "last20": last, "improved": last < first}


def probe_microbatch(model, device, mu, sigma):
    chosen = 4
    for micro in (16, 8, 4):
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            pitch = torch.randint(0, 128, (micro, CROP_MAX), device=device)
            timing = torch.randn(micro, CROP_MAX, 12, device=device)
            voice = torch.zeros(micro, CROP_MAX, dtype=torch.long, device=device)
            velocity = torch.randint(1, 128, (micro, CROP_MAX), device=device).float()
            available = torch.ones(micro, CROP_MAX, dtype=torch.bool, device=device)
            pad = torch.ones(micro, CROP_MAX, dtype=torch.bool, device=device)
            batch = {
                "pitch": pitch,
                "timing": timing,
                "voice": voice,
                "velocity": velocity,
                "available": available,
                "pad_mask": pad,
            }
            model.train()
            model.zero_grad(set_to_none=True)
            loss = diffusion_loss(model, batch, mu, sigma)
            loss.backward()
            model.zero_grad(set_to_none=True)
            chosen = micro
            break
        except RuntimeError as exc:
            model.zero_grad(set_to_none=True)
            if "memory" not in str(exc).lower():
                raise
            print(f"microbatch {micro} did not fit: {exc}", flush=True)
            chosen = 4
    print(f"using microbatch {chosen} with accumulation to batch {BATCH}", flush=True)
    return chosen


def midi_roundtrip():
    if not PERPETUITY.exists():
        raise RuntimeError(f"missing {PERPETUITY}")
    notes, _arrays = deadpan_from_midi(PERPETUITY)
    updates = {(int(note["track"]), int(note["on_msg"])): 90 for note in notes}
    dst = DEMO / "_checks" / "perpetuity_vel90.mid"
    apply_note_velocities(PERPETUITY, dst, updates)
    import mido

    original = mido.MidiFile(str(PERPETUITY))
    edited = mido.MidiFile(str(dst))
    if len(original.tracks) != len(edited.tracks):
        raise AssertionError("track count changed")
    for ti, (left, right) in enumerate(zip(original.tracks, edited.tracks)):
        if len(left) != len(right):
            raise AssertionError("event count changed")
        for mi, (a, b) in enumerate(zip(left, right)):
            if a.type != b.type or int(a.time) != int(b.time):
                raise AssertionError("pitch, time, or event order changed")
            if a.is_meta or b.is_meta:
                if a.dict() != b.dict():
                    raise AssertionError("a non-note event changed")
                continue
            if a.type == "note_on" and int(a.velocity) > 0 and (ti, mi) in updates:
                if int(b.velocity) != 90 or int(b.note) != int(a.note):
                    raise AssertionError("note-on pitch changed or velocity was not applied")
                continue
            if a.dict() != b.dict():
                raise AssertionError("an event other than the requested velocity changed")
    excerpt = DEMO / "_checks" / "perpetuity_excerpt.mid"
    arrays_notes = notes
    velocities = np.array([80 if i % 2 == 0 else 100 for i in range(len(arrays_notes))], dtype=np.int64)
    bounds = excerpt_bounds(
        np.array([note["start"] for note in arrays_notes]),
        np.arange(len(arrays_notes)),
    )
    if bounds is None:
        bounds = (0, min(32, len(arrays_notes)))
    start, end = bounds
    # Onset groups are not known here; the preservation check only needs a span.
    write_excerpt(PERPETUITY, excerpt, arrays_notes, velocities, start, end)
    print(f"midi preservation check passed ({len(notes)} notes, excerpt {start}:{end})", flush=True)
    return {"notes": len(notes), "excerpt": [int(start), int(end)]}


def panel_items(meta, panel):
    lookup = {record["file"]: record for record in meta["performances"]}
    items = []
    for spec in panel:
        arrays = load_perf(lookup[spec["file"]])
        items.append((spec, slice_arrays(arrays, spec["start"], spec["end"])))
    return items


def score_panel(kind, model, items, device, mu, sigma):
    model.eval()
    scores = []
    for spec, item in items:
        batch = to_device(collate([item]), device)
        if kind == "diffusion":
            value = masked_epsilon_mse(model, batch, mu, sigma, PANEL_T, seed=123)
        else:
            loss = decoder_loss(model, batch)
            value = None if loss is None else float(loss.detach().cpu())
        scores.append(value)
        print(f"panel {kind} {spec['composer']} {spec['title']} {value}", flush=True)
    good = [value for value in scores if value is not None]
    return float(np.mean(good)) if good else None


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or float(left.std()) < 1e-8 or float(right.std()) < 1e-8:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(value):
        return None
    return value


def contour_corr(pred, ref, onset, mask):
    pred_means = []
    ref_means = []
    for start, end in group_spans(onset):
        chosen = mask[start:end]
        if not np.any(chosen):
            continue
        pred_means.append(float(pred[start:end][chosen].mean()))
        ref_means.append(float(ref[start:end][chosen].mean()))
    if len(pred_means) < 2:
        return None
    return pearson(pred_means, ref_means)


def pair_metrics(raw, ref, onset, mask):
    raw = np.asarray(raw, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return None
    clipped = np.clip(raw, 1, 127)
    err = clipped[mask] - ref[mask]
    below = raw[mask] < 1
    above = raw[mask] > 127
    return {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": pearson(clipped[mask], ref[mask]),
        "pred_std": float(clipped[mask].std()),
        "true_std": float(ref[mask].std()),
        "frac_below_1": float(below.mean()),
        "frac_above_127": float(above.mean()),
        "frac_clipped": float((below | above).mean()),
        "endpoint_rate": float(np.mean((clipped[mask] <= 1.0 + 1e-6) | (clipped[mask] >= 127.0 - 1e-6))),
        "contour_r": contour_corr(clipped, ref, onset, mask),
        "preclip_std": float(raw[mask].std()),
    }


def compose_equal(rows):
    keys = [
        "rmse",
        "mae",
        "pearson",
        "pred_std",
        "true_std",
        "frac_below_1",
        "frac_above_127",
        "frac_clipped",
        "endpoint_rate",
        "contour_r",
        "preclip_std",
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["composer"], row["title"])].append(row["metrics"])
    summary = {}
    for key in keys:
        values = []
        undefined = 0
        for metrics in groups.values():
            good = []
            for item in metrics:
                if item is None or item.get(key) is None:
                    continue
                if isinstance(item[key], float) and not np.isfinite(item[key]):
                    continue
                good.append(float(item[key]))
            if not good:
                undefined += 1
                continue
            values.append(float(np.mean(good)))
        summary[key] = {
            "value": float(np.mean(values)) if values else None,
            "compositions": len(values),
            "undefined_compositions": undefined,
        }
    summary["pieces"] = len(groups)
    return summary


def generate_diffusion(model, arrays, mu, sigma, seed, device):
    raw = np.zeros(len(arrays["pitch"]), dtype=np.float64)
    held = capture_rng()
    started = time.time()
    try:
        for index, (start, end) in enumerate(window_spans(arrays["onset_id"])):
            torch.manual_seed(int(seed) + 10007 * index)
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(int(seed) + 10007 * index)
            batch = to_device(collate([slice_arrays(arrays, start, end)]), device)
            pred = sample_diffusion(
                model, batch["pitch"], batch["timing"], batch["voice"], batch["pad_mask"], mu, sigma
            )
            raw[start:end] = pred[0, : end - start].detach().cpu().numpy()
    finally:
        restore_rng(held)
    return raw, time.time() - started


def generate_decoder(model, arrays, mode, seed, device):
    raw = np.zeros(len(arrays["pitch"]), dtype=np.float64)
    started = time.time()
    for index, (start, end) in enumerate(window_spans(arrays["onset_id"])):
        batch = to_device(collate([slice_arrays(arrays, start, end)]), device)
        generator = None
        if mode == "sample":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + 10007 * index)
        pred = rollout_decoder(
            model,
            batch["pitch"],
            batch["timing"],
            batch["voice"],
            batch["pad_mask"],
            mode="sample" if mode == "sample" else "greedy",
            temperature=1.0,
            generator=generator,
        )
        raw[start:end] = pred[0, : end - start].detach().cpu().numpy()
    return raw, time.time() - started


def clip_report(raw):
    raw = np.asarray(raw, dtype=np.float64)
    below = raw < 1
    above = raw > 127
    return {
        "n": int(raw.size),
        "min": float(raw.min()) if raw.size else None,
        "max": float(raw.max()) if raw.size else None,
        "mean": float(raw.mean()) if raw.size else None,
        "std": float(raw.std()) if raw.size else None,
        "frac_below_1": float(below.mean()) if raw.size else None,
        "frac_above_127": float(above.mean()) if raw.size else None,
        "frac_clipped": float((below | above).mean()) if raw.size else None,
    }


def as_midi_velocities(raw):
    return np.clip(np.rint(raw), 1, 127).astype(np.int64)


def slug(text):
    cleaned = []
    for char in str(text):
        cleaned.append(char if char.isalnum() or char in "-_" else "_")
    return "".join(cleaned)[:80]


def write_html(manifest):
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<title>Velocity humanization demo</title>",
        "<style>body{font-family:Georgia,serif;max-width:52rem;margin:2rem auto;padding:0 1rem;background:#f6f1e7;color:#1c1915}",
        "a{color:#6b3a1f} section{margin:1.5rem 0;padding:1rem;background:#fffaf3;border:1px solid #e2d5c3}",
        "h1{font-weight:500} .row{margin:.4rem 0}</style></head><body>",
        "<h1>Velocity humanization</h1>",
        "<p>Same notes and timing in every file. Only key velocity changes. "
        "Audio players appear when a WAV render exists. MIDI links work either way.</p>",
    ]
    if not manifest.get("audio"):
        parts.append(
            "<p>No WAV files. fluidsynth is not installed, so these excerpts were not rendered. "
            "The local soundfont <code>venv_ml/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2</code> "
            "is present and was not used.</p>"
        )
    for section in manifest["sections"]:
        parts.append(f"<section><h2>{section['title']}</h2>")
        if section.get("note"):
            parts.append(f"<p>{section['note']}</p>")
        for item in section["files"]:
            parts.append("<div class='row'>")
            parts.append(f"<div>{item['label']}</div>")
            if item.get("wav"):
                parts.append(f"<audio controls src='{item['wav']}'></audio>")
            parts.append(f"<div><a href='{item['mid']}'>{item['mid']}</a></div>")
            parts.append("</div>")
        parts.append("</section>")
    parts.append("</body></html>")
    (DEMO / "index.html").write_text("\n".join(parts))


def write_readme(meta, state, metrics):
    steps = state["steps"]
    text = f"""# Velocity-only demo

This folder is a local listening demo. Two models predict **MIDI velocity only**.
Pitch, onset, duration, track, and note order are copied from the mechanical score.
Other MIDI events in an excerpt (tempo, time signature, pedal and controller events
that are not note messages) stay on the original timeline.

## What was trained

- Data: corrected (n)ASAP alignments already parsed in `src/eval_deexpress.py`.
  The global-stretch pitch matcher was not used. A match id is a MusicXML id with
  a repeat suffix. It is attached to a score note only when that note is the unique
  same-pitch note at the alignment's written onset. Performance velocities come from
  `link_notes` (same pitch, 30 ms). Unresolved notes have no target.
- Pre-filter compositions (composer, title): {meta['prefilter_compositions']}.
- Post-filter compositions: {meta['postfilter_compositions']}.
- Post-filter performances: {meta['postfilter_performances']}.
- Excluded performances: {meta['excluded_performances']} ({meta['exclusion_reasons']}).
- Supervised targets: {meta['supervised_targets']}. Score notes without a target: {meta['unmatched_targets']}.
- Cache: `data/processed/velocity_corrected/cache_v1` fingerprint `{meta['fingerprint']}`.
- Normalization for diffusion, training targets only: mu={meta['mu_train']:.4f}, sigma={meta['sigma_train']:.4f}.
  Inverted once at export. The decoder predicts MIDI velocities 1..127 directly.
- No GigaMIDI in this run. No pretraining, no adversarial training, no velocity exaggeration.
- Old checkpoints under `data/processed/*.pt` were not resumed and were not overwritten.

## Budgets

- Planned: 20,000 AdamW steps per model, batch 16, seed 42, fresh initialization.
- Completed optimizer steps: diffusion {steps['diffusion']}, encoder-decoder {steps['encdec']}.
- Equal step budget: {steps['diffusion'] == steps['encdec']}.
- Wall clock from launcher start: {elapsed(state)/3600:.2f} hours when this file was written.
- These counts are the run that happened. They are not an architecture ranking when they differ.

## Checkpoints

- Diffusion last: `data/processed/velocity_corrected/checkpoints/diffusion/last.pt`
- Diffusion best panel epsilon-MSE: `data/processed/velocity_corrected/checkpoints/diffusion/best_validation.pt`
- Decoder last: `data/processed/velocity_corrected/checkpoints/encdec/last.pt`
- Decoder best teacher-forced cross-entropy: `data/processed/velocity_corrected/checkpoints/encdec/best_validation.pt`
- Periodic copies: `step_XXXXXX.pt` in the same folders.
- Official diffusion samples use the full 200-step DDPM chain. DDIM-50 is implemented and was not used to pick checkpoints.

## Samples

- Diffusion seeds 0, 1, and 2 are three separate performances.
- Decoder greedy is the default autoregressive output. Seeds 0, 1, and 2 are temperature-1 categorical samples with no top-k or top-p filter.
- Context windows are at most 256 notes, split on onset groups. A longer excerpt is several windows, each starting from BOS for the decoder.
- Human reference files put the matched human velocity on the mechanical score's timing. Unmatched score notes are velocity 80. That is not the original performance MIDI.

## Audio

fluidsynth is not installed. MIDI files are the render. A General MIDI soundfont ships inside pretty_midi (`TimGM6mb.sf2`) and was not applied, because there is no local synthesizer to drive it. There is no loudness normalization because there are no audio files.

## Reproduce

```bash
cd {ROOT}
source venv_ml/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1
python -u src/velocity_run.py
```

The detached launcher is `scripts/launch_velocity_experiment.sh`. It resumes from `run_state.json` and the `last.pt` checkpoints. Test performances are counted in the filter report and are not used for training or model selection.

## How to read the metrics

RMSE, MAE, and Pearson correlation compare a generated velocity curve with one human performance. A lower correlation is not automatically a failure, and a lower RMSE is not automatically a better performance. Constant 80 and the training-set mean are the deployable baselines. A crop's own mean velocity is an oracle, not a model. This run does not claim a listening study, a literature reproduction, or state of the art.

Panel history and validation numbers are in `metrics.json`.
"""
    (DEMO / "README.md").write_text(text)
    blurb = metrics.get("blurb") or (
        "A local velocity-only experiment is in progress. Pitch and timing still come from the score."
    )
    (DEMO / "blurb.txt").write_text(blurb + "\n")


def blurb_from(meta, state, validation):
    steps = state["steps"]
    sentence = (
        f"Two small models learned piano key velocity from the written score, "
        f"using published (n)ASAP alignments ({meta['excluded_performances']} performances excluded, "
        f"{meta['unmatched_targets']} score notes left without a velocity target). "
        f"Training on this machine reached {steps['diffusion']} diffusion updates and "
        f"{steps['encdec']} autoregressive updates."
    )
    if not validation or not validation.get("diffusion"):
        return sentence + " Pitch and timing are copied from the score. This is not a full performance model and not a comparison with published systems."
    base = validation.get("baselines", {}).get("constant_80", {})
    diff = validation.get("diffusion", {}).get("seed_0", {})
    rmse_80 = (base.get("rmse") or {}).get("value")
    rmse_d = (diff.get("rmse") or {}).get("value")
    if rmse_80 is not None and rmse_d is not None:
        sentence += (
            f" On the fixed validation crops, diffusion seed 0 has RMSE {rmse_d:.1f} "
            f"against one human take, and always playing velocity 80 has RMSE {rmse_80:.1f}."
        )
    return (
        sentence
        + " Those distances describe agreement with a single reference, not a listening result, "
        "and they do not rank this run against published systems. The models do not generate timing, duration, or pedaling."
    )


def run_block(kind, count, models, opts, scheds, train, device, state, panel, perpetuity):
    model = models[kind]
    opt, sched = opts[kind], scheds[kind]
    model.train()
    micro = int(state["microbatch"])
    accum = max(1, BATCH // micro)
    mu = float(state["mu"])
    sigma = float(state["sigma"])
    for _ in range(count):
        if STOP["flag"] or elapsed(state) >= TRAIN_BUDGET_SEC:
            print(
                f"stop new updates at {elapsed(state)/3600:.2f} h "
                f"steps diffusion {state['steps']['diffusion']} encdec {state['steps']['encdec']}",
                flush=True,
            )
            return False
        opt.zero_grad(set_to_none=True)
        lr = sched.step()
        losses = []
        for _micro in range(accum):
            batch = to_device(sample_batch(train, micro), device)
            loss = diffusion_loss(model, batch, mu, sigma) if kind == "diffusion" else decoder_loss(model, batch)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {kind} loss at step {state['steps'][kind]}")
            (loss / accum).backward()
            losses.append(float(loss.detach().cpu()))
        grads = [param.grad for param in model.parameters() if param.grad is not None]
        if not grads:
            raise RuntimeError(f"{kind} produced no gradients")
        torch.nn.utils.clip_grad_norm_(grads, 1.0)
        opt.step()
        state["steps"][kind] += 1
        step = state["steps"][kind]
        if step == 1 or step % 10 == 0:
            print(
                f"update {kind} step {step} loss {float(np.mean(losses)):.4f} lr {lr:.6g} "
                f"elapsed {elapsed(state)/60:.1f} min",
                flush=True,
            )
        if step % 500 == 0:
            save_checkpoint(kind, "last", model, opt, sched, state)
            save_checkpoint(kind, f"step_{step:06d}", model, opt, sched, state)
            save_state(state)
            if step == 500:
                export_early(kind, model, perpetuity, device, mu, sigma, step)
    return True


def export_early(kind, model, perpetuity, device, mu, sigma, step):
    if perpetuity is None:
        return
    notes, arrays, bounds = perpetuity
    start, end = bounds
    window = slice_arrays(arrays, start, end)
    # One window, so the first sample stays short.
    first = window_spans(window["onset_id"])[:1]
    if not first:
        return
    piece = slice_arrays(window, first[0][0], first[0][1])
    model.eval()
    if kind == "diffusion":
        raw, seconds = generate_diffusion(model, piece, mu, sigma, seed=0, device=device)
        name = f"perpetuity_diffusion_step{step}_seed0.mid"
    else:
        raw, seconds = generate_decoder(model, piece, mode="greedy", seed=0, device=device)
        name = f"perpetuity_encdec_greedy_step{step}.mid"
    report = clip_report(raw)
    velocities = np.zeros(len(arrays["pitch"]), dtype=np.int64)
    velocities[start:end] = 80
    local = as_midi_velocities(raw)
    a, b = first[0]
    velocities[start + a : start + b] = local
    dest = DEMO / "early" / name
    write_excerpt(PERPETUITY, dest, notes, velocities, start, start + (b if False else (end - start)))
    # The excerpt above still contains the whole 30–45 s span; only the first window was generated.
    # Rewrite so the file is exactly the generated window.
    write_excerpt(PERPETUITY, dest, notes, velocities, start + a, start + b)
    print(f"early sample {dest} render {seconds:.1f}s clip {report['frac_clipped']}", flush=True)


def free_run_log(model, items, device):
    rows = []
    for spec, item in items:
        ref = item["velocity"]
        mask = item["available"]
        onset = item["onset_id"]
        greedy, _seconds = generate_decoder(model, item, "greedy", 0, device)
        entry = {"composer": spec["composer"], "title": spec["title"], "greedy": pair_metrics(greedy, ref, onset, mask)}
        for seed in (0, 1, 2):
            sample, _seconds = generate_decoder(model, item, "sample", seed, device)
            entry[f"seed_{seed}"] = pair_metrics(sample, ref, onset, mask)
        rows.append(entry)
        print(
            f"free-run {spec['title']} greedy rmse "
            f"{None if entry['greedy'] is None else round(entry['greedy']['rmse'], 2)}",
            flush=True,
        )
    return rows


def diagnostic_ddpm(model, items, device, mu, sigma):
    if not items:
        return None
    spec, item = items[0]
    raw, seconds = generate_diffusion(model, item, mu, sigma, seed=0, device=device)
    metrics = pair_metrics(raw, item["velocity"], item["onset_id"], item["available"])
    print(
        f"diagnostic full DDPM {spec['title']} {seconds:.1f}s "
        f"rmse {None if metrics is None else round(metrics['rmse'], 2)}",
        flush=True,
    )
    return {"title": spec["title"], "seconds": seconds, "metrics": metrics}


def schedule_position(steps):
    """Where the alternating blocks are, derived from completed optimizer steps."""
    plan = build_plan()
    consumed = {"diffusion": 0, "encdec": 0}
    for index, (kind, block) in enumerate(plan):
        done_here = int(steps[kind]) - consumed[kind]
        if done_here < block:
            return plan, index, kind, done_here, block
        consumed[kind] += block
    return plan, len(plan), None, 0, 0


def block_scored(state, kind, step):
    return any(item.get("kind") == kind and int(item.get("step", -1)) == int(step) for item in state["panel_history"])


def train(meta, state, models, opts, scheds, train_data, device, panel, perpetuity):
    while True:
        if STOP["flag"] or elapsed(state) >= TRAIN_BUDGET_SEC:
            break
        if state["steps"]["diffusion"] >= TOTAL_STEPS and state["steps"]["encdec"] >= TOTAL_STEPS:
            break
        plan, index, kind, done_here, block = schedule_position(state["steps"])
        state["plan_index"] = index
        state["into_block"] = done_here
        if kind is None:
            break
        if done_here == 0 and index > 0:
            previous = plan[index - 1][0]
            if not block_scored(state, previous, state["steps"][previous]):
                _record_panel(previous, meta, state, models, opts, scheds, device, panel)
        remaining = block - done_here
        print(f"block {kind} updates {remaining} (plan {index}, {done_here}/{block})", flush=True)
        finished = run_block(kind, remaining, models, opts, scheds, train_data, device, state, panel, perpetuity)
        save_checkpoint(kind, "last", models[kind], opts[kind], scheds[kind], state)
        _plan, index_after, _kind_now, done_now, _block_now = schedule_position(state["steps"])
        if done_now == 0 and index_after > 0 and finished:
            finished_kind = _plan[index_after - 1][0]
            if not block_scored(state, finished_kind, state["steps"][finished_kind]):
                _record_panel(finished_kind, meta, state, models, opts, scheds, device, panel)
        save_state(state)
        if not finished:
            break
    save_state(state)


def _record_panel(kind, meta, state, models, opts, scheds, device, panel):
    items = panel_items(meta, panel)
    score = score_panel(kind, models[kind], items, device, state["mu"], state["sigma"])
    entry = {"kind": kind, "step": state["steps"][kind], "score": score, "elapsed_sec": elapsed(state)}
    if score is not None and score < state["best_score"][kind]:
        state["best_score"][kind] = score
        save_checkpoint(kind, "best_validation", models[kind], opts[kind], scheds[kind], state, score)
        print(f"new best {kind} {score:.4f}", flush=True)
    if kind == "diffusion":
        entry["diagnostic_ddpm"] = diagnostic_ddpm(models[kind], items, device, state["mu"], state["sigma"])
    else:
        entry["free_run"] = free_run_log(models[kind], items, device)
    state["panel_history"].append(entry)


def export_variants(folder, src, notes, arrays, start, end, generated, human=None):
    folder.mkdir(parents=True, exist_ok=True)
    length = len(arrays["pitch"])
    base = np.array([int(arrays["velocity"][i]) if False else 80 for i in range(length)], dtype=np.int64)
    # Mechanical score velocities are not in `arrays` for Perpetuity (zeros). Callers pass notes.
    mechanical = np.array([int(note.get("velocity", 80)) for note in notes], dtype=np.int64)
    if len(mechanical) != length:
        mechanical = np.full(length, 80, dtype=np.int64)
    files = []
    specs = {
        "flat80.mid": np.full(length, 80, dtype=np.int64),
    }
    specs.update(generated)
    if human is not None:
        specs = {"human_velocity_reference.mid": human, **specs}
    # Original mechanical excerpt under a stable name when provided by the caller via notes' velocity.
    write_excerpt(src, folder / "score_mechanical.mid", notes, mechanical, start, end)
    files.append("score_mechanical.mid")
    reports = {}
    for name, values in specs.items():
        raw_values = np.asarray(values, dtype=np.float64)
        if name.startswith("flat") or name.startswith("human") or name.startswith("score"):
            midi_values = np.clip(np.rint(raw_values), 1, 127).astype(np.int64)
            reports[name] = clip_report(raw_values)
        else:
            reports[name] = clip_report(raw_values[start:end] if raw_values.shape[0] == length else raw_values)
            midi_values = np.full(length, 80, dtype=np.int64)
            midi_values[start:end] = as_midi_velocities(raw_values if raw_values.shape[0] == (end - start) else raw_values[start:end])
            if raw_values.shape[0] == length:
                midi_values = as_midi_velocities(raw_values)
                midi_values[:start] = 80
                midi_values[end:] = 80
        write_excerpt(src, folder / name, notes, midi_values if name not in ("flat80.mid", "human_velocity_reference.mid") else np.clip(np.rint(raw_values), 1, 127).astype(np.int64), start, end)
        files.append(name)
    return files, reports


def package(meta, state, models, device, panel_spec, excerpt_spec, perpetuity, old_snapshot):
    print("packaging demo", flush=True)
    mu, sigma = float(state["mu"]), float(state["sigma"])
    for kind in ("diffusion", "encdec"):
        best = CKPT / kind / "best_validation.pt"
        last = CKPT / kind / "last.pt"
        path = best if best.exists() else last
        if path.exists() and state["steps"][kind] > 0:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            models[kind].load_state_dict(ckpt["model"])
            models[kind].eval()
            state.setdefault("selected", {})[kind] = str(path.relative_to(ROOT))
    manifest = {"audio": False, "sections": []}
    clip_logs = {}
    runtime = {}

    notes, arrays, bounds = perpetuity
    start, end = bounds
    piece = slice_arrays(arrays, start, end)
    generated = {}
    if state["steps"]["diffusion"] > 0:
        for seed in (0, 1, 2):
            raw, seconds = generate_diffusion(models["diffusion"], piece, mu, sigma, seed, device)
            full = np.zeros(len(arrays["pitch"]), dtype=np.float64)
            full[start:end] = raw
            generated[f"diffusion_seed{seed}.mid"] = full
            runtime[f"perpetuity_diffusion_seed{seed}"] = seconds
    if state["steps"]["encdec"] > 0:
        raw, seconds = generate_decoder(models["encdec"], piece, "greedy", 0, device)
        full = np.zeros(len(arrays["pitch"]), dtype=np.float64)
        full[start:end] = raw
        generated["encdec_greedy.mid"] = full
        runtime["perpetuity_encdec_greedy"] = seconds
        for seed in (0, 1, 2):
            raw, seconds = generate_decoder(models["encdec"], piece, "sample", seed, device)
            full = np.zeros(len(arrays["pitch"]), dtype=np.float64)
            full[start:end] = raw
            generated[f"encdec_seed{seed}.mid"] = full
            runtime[f"perpetuity_encdec_seed{seed}"] = seconds
    folder = DEMO / "perpetuity"
    _files, reports = export_variants(folder, PERPETUITY, notes, arrays, start, end, generated)
    # The requested source name.
    mechanical = np.array([int(note["velocity"]) for note in notes], dtype=np.int64)
    write_excerpt(PERPETUITY, folder / "Perpetuity-Piano.mid", notes, mechanical, start, end)
    clip_logs["perpetuity"] = reports
    section_files = [{"label": "Perpetuity-Piano.mid (mechanical excerpt)", "mid": "perpetuity/Perpetuity-Piano.mid"}]
    for name in [
        "flat80.mid",
        "diffusion_seed0.mid",
        "diffusion_seed1.mid",
        "diffusion_seed2.mid",
        "encdec_greedy.mid",
        "encdec_seed0.mid",
        "encdec_seed1.mid",
        "encdec_seed2.mid",
    ]:
        if (folder / name).exists():
            section_files.append({"label": name, "mid": f"perpetuity/{name}"})
    manifest["sections"].append(
        {
            "title": "Perpetuity",
            "note": f"Score excerpt {bounds[0]}:{bounds[1]}, {float(arrays['start'][end-1] - arrays['start'][start]):.1f} s from the first note.",
            "files": section_files,
        }
    )

    lookup = {record["file"]: record for record in meta["performances"]}
    excerpt_metrics = []
    for spec in excerpt_spec:
        record = lookup[spec["file"]]
        arrays = load_perf(record)
        notes = notes_for_excerpt(arrays)
        for note, vel in zip(notes, collect_notes(ASAP / record["midi_score"])[1]):
            note["velocity"] = int(vel["velocity"])
        start, end = int(spec["start"]), int(spec["end"])
        piece = slice_arrays(arrays, start, end)
        generated = {}
        if state["steps"]["diffusion"] > 0:
            for seed in (0, 1, 2):
                raw, seconds = generate_diffusion(models["diffusion"], piece, mu, sigma, seed, device)
                full = np.full(len(arrays["pitch"]), 80.0)
                full[start:end] = raw
                generated[f"diffusion_seed{seed}.mid"] = full
                runtime[f"{spec['title']}_diffusion_seed{seed}"] = seconds
        if state["steps"]["encdec"] > 0:
            raw, seconds = generate_decoder(models["encdec"], piece, "greedy", 0, device)
            full = np.full(len(arrays["pitch"]), 80.0)
            full[start:end] = raw
            generated["encdec_greedy.mid"] = full
            runtime[f"{spec['title']}_encdec_greedy"] = seconds
            for seed in (0, 1, 2):
                raw, seconds = generate_decoder(models["encdec"], piece, "sample", seed, device)
                full = np.full(len(arrays["pitch"]), 80.0)
                full[start:end] = raw
                generated[f"encdec_seed{seed}.mid"] = full
        human = np.full(len(arrays["pitch"]), 80.0)
        human_mask = np.asarray(arrays["available"], dtype=bool)
        human[human_mask] = np.asarray(arrays["velocity"], dtype=np.float64)[human_mask]
        folder_name = slug(spec["composer"]) + "__" + slug(spec["title"])
        folder = DEMO / folder_name
        _files, reports = export_variants(
            folder, ASAP / record["midi_score"], notes, arrays, start, end, generated, human=human
        )
        clip_logs[folder_name] = reports
        ref = np.asarray(arrays["velocity"][start:end], dtype=np.float64)
        mask = np.asarray(arrays["available"][start:end], dtype=bool)
        onset = np.asarray(arrays["onset_id"][start:end])
        entry = {
            "composer": spec["composer"],
            "title": spec["title"],
            "unmatched": int((~mask).sum()),
            "notes": int(end - start),
            "duration_sec": spec.get("duration_sec"),
        }
        entry["constant_80"] = pair_metrics(np.full(end - start, 80.0), ref, onset, mask)
        entry["train_mean"] = pair_metrics(np.full(end - start, mu), ref, onset, mask)
        if state["steps"]["diffusion"] > 0:
            for seed in (0, 1, 2):
                raw = generated[f"diffusion_seed{seed}.mid"][start:end]
                entry[f"diffusion_seed_{seed}"] = pair_metrics(raw, ref, onset, mask)
        if state["steps"]["encdec"] > 0:
            entry["encdec_greedy"] = pair_metrics(generated["encdec_greedy.mid"][start:end], ref, onset, mask)
            for seed in (0, 1, 2):
                entry[f"encdec_seed_{seed}"] = pair_metrics(generated[f"encdec_seed{seed}.mid"][start:end], ref, onset, mask)
        excerpt_metrics.append(entry)
        section_files = []
        for path in sorted(folder.glob("*.mid")):
            rel = f"{folder_name}/{path.name}"
            wav = path.with_suffix(".wav")
            section_files.append(
                {
                    "label": path.name,
                    "mid": rel,
                    "wav": f"{folder_name}/{wav.name}" if wav.exists() else None,
                }
            )
        manifest["sections"].append(
            {
                "title": f"{spec['composer']} — {spec['title']}",
                "note": (
                    f"Notes {start}:{end}, {spec.get('duration_sec'):.1f} s. "
                    f"Unmatched score notes set to 80 in the human reference: {int((~mask).sum())}."
                ),
                "files": section_files,
            }
        )

    validation = evaluate_validation(meta, state, models, device, panel_spec)
    metrics = {
        "version": VERSION,
        "fingerprint": meta["fingerprint"],
        "steps": state["steps"],
        "equal_step_budget": state["steps"]["diffusion"] == state["steps"]["encdec"],
        "elapsed_hours": elapsed(state) / 3600,
        "mu_train": mu,
        "sigma_train": sigma,
        "prefilter_compositions": meta["prefilter_compositions"],
        "postfilter_compositions": meta["postfilter_compositions"],
        "postfilter_performances": meta["postfilter_performances"],
        "excluded_performances": meta["excluded_performances"],
        "exclusion_reasons": meta["exclusion_reasons"],
        "supervised_targets": meta["supervised_targets"],
        "unmatched_targets": meta["unmatched_targets"],
        "selected_checkpoints": state.get("selected", {}),
        "panel_history": state.get("panel_history", []),
        "excerpt_metrics": excerpt_metrics,
        "clip_before_export": clip_logs,
        "generation_runtime_sec": runtime,
        "validation": validation,
        "audio": None,
        "audio_note": "fluidsynth is not installed; no WAV files were written",
    }
    metrics["blurb"] = blurb_from(meta, state, validation)
    (DEMO / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    write_html(manifest)
    write_readme(meta, state, metrics)
    state["packaged"] = True
    save_state(state)
    assert_old_unchanged(old_snapshot)
    print("packaging done", flush=True)
    print(metrics["blurb"], flush=True)


def _json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def evaluate_validation(meta, state, models, device, panel_spec):
    """Full validation crops until the six-hour mark. Panel and demos are already saved."""
    mu, sigma = float(state["mu"]), float(state["sigma"])
    crops = json.loads((DEMO / "val_crops.json").read_text())
    lookup = {record["file"]: record for record in meta["performances"]}
    scored = []
    diffusion_rows = {seed: [] for seed in (0, 1, 2)}
    greedy_rows = []
    sample_rows = {seed: [] for seed in (0, 1, 2)}
    teacher_rows = []
    base80_rows = []
    mean_rows = []
    oracle_rows = []
    complete = True
    for index, spec in enumerate(crops):
        if elapsed(state) >= PACK_BUDGET_SEC:
            complete = False
            print(f"validation stopped after {index} crops to keep the time reserve", flush=True)
            break
        arrays = load_perf(lookup[spec["file"]])
        item = slice_arrays(arrays, spec["start"], spec["end"])
        ref = item["velocity"]
        mask = item["available"]
        onset = item["onset_id"]
        identity = {"composer": spec["composer"], "title": spec["title"], "file": spec["file"]}
        base80_rows.append({**identity, "metrics": pair_metrics(np.full(len(ref), 80.0), ref, onset, mask)})
        mean_rows.append({**identity, "metrics": pair_metrics(np.full(len(ref), mu), ref, onset, mask)})
        if int(mask.sum()) > 0:
            oracle_rows.append(
                {**identity, "metrics": pair_metrics(np.full(len(ref), float(ref[mask].mean())), ref, onset, mask)}
            )
        if state["steps"]["diffusion"] > 0:
            for seed in (0, 1, 2):
                raw, _seconds = generate_diffusion(models["diffusion"], item, mu, sigma, seed, device)
                diffusion_rows[seed].append({**identity, "metrics": pair_metrics(raw, ref, onset, mask)})
        if state["steps"]["encdec"] > 0:
            models["encdec"].eval()
            batch = to_device(collate([item]), device)
            logits = models["encdec"](
                batch["pitch"], batch["timing"], batch["voice"], batch["velocity"], batch["available"], batch["pad_mask"]
            )
            pred = (logits.argmax(dim=-1) + 1).detach().cpu().numpy()[0]
            # Teacher-forced argmax is not a free run. Distance uses that argmax only as a diagnostic beside CE.
            loss = decoder_loss(models["encdec"], batch)
            teacher_rows.append(
                {
                    **identity,
                    "metrics": pair_metrics(pred, ref, onset, mask),
                    "cross_entropy": None if loss is None else float(loss.detach().cpu()),
                }
            )
            raw, _seconds = generate_decoder(models["encdec"], item, "greedy", 0, device)
            greedy_rows.append({**identity, "metrics": pair_metrics(raw, ref, onset, mask)})
            for seed in (0, 1, 2):
                raw, _seconds = generate_decoder(models["encdec"], item, "sample", seed, device)
                sample_rows[seed].append({**identity, "metrics": pair_metrics(raw, ref, onset, mask)})
        scored.append(spec["file"])
        if index % 5 == 0:
            print(f"validation crops {index + 1}/{len(crops)}", flush=True)
    ce_groups = defaultdict(list)
    for row in teacher_rows:
        if row.get("cross_entropy") is None:
            continue
        ce_groups[(row["composer"], row["title"])].append(float(row["cross_entropy"]))
    ce_values = [float(np.mean(values)) for values in ce_groups.values() if values]
    teacher_ce_summary = {
        "value": float(np.mean(ce_values)) if ce_values else None,
        "compositions": len(ce_values),
        "undefined_compositions": 0,
    }
    return {
        "complete": complete,
        "crops_scored": len(scored),
        "crops_total": len(crops),
        "baselines": {
            "constant_80": compose_equal(base80_rows),
            "train_mean": compose_equal(mean_rows),
            "oracle_crop_mean": compose_equal(oracle_rows),
        },
        "diffusion": {f"seed_{seed}": compose_equal(diffusion_rows[seed]) for seed in (0, 1, 2)},
        "encdec_teacher_forced_argmax": compose_equal(
            [{"composer": row["composer"], "title": row["title"], "metrics": row["metrics"]} for row in teacher_rows]
        ),
        "encdec_teacher_forced_cross_entropy": teacher_ce_summary,
        "encdec_greedy": compose_equal(greedy_rows),
        "encdec_sample": {f"seed_{seed}": compose_equal(sample_rows[seed]) for seed in (0, 1, 2)},
        "note": "Teacher-forced numbers are not generation metrics. Oracle crop mean is not a deployable baseline.",
        "teacher_ce_values_pooled_count": int(sum(len(values) for values in ce_groups.values())),
    }


def main():
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    DEMO.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / "velocity_experiment.pid").write_text(str(os.getpid()))
    old_snapshot = snapshot_old()
    print(f"pid {os.getpid()} device check starting", flush=True)
    meta = build_cache(force=False)
    print(
        f"counts pre {meta['prefilter_compositions']} post {meta['postfilter_compositions']} "
        f"excluded {meta['excluded_performances']}",
        flush=True,
    )
    spot = spot_check(meta, 100)
    if spot["checked"] < 100 or spot["train"] < 1 or spot["val"] < 1:
        raise RuntimeError(f"spot check did not cover train and val: {spot}")
    selection = select_panel_and_excerpts(meta)
    (DEMO / "panel.json").write_text(json.dumps(selection["panel"], indent=2))
    (DEMO / "excerpts.json").write_text(json.dumps(selection["excerpts"], indent=2))
    (DEMO / "val_crops.json").write_text(json.dumps(selection["val_crops"], indent=2))
    print(
        f"panel {len(selection['panel'])} excerpts {len(selection['excerpts'])} "
        f"val crops {len(selection['val_crops'])}",
        flush=True,
    )
    if len(selection["panel"]) < 1 or len(selection["excerpts"]) < 2:
        raise RuntimeError("not enough validation excerpts for the demo")

    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    else:
        state = {
            "started_unix": time.time(),
            "plan_index": 0,
            "into_block": 0,
            "steps": {"diffusion": 0, "encdec": 0},
            "best_score": {"diffusion": 1e9, "encdec": 1e9},
            "checks_passed": False,
            "microbatch": 16,
            "fingerprint": meta["fingerprint"],
            "mu": meta["mu_train"],
            "sigma": meta["sigma_train"],
            "packaged": False,
            "panel_history": [],
            "checks": {},
        }
        save_state(state)
    if state["fingerprint"] != meta["fingerprint"]:
        raise SystemExit("cache fingerprint changed; refusing to resume on a different dataset")

    device = get_device()
    print("device", device, flush=True)
    notes, arrays = deadpan_from_midi(PERPETUITY)
    bounds = excerpt_bounds(arrays["start"], arrays["onset_id"])
    if bounds is None:
        bounds = prefix_crop(arrays["onset_id"], CROP_MAX)
    perpetuity = (notes, arrays, bounds)
    (DEMO / "perpetuity_bounds.json").write_text(
        json.dumps({"start": int(bounds[0]), "end": int(bounds[1]),
                    "t0": float(arrays["start"][bounds[0]]),
                    "t1": float(arrays["start"][bounds[1] - 1])}, indent=2)
    )

    if not state.get("checks_passed"):
        check_crops()
        midi_info = midi_roundtrip()
        set_seed(42)
        models = make_models(device)
        print(f"parameters diffusion {param_count(models['diffusion'])} encdec {param_count(models['encdec'])}", flush=True)
        initial = {kind: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()} for kind, model in models.items()}
        check_shift(models["encdec"], device)
        check_chain(models["diffusion"], state["mu"], state["sigma"], device)
        train_data = corpus_from(meta, "train")
        check_grads(models, train_data, device, state["mu"], state["sigma"])
        tiny = {}
        for kind in ("diffusion", "encdec"):
            clone = make_models(device)[kind]
            clone.load_state_dict(models[kind].state_dict())
            tiny[kind] = tiny_fit(kind, clone, train_data, device, state["mu"], state["sigma"], steps=200)
            del clone
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        for kind, model in models.items():
            model.load_state_dict(initial[kind])
        state["microbatch"] = probe_microbatch(models["diffusion"], device, state["mu"], state["sigma"])
        for kind, model in models.items():
            model.load_state_dict(initial[kind])
            model.zero_grad(set_to_none=True)
        state["checks"] = {"spot": spot, "midi": midi_info, "tiny_fit": tiny,
                           "parameters": {kind: param_count(model) for kind, model in models.items()}}
        state["checks_passed"] = True
        save_state(state)
        (DEMO / "checks.json").write_text(json.dumps(state["checks"], indent=2))
        print("functional checks passed", flush=True)
    else:
        train_data = corpus_from(meta, "train")
        set_seed(42)
        models = make_models(device)
        print(f"parameters diffusion {param_count(models['diffusion'])} encdec {param_count(models['encdec'])}", flush=True)

    opts = {}
    scheds = {}
    for kind, model in models.items():
        opts[kind], scheds[kind] = make_opt(model)
    if state["steps"]["diffusion"] or state["steps"]["encdec"]:
        newest = None
        newest_mtime = -1.0
        for kind in ("diffusion", "encdec"):
            if state["steps"][kind] <= 0:
                continue
            ckpt = load_checkpoint(kind, "last", models[kind], opts[kind], scheds[kind], device)
            mtime = (CKPT / kind / "last.pt").stat().st_mtime
            if mtime >= newest_mtime:
                newest_mtime = mtime
                newest = ckpt
            print(f"resumed {kind} at step {state['steps'][kind]}", flush=True)
        if newest is not None:
            restore_rng(newest["rng"])
    else:
        set_seed(42)

    if not state.get("packaged") and elapsed(state) < TRAIN_BUDGET_SEC and not (
        state["steps"]["diffusion"] >= TOTAL_STEPS and state["steps"]["encdec"] >= TOTAL_STEPS
    ):
        train(meta, state, models, opts, scheds, train_data, device, selection["panel"], perpetuity)
    for kind in ("diffusion", "encdec"):
        if state["steps"][kind] > 0:
            save_checkpoint(kind, "last", models[kind], opts[kind], scheds[kind], state)
    save_state(state)
    package(meta, state, models, device, selection["panel"], selection["excerpts"], perpetuity, old_snapshot)
    assert_old_unchanged(old_snapshot)
    print(
        f"done steps diffusion {state['steps']['diffusion']} encdec {state['steps']['encdec']} "
        f"elapsed {elapsed(state)/3600:.2f} h",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
