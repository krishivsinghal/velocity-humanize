"""
Phrase encoder-decoder. A first look, not the velocity-diffusion comparison.

The encoder reads a score phrase. Phrase length is uniform from 8 to 100 notes,
the "about 50, give or take 50" cut. The decoder writes the same notes back in
performance language:

  velocity, onset offset, IOI delta, duration, duration delta

Offset is the early/late of one note after the local tempo is removed.
IOI delta is performed gap minus score gap, in beats, so it still carries
tempo. Duration is the performed length in beats. Duration delta is that
length minus the score length.

The decoder is not handed the current pitch. It has to read the note from the
encoder. Cross-entropy on pitch is weighted so a wrong note costs far more
than a small timing or dynamic error. Once the notes are right, that term
goes quiet and the expressive losses can move.

ASAP timing is supervised only when the performance onset agrees with the
local tempo to within a beat. The pitch alignment sometimes gives a repeated
pitch the right key and the wrong time; those residuals are many beats and
are not an early or late. Velocity is still taken from the paired note.

  PYTHONPATH=src python src/train_seq2seq.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import SpanDataset
from features import N_TIMING_STRUCT, N_VOICE, assemble_timing, note_features
from score_structure import load_structure
from train_diffusion_struct import StructuredVelocityDataset
from train_transformer import (
    CENTER_VEL,
    D_MODEL,
    DROPOUT,
    EMBED_LR,
    EncoderLayer,
    LR,
    N_HEADS,
    N_LAYERS,
    N_PITCH,
    RotaryEmbedding,
    SwiGLU,
    get_device,
)

PROCESSED = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/processed")
SAVE_PATH = PROCESSED / "phrase_seq2seq.pt"

PHRASE_MIN = 8
PHRASE_MAX = 100
BATCH_SIZE = 16
STEPS = 4000
# Cross-entropy is a few nats when the note is wrong and ~0 when it is right.
# 25 puts note identity well above the expressive Huber terms until it locks.
NOTE_PENALTY = 25.0
EXPR_NAMES = ("velocity", "offset", "ioi_delta", "duration", "duration_delta")
# Divides each target so a typical error is order 1. Velocity is centered 0–1.
# Offset and IOI scales match the inlier ASAP residuals (a few tenths of a beat).
EXPR_SCALE = (0.12, 0.30, 0.45, 1.5, 0.80)
# A residual larger than this is a mismatched note, not a musical early/late.
TIMING_GATE_BEATS = 1.0


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.norm_cross = nn.LayerNorm(D_MODEL)
        self.cross_q = nn.Linear(D_MODEL, D_MODEL)
        self.cross_kv = nn.Linear(D_MODEL, 2 * D_MODEL)
        self.cross_proj = nn.Linear(D_MODEL, D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.mlp = SwiGLU(D_MODEL, 384)
        self.drop = nn.Dropout(DROPOUT)

    def _attend(self, q, k, v, mask):
        q = q.view(q.size(0), q.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).contiguous().view(q.size(0), -1, D_MODEL)

    def forward(self, x, memory, self_mask, cross_mask, rope):
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rope(q.view(q.size(0), q.size(1), self.n_heads, self.head_dim).transpose(1, 2))
        k = rope(k.view(k.size(0), k.size(1), self.n_heads, self.head_dim).transpose(1, 2))
        v = v.view(v.size(0), v.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=self_mask)
        attn = attn.transpose(1, 2).contiguous().view(x.size(0), x.size(1), D_MODEL)
        x = x + self.drop(self.proj(attn))

        h = self.norm_cross(x)
        q = self.cross_q(h)
        k, v = self.cross_kv(memory).chunk(2, dim=-1)
        cross = self._attend(q, k, v, cross_mask)
        x = x + self.drop(self.cross_proj(cross))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class PhraseSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.pitch_emb = nn.Embedding(N_PITCH, D_MODEL)
        self.voice_emb = nn.Embedding(N_VOICE, D_MODEL)
        self.timing_proj = nn.Linear(N_TIMING_STRUCT, D_MODEL)
        self.expr_proj = nn.Linear(len(EXPR_NAMES), D_MODEL)
        self.start = nn.Parameter(torch.zeros(D_MODEL))
        self.rope = RotaryEmbedding(D_MODEL // N_HEADS, max_len=PHRASE_MAX + 8)
        self.encoder = nn.ModuleList([EncoderLayer(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)])
        self.decoder = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])
        self.enc_norm = nn.LayerNorm(D_MODEL)
        self.dec_norm = nn.LayerNorm(D_MODEL)
        self.pitch_head = nn.Linear(D_MODEL, N_PITCH)
        self.expr_head = nn.Linear(D_MODEL, len(EXPR_NAMES))

    def encode(self, pitch, timing, voice, pad_mask):
        x = self.pitch_emb(pitch) + self.voice_emb(voice) + self.timing_proj(timing)
        batch, seq = pad_mask.shape
        mask = torch.zeros(batch, 1, 1, seq, device=x.device, dtype=x.dtype)
        mask = mask.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        for layer in self.encoder:
            x = layer(x, self.rope, mask)
        return self.enc_norm(x)

    def decode(self, memory, prev_pitch, prev_expr, pad_mask):
        dec = self.pitch_emb(prev_pitch) + self.expr_proj(prev_expr)
        dec = dec.clone()
        dec[:, 0] = self.start
        batch, seq = pad_mask.shape
        causal = torch.triu(torch.full((seq, seq), float("-inf"), device=dec.device), diagonal=1)
        pad = torch.zeros(batch, 1, 1, seq, device=dec.device, dtype=dec.dtype)
        pad = pad.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        self_mask = causal.view(1, 1, seq, seq) + pad
        cross_mask = pad
        for layer in self.decoder:
            dec = layer(dec, memory, self_mask, cross_mask, self.rope)
        # Position i is note i. The encoder state is added so the pitch head
        # can read that note; the loss still charges a miss. The decoder state
        # on top of it is what carries offset, timing, and duration.
        h = self.dec_norm(dec) + memory
        return self.pitch_head(h), self.expr_head(h)

    def forward(self, pitch, timing, voice, expr, pad_mask):
        prev_pitch = torch.zeros_like(pitch)
        prev_expr = torch.zeros_like(expr)
        prev_pitch[:, 1:] = pitch[:, :-1]
        prev_expr[:, 1:] = expr[:, :-1]
        memory = self.encode(pitch, timing, voice, pad_mask)
        return self.decode(memory, prev_pitch, prev_expr, pad_mask)


def _local_median(times, values, half, keep=None):
    """Median of values whose times fall in a window. keep drops mismatched attacks."""
    n = len(times)
    order = np.argsort(times, kind="mergesort")
    ordered_t = times[order]
    ordered_v = values[order]
    ordered_keep = None if keep is None else keep[order]
    out = np.zeros(n, dtype=np.float64)
    left = 0
    right = 0
    for k in range(n):
        while ordered_t[left] < ordered_t[k] - half:
            left += 1
        while right < n and ordered_t[right] <= ordered_t[k] + half:
            right += 1
        window = ordered_v[left:right]
        if ordered_keep is not None:
            chosen = window[ordered_keep[left:right]]
            if len(chosen):
                window = chosen
        out[order[k]] = np.median(window)
    return out


def _attack_medians(onset, score, perf):
    n_att = int(onset.max()) + 1
    order = np.argsort(onset, kind="mergesort")
    sorted_onset = onset[order]
    score_s = score[order]
    perf_s = perf[order]
    splits = np.flatnonzero(np.diff(sorted_onset)) + 1
    starts = np.concatenate(([0], splits))
    ends = np.concatenate((splits, [len(onset)]))
    score_m = np.empty(n_att, dtype=np.float64)
    perf_m = np.empty(n_att, dtype=np.float64)
    for start, end in zip(starts, ends):
        attack = int(sorted_onset[start])
        score_m[attack] = np.median(score_s[start:end])
        perf_m[attack] = np.median(perf_s[start:end])
    return score_m, perf_m


def asap_timing_targets(score, perf, onset, beat):
    """Offset and IOI delta, plus a mask for notes the alignment got wrong.

    Offset is the residual after a four-beat median removes the local tempo.
    IOI delta keeps the tempo: performed attack gap minus score attack gap.
    A note more than a beat away from that local tempo is left unsupervised.
    """
    n = len(score)
    offset = np.zeros(n, dtype=np.float32)
    ioi_delta = np.zeros(n, dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    if n == 0 or len(onset) == 0:
        return offset, ioi_delta, ok
    score_m, perf_m = _attack_medians(onset, score, perf)
    drift = perf_m - score_m
    half = 4.0 * beat
    smooth = _local_median(score_m, drift, half)
    keep = np.abs(drift - smooth) <= TIMING_GATE_BEATS * beat
    smooth = _local_median(score_m, drift, half, keep)
    keep = np.abs(drift - smooth) <= TIMING_GATE_BEATS * beat

    residual = (perf - score - smooth[onset]) / beat
    offset[:] = residual.astype(np.float32)
    near_attack = np.abs(perf - perf_m[onset]) <= TIMING_GATE_BEATS * beat
    ok[:] = keep[onset] & near_attack

    n_att = len(score_m)
    if n_att > 1:
        gap = np.zeros(n_att, dtype=np.float64)
        gap[1:] = (np.diff(perf_m) - np.diff(score_m)) / beat
        prev_ok = np.ones(n_att, dtype=bool)
        prev_ok[1:] = keep[:-1]
        ok &= prev_ok[onset]
        ioi_delta[:] = gap[onset].astype(np.float32)
    ok &= np.abs(offset) <= TIMING_GATE_BEATS
    ok &= np.abs(ioi_delta) <= 4.0
    return offset, ioi_delta, ok


def expression_matrix(item, built, structure):
    """Full-piece expressive targets and which timing columns to supervise.

    Velocity is filled in later, per phrase. The returned mask is per note:
    True means offset, IOI, duration, and duration delta are trustworthy.
    """
    n = len(item["pitch"])
    beat = max(float(built["beat_seconds"]), 1e-3)
    if item["snap"]:
        offset = np.asarray(built["offset_beats"], dtype=np.float32)
        score_start = np.asarray(built["grid_start"], dtype=np.float64)
        perf_start = np.asarray(item["start"], dtype=np.float64)
        perf_dur = np.asarray(item["duration"], dtype=np.float64)
        score_dur = np.asarray(built["grid_duration"], dtype=np.float64)
        ioi_delta = np.zeros(n, dtype=np.float32)
        if n > 1:
            ioi_delta[1:] = ((np.diff(perf_start) - np.diff(score_start)) / beat).astype(np.float32)
        timing_ok = np.ones(n, dtype=bool)
    else:
        if structure is None or "perf_duration" not in structure:
            return None
        score_start = np.asarray(item["start"], dtype=np.float64)
        perf_start = np.asarray(item["perf_start"], dtype=np.float64)
        offset, ioi_delta, timing_ok = asap_timing_targets(
            score_start, perf_start, built["onset_id"], beat
        )
        perf_dur = np.asarray(structure["perf_duration"], dtype=np.float64)
        score_dur = np.asarray(item["duration"], dtype=np.float64)
    columns = np.stack(
        [
            np.zeros(n, dtype=np.float32),
            np.nan_to_num(offset).astype(np.float32),
            np.nan_to_num(ioi_delta).astype(np.float32),
            np.nan_to_num(perf_dur / beat).astype(np.float32),
            np.nan_to_num((perf_dur - score_dur) / beat).astype(np.float32),
        ],
        axis=-1,
    )
    return columns, timing_ok


def phrase_tensors(item, start, end):
    built = note_features(
        item["pitch"], item["start"], item["duration"], snap=bool(item["snap"])
    )
    structure = item.get("structure")
    built_expr = expression_matrix(item, built, structure)
    if built_expr is None:
        return None
    expr, timing_ok = built_expr
    timing, voice = assemble_timing(built, structure)
    sl = slice(start, end)
    vel = np.asarray(item["velocity"][sl], dtype=np.float32) / 127.0
    vel = vel - float(vel.mean()) + CENTER_VEL
    expr = expr[sl].copy()
    ok = np.asarray(timing_ok[sl], dtype=bool)
    expr[:, 0] = vel
    # Mismatched notes stay in the phrase for pitch, but their timing targets
    # are zero so a bad alignment cannot dominate the decoder input.
    expr[~ok, 1:] = 0.0
    mask = np.ones_like(expr, dtype=np.float32)
    mask[~ok, 1:] = 0.0
    return {
        "pitch": torch.from_numpy(built["pitch"][sl].astype(np.int64)),
        "timing": torch.from_numpy(np.ascontiguousarray(timing[sl])),
        "voice": torch.from_numpy(np.ascontiguousarray(voice[sl])),
        "expr": torch.from_numpy(np.ascontiguousarray(expr)),
        "mask": torch.from_numpy(np.ascontiguousarray(mask)),
    }


def draw_phrase(asap, giga, source):
    ds = asap if source == "asap" else giga
    for _ in range(8):
        item = ds[random.randrange(len(ds))]
        n = len(item["pitch"])
        if n < PHRASE_MIN:
            continue
        length = random.randint(PHRASE_MIN, min(PHRASE_MAX, n))
        start = random.randint(0, n - length)
        example = phrase_tensors(item, start, start + length)
        if example is not None:
            return example
    return None


def collate(batch):
    max_len = max(x["pitch"].size(0) for x in batch)
    count = len(batch)
    pitch = torch.zeros(count, max_len, dtype=torch.long)
    voice = torch.zeros(count, max_len, dtype=torch.long)
    timing = torch.zeros(count, max_len, N_TIMING_STRUCT)
    expr = torch.zeros(count, max_len, len(EXPR_NAMES))
    mask = torch.zeros(count, max_len, len(EXPR_NAMES))
    pad = torch.zeros(count, max_len, dtype=torch.bool)
    for i, x in enumerate(batch):
        n = x["pitch"].size(0)
        pitch[i, :n] = x["pitch"]
        voice[i, :n] = x["voice"]
        timing[i, :n] = x["timing"]
        expr[i, :n] = x["expr"]
        mask[i, :n] = x["mask"]
        pad[i, :n] = True
    return pitch, timing, voice, expr, mask, pad


def batch_loss(model, batch, device):
    pitch, timing, voice, expr, mask, pad = [t.to(device) for t in batch]
    logits, pred = model(pitch, timing, voice, expr, pad)
    pitch_loss = F.cross_entropy(logits[pad], pitch[pad])
    scale = pred.new_tensor(EXPR_SCALE)
    err = F.huber_loss(pred / scale, expr / scale, delta=1.0, reduction="none")
    weight = mask * pad.unsqueeze(-1)
    expr_loss = (err * weight).sum() / weight.sum().clamp(min=1.0)
    total = NOTE_PENALTY * pitch_loss + expr_loss
    with torch.no_grad():
        acc = (logits[pad].argmax(-1) == pitch[pad]).float().mean()
    return total, float(pitch_loss.detach()), float(expr_loss.detach()), float(acc)


@torch.no_grad()
def evaluate(model, phrases, device):
    model.eval()
    accs = []
    mae = {name: [] for name in EXPR_NAMES}
    greedy_acc = []
    for example in phrases:
        batch = collate([example])
        pitch, timing, voice, expr, mask, pad = [t.to(device) for t in batch]
        logits, pred = model(pitch, timing, voice, expr, pad)
        m = pad[0]
        accs.append(float((logits[0, m].argmax(-1) == pitch[0, m]).float().mean()))
        err = (pred[0, m] - expr[0, m]).abs().cpu().numpy()
        keep = mask[0, m].cpu().numpy() > 0.5
        for i, name in enumerate(EXPR_NAMES):
            take = err[keep[:, i], i]
            if len(take):
                mae[name].append(float(take.mean()))
        if len(greedy_acc) < 8:
            greedy_acc.append(greedy_accuracy(model, pitch, timing, voice, pad))
    model.train()
    summary = {
        "pitch_acc": float(np.mean(accs)),
        "greedy_pitch_acc": float(np.mean(greedy_acc)) if greedy_acc else float("nan"),
    }
    for name in EXPR_NAMES:
        value = float(np.mean(mae[name])) if mae[name] else float("nan")
        if name == "velocity":
            value *= 127.0
        summary[name + "_mae"] = value
    return summary


def _mae_channels(pred, expr, mask):
    err = (pred - expr).abs()
    out = {}
    for i, name in enumerate(EXPR_NAMES):
        take = err[mask[:, i] > 0.5, i]
        value = float(take.mean()) if len(take) else float("nan")
        if name == "velocity":
            value *= 127.0
        out[name] = value
    return out


@torch.no_grad()
def expression_conditions(model, phrases, device):
    """Teacher-forced expression, expression with the history removed, and free run.

    Previous performance controls enter through expr_proj on every position
    after the first. no_history keeps the true previous pitch and zeroes that
    control. generated feeds the decoder its own previous pitch and expression.
    """
    model.eval()
    buckets = {name: [] for name in ("teacher", "no_history", "generated")}
    pitch_acc = {name: [] for name in buckets}
    duration_written = []
    duration_mean = []
    for example in phrases:
        pitch, timing, voice, expr, mask, pad = [t.to(device) for t in collate([example])]
        memory = model.encode(pitch, timing, voice, pad)

        teacher_logits, teacher_pred = model.decode(
            memory,
            _shift_pitch(pitch),
            _shift_expr(expr),
            pad,
        )
        _collect(buckets["teacher"], pitch_acc["teacher"], teacher_logits, teacher_pred, pitch, expr, mask, pad)

        hist_logits, hist_pred = model.decode(
            memory,
            _shift_pitch(pitch),
            torch.zeros_like(expr),
            pad,
        )
        _collect(buckets["no_history"], pitch_acc["no_history"], hist_logits, hist_pred, pitch, expr, mask, pad)

        gen_logits, gen_pred = _free_run(model, memory, pitch, pad)
        _collect(buckets["generated"], pitch_acc["generated"], gen_logits, gen_pred, pitch, expr, mask, pad)

        m = pad[0]
        keep = mask[0, m, 3] > 0.5
        written = timing[0, m, 0]
        played = expr[0, m, 3]
        if int(keep.sum()):
            duration_written.append(float((played[keep] - written[keep]).abs().mean()))
            duration_mean.append(float((played[keep] - played[keep].mean()).abs().mean()))
        # The score already says: on the grid, the written length, no extra gap.
        score_targets = {
            "offset": torch.zeros_like(expr[0, m, 1]),
            "ioi_delta": torch.zeros_like(expr[0, m, 2]),
            "duration": written,
            "duration_delta": torch.zeros_like(expr[0, m, 4]),
        }
        if "score_baselines" not in buckets:
            buckets["score_baselines"] = []
        base_err = {}
        for name, target in score_targets.items():
            col = EXPR_NAMES.index(name)
            take = (expr[0, m, col] - target).abs()
            keep_c = mask[0, m, col] > 0.5
            base_err[name] = float(take[keep_c].mean()) if int(keep_c.sum()) else float("nan")
        vel = expr[0, m, 0]
        keep_v = mask[0, m, 0] > 0.5
        base_err["velocity"] = float((vel[keep_v] - vel[keep_v].mean()).abs().mean()) * 127.0 if int(keep_v.sum()) else float("nan")
        buckets["score_baselines"].append(base_err)
    model.train()
    report = {}
    for name in ("teacher", "no_history", "generated"):
        report[name] = {"pitch_acc": float(np.mean(pitch_acc[name]))}
        for key in EXPR_NAMES:
            report[name][key] = float(np.nanmean([row[key] for row in buckets[name]]))
    report["score_baseline"] = {}
    for key in EXPR_NAMES:
        report["score_baseline"][key] = float(np.nanmean([row[key] for row in buckets["score_baselines"]]))
    report["duration_baseline_phrase_mean"] = float(np.mean(duration_mean)) if duration_mean else float("nan")
    return report


def _shift_pitch(pitch):
    prev = torch.zeros_like(pitch)
    prev[:, 1:] = pitch[:, :-1]
    return prev


def _shift_expr(expr):
    prev = torch.zeros_like(expr)
    prev[:, 1:] = expr[:, :-1]
    return prev


def _collect(bucket, accs, logits, pred, pitch, expr, mask, pad):
    m = pad[0]
    pred_m = pred[0, m]
    expr_m = expr[0, m]
    mask_m = mask[0, m]
    bucket.append(_mae_channels(pred_m, expr_m, mask_m))
    accs.append(float((logits[0, m].argmax(-1) == pitch[0, m]).float().mean()))


@torch.no_grad()
def _free_run(model, memory, pitch, pad):
    prev_pitch = torch.zeros_like(pitch)
    prev_expr = torch.zeros(pitch.size(0), pitch.size(1), len(EXPR_NAMES), device=pitch.device)
    steps = int(pad[0].sum())
    logits_out = torch.zeros(pitch.size(0), pitch.size(1), model.pitch_head.out_features, device=pitch.device)
    pred_out = torch.zeros_like(prev_expr)
    for i in range(steps):
        logits, pred = model.decode(memory, prev_pitch, prev_expr, pad)
        logits_out[:, i] = logits[:, i]
        pred_out[:, i] = pred[:, i]
        if i + 1 < pitch.size(1):
            prev_pitch[:, i + 1] = logits[:, i].argmax(-1)
            prev_expr[:, i + 1] = pred[:, i]
    return logits_out, pred_out


@torch.no_grad()
def greedy_accuracy(model, pitch, timing, voice, pad):
    """Feed the decoder its own previous notes. Pitch still has to come from the encoder."""
    memory = model.encode(pitch, timing, voice, pad)
    prev_pitch = torch.zeros_like(pitch)
    prev_expr = torch.zeros(pitch.size(0), pitch.size(1), len(EXPR_NAMES), device=pitch.device)
    correct = total = 0
    steps = int(pad[0].sum())
    for i in range(steps):
        logits, pred = model.decode(memory, prev_pitch, prev_expr, pad)
        choice = logits[:, i].argmax(-1)
        correct += int((choice == pitch[:, i]).sum())
        total += choice.numel()
        if i + 1 < pitch.size(1):
            prev_pitch[:, i + 1] = choice
            prev_expr[:, i + 1] = pred[:, i]
    return correct / max(total, 1)


def fixed_val_phrases(asap, giga, n=32):
    state = random.getstate()
    random.seed(0)
    phrases = []
    while len(phrases) < n:
        source = "asap" if len(phrases) % 2 == 0 else "giga"
        example = draw_phrase(asap, giga, source)
        if example is not None:
            phrases.append(example)
    random.setstate(state)
    return phrases


def train(device_name):
    random.seed(2)
    torch.manual_seed(2)
    if device_name:
        device = torch.device(device_name)
    else:
        device = get_device()
    print("device", device, "steps", STEPS, "note penalty", NOTE_PENALTY, flush=True)
    asap_train = StructuredVelocityDataset(split="train")
    giga_train = SpanDataset(split="train")
    asap_val = StructuredVelocityDataset(split="val")
    giga_val = SpanDataset(split="val")
    val_phrases = fixed_val_phrases(asap_val, giga_val)
    print(f"val phrases {len(val_phrases)}", flush=True)

    model = PhraseSeq2Seq().to(device)
    print(f"parameters {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if n.endswith("emb.weight")], "lr": EMBED_LR},
            {"params": [p for n, p in model.named_parameters() if not n.endswith("emb.weight")], "lr": LR},
        ]
    )
    best = float("inf")
    model.train()
    running = pitch_run = expr_run = acc_run = 0.0
    window = 0
    for step in range(1, STEPS + 1):
        batch = []
        while len(batch) < BATCH_SIZE:
            source = "asap" if len(batch) < BATCH_SIZE // 2 else "giga"
            example = draw_phrase(asap_train, giga_train, source)
            if example is not None:
                batch.append(example)
        opt.zero_grad()
        loss, pitch_loss, expr_loss, acc = batch_loss(model, collate(batch), device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += float(loss.detach())
        pitch_run += pitch_loss
        expr_run += expr_loss
        acc_run += acc
        window += 1
        if step == 1 or step % 200 == 0:
            print(
                f"step {step}/{STEPS}  loss {running / window:.3f}  "
                f"pitch_ce {pitch_run / window:.3f}  expr {expr_run / window:.3f}  "
                f"pitch_acc {acc_run / window:.3f}",
                flush=True,
            )
            running = pitch_run = expr_run = acc_run = 0.0
            window = 0
        if step % 800 == 0 or step == STEPS:
            metrics = evaluate(model, val_phrases, device)
            score = metrics["velocity_mae"] + 20.0 * (1.0 - metrics["pitch_acc"])
            print(
                f"  val pitch_acc {metrics['pitch_acc']:.3f}  "
                f"greedy {metrics['greedy_pitch_acc']:.3f}  "
                f"vel MAE {metrics['velocity_mae']:.2f}  "
                f"offset {metrics['offset_mae']:.4f}  "
                f"ioi {metrics['ioi_delta_mae']:.4f}  "
                f"dur {metrics['duration_mae']:.3f}  "
                f"dur_delta {metrics['duration_delta_mae']:.3f}",
                flush=True,
            )
            if score < best:
                best = score
                SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "step": step, "metrics": metrics, "note_penalty": NOTE_PENALTY},
                    SAVE_PATH,
                )
                print("  saved", SAVE_PATH, flush=True)


def eval_saved(device_name):
    device = torch.device(device_name or "cpu")
    asap_val = StructuredVelocityDataset(split="val")
    giga_val = SpanDataset(split="val")
    phrases = fixed_val_phrases(asap_val, giga_val)
    model = PhraseSeq2Seq().to(device)
    saved = torch.load(SAVE_PATH, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    report = expression_conditions(model, phrases, device)
    print(json.dumps({"step": saved.get("step"), **report}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="", help="cpu or mps. Empty uses the training default.")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.eval_only:
        eval_saved(args.device or "cpu")
    else:
        train(args.device)
