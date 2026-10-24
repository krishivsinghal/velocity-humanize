"""
Decide whether a performance can be snapped to one grid without moving a note
onto a different beat.

A stable piece has one subdivision length from start to end. Rubato and
ritardando show up as that length drifting. Microtiming around a steady beat
is kept: that is the timing offset the model should learn.

  python src/grid_stability.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

GIGAMIDI = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/gigamidi")
# Slowest local pulse over fastest local pulse. 1.12 is a 12% tempo change
# across the piece, enough to call it a ritardando or accelerando.
MAX_TEMPO_RATIO = 1.12
# How much the pulse wobbles from window to window. Above this is rubato.
MAX_TEMPO_CV = 0.03
# Share of gaps that are integer multiples of one subdivision.
MIN_EXPLAINED = 0.85


def cluster_onsets(starts, tol=0.035):
    if len(starts) == 0:
        return np.array([], dtype=np.float64)
    ordered = np.sort(starts.astype(np.float64))
    keep = [ordered[0]]
    for t in ordered[1:]:
        if t - keep[-1] >= tol:
            keep.append(t)
    return np.asarray(keep, dtype=np.float64)


def candidate_quanta(onsets):
    ioi = np.diff(onsets)
    ioi = ioi[(ioi >= 0.05) & (ioi <= 1.5)]
    if len(ioi) < 12:
        return []
    short = ioi[ioi <= np.percentile(ioi, 60)]
    hist, edges = np.histogram(short, bins=np.arange(0.05, 1.05, 0.005))
    if len(hist) == 0 or hist.max() < 6:
        return []
    peak = int(hist.argmax())
    q = float((edges[peak] + edges[peak + 1]) / 2)
    # The histogram peak can be an eighth, a quarter, or a triplet.
    return [q, q / 2, q * 2, q * 2 / 3, q * 3 / 2]


def explain_rate(onsets, q):
    ioi = np.diff(onsets)
    ioi = ioi[ioi >= 0.04]
    if len(ioi) == 0 or q < 0.04 or q > 1.5:
        return 0.0
    steps = np.round(ioi / q)
    rel = np.abs(ioi - steps * q) / q
    good = (steps >= 1) & (steps <= 8) & (rel <= 0.25)
    return float(good.mean())


def local_pulses(onsets, q):
    win = max(4.0, q * 24)
    hop = win / 2
    periods = []
    t = float(onsets[0])
    end = float(onsets[-1])
    while t + win * 0.5 <= end:
        seg = onsets[(onsets >= t) & (onsets < t + win)]
        if len(seg) >= 8:
            ioi = np.diff(seg)
            steps = np.round(ioi / q)
            ok = (steps >= 1) & (steps <= 6)
            if int(ok.sum()) >= 4:
                pulse = (ioi[ok] / steps[ok])
                pulse = pulse[(pulse > 0.7 * q) & (pulse < 1.3 * q)]
                if len(pulse) >= 4:
                    periods.append(float(np.median(pulse)))
        t += hop
    return np.asarray(periods, dtype=np.float64)


def score_piece(starts):
    onsets = cluster_onsets(starts)
    empty = {
        "n_onsets": int(len(onsets)),
        "quantum_sec": np.nan,
        "explained": 0.0,
        "tempo_ratio": np.nan,
        "tempo_cv": np.nan,
        "grid_stable": False,
    }
    if len(onsets) < 16:
        return empty

    best = None
    for q in candidate_quanta(onsets):
        explained = explain_rate(onsets, q)
        if best is None or explained > best[0]:
            best = (explained, q)
    if best is None:
        return empty
    explained, q = best
    pulses = local_pulses(onsets, q)
    if len(pulses) < 3:
        ratio, cv = np.nan, np.nan
    else:
        ratio = float(pulses.max() / max(pulses.min(), 1e-6))
        cv = float(pulses.std() / max(pulses.mean(), 1e-6))
    stable = (
        explained >= MIN_EXPLAINED
        and ratio == ratio
        and ratio <= MAX_TEMPO_RATIO
        and cv <= MAX_TEMPO_CV
    )
    return {
        "n_onsets": int(len(onsets)),
        "quantum_sec": q,
        "explained": explained,
        "tempo_ratio": ratio,
        "tempo_cv": cv,
        "grid_stable": bool(stable),
    }


MAX_SNAP_P95 = 0.30
BLOCK_SEC = 6.0


def fit_grid(onsets, q_hint):
    """Fit one grid. Returns quantum, p95 error as a fraction of that quantum, explain rate."""
    if len(onsets) < 8 or q_hint is None or q_hint <= 0:
        return None
    q = float(q_hint)
    t0 = float(onsets[0])
    idx = np.round((onsets - t0) / q)
    for _ in range(4):
        if np.unique(idx).size < 2:
            return None
        design = np.column_stack([np.ones(len(onsets)), idx])
        t0, q = np.linalg.lstsq(design, onsets, rcond=None)[0]
        if not (0.04 < q < 1.5):
            return None
        idx = np.round((onsets - t0) / q)
    err = np.abs(onsets - (t0 + idx * q)) / q
    return float(q), float(np.percentile(err, 95)), explain_rate(onsets, float(q))


def span_ok(onsets, q_hint):
    """True when the gaps quantize cleanly and the pulse does not drift."""
    if len(onsets) < 8 or q_hint is None:
        return False
    if explain_rate(onsets, q_hint) < MIN_EXPLAINED:
        return False
    pulses = local_pulses(onsets, q_hint)
    if len(pulses) < 2:
        return True
    ratio = float(pulses.max() / max(pulses.min(), 1e-6))
    cv = float(pulses.std() / max(pulses.mean(), 1e-6))
    return ratio <= MAX_TEMPO_RATIO and cv <= MAX_TEMPO_CV


def stable_spans(starts):
    """Longest stretches whose pulse stays steady.

    A ritardando used to reject the whole file. A 6 second block is kept when
    its gaps fall on one subdivision, and neighboring blocks are merged only
    while the combined stretch still has a steady pulse.
    """
    onsets = cluster_onsets(starts)
    if len(onsets) < 16:
        return []
    quanta = candidate_quanta(onsets)
    if not quanta:
        return []
    q_hint = max(quanta, key=lambda q: explain_rate(onsets, q))
    edges = list(np.arange(onsets[0], onsets[-1], BLOCK_SEC))
    if not edges:
        return []
    blocks = []
    for edge in edges:
        seg = onsets[(onsets >= edge) & (onsets < edge + BLOCK_SEC)]
        blocks.append(span_ok(seg, q_hint))

    spans = []
    i = 0
    n = len(blocks)
    while i < n:
        if not blocks[i]:
            i += 1
            continue
        j = i + 1
        while j < n and blocks[j]:
            seg = onsets[(onsets >= edges[i]) & (onsets < edges[j] + BLOCK_SEC)]
            if not span_ok(seg, q_hint):
                break
            j += 1
        t0 = float(edges[i])
        t1 = float(min(edges[j - 1] + BLOCK_SEC, onsets[-1] + 1e-3))
        if t1 - t0 >= 4.0:
            spans.append((t0, t1))
        if j < n and blocks[j]:
            i = j
        else:
            i = j + 1
    return spans


def main_spans():
    index = pd.read_csv(GIGAMIDI / "index.csv")
    rows = []
    covered = 0
    total = 0
    for i, row in index.iterrows():
        starts = np.load(GIGAMIDI / "pairs" / row["file"])["start"]
        total += len(starts)
        for span_i, (t0, t1) in enumerate(stable_spans(starts)):
            n_notes = int(np.sum((starts >= t0) & (starts < t1)))
            if n_notes < 32:
                continue
            covered += n_notes
            rows.append(
                {
                    "file": row["file"],
                    "span": span_i,
                    "t0": t0,
                    "t1": t1,
                    "n_notes": n_notes,
                    "composer": row["composer"],
                    "title": row["title"],
                    "split": row["split"],
                }
            )
        if (i + 1) % 2000 == 0:
            print(f"scanned {i + 1} spans {len(rows)} notes {covered}", flush=True)
    spans = pd.DataFrame(rows)
    spans.to_csv(GIGAMIDI / "index_spans.csv", index=False)
    print(f"spans {len(spans)} notes {covered} of {total}")


def main():
    index = pd.read_csv(GIGAMIDI / "index.csv")
    rows = []
    for i, row in index.iterrows():
        starts = np.load(GIGAMIDI / "pairs" / row["file"])["start"]
        scored = score_piece(starts)
        scored["file"] = row["file"]
        rows.append(scored)
        if (i + 1) % 2000 == 0:
            kept = sum(r["grid_stable"] for r in rows)
            print(f"scored {i + 1} kept {kept}", flush=True)
    metrics = pd.DataFrame(rows)
    merged = index.merge(metrics, on="file", how="left")
    merged.to_csv(GIGAMIDI / "index_grid.csv", index=False)
    stable = merged[merged["grid_stable"] == True]
    print(f"kept {len(stable)} of {len(merged)}")
    print(f"notes {int(stable['n_notes'].sum()) if len(stable) else 0}")
    print(stable["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
