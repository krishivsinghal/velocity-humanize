"""Corrected ASAP velocity pairs from published (n)ASAP alignments.

The input is every note of the mechanical score MIDI. Targets are human
velocities from the performance MIDI, attached only when the audited match
parser and the audited performed-note lookup agree.

A match id is the MusicXML id plus repeat suffix (``n112-1``). It is never
used as an index into a sorted note array. The note it names is the unique
score-MIDI note whose pitch and written onset agree with that alignment
record. Written time uses the audited match-file conversion. Performed notes
use ``link_notes`` (same pitch, 30 ms). Anything unresolved is left without a
target. Global-stretch pitch-order matching is not used.

  PYTHONPATH=src python src/corrected_align.py
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import mido
import numpy as np
import pandas as pd
import pretty_midi as pmd

from eval_deexpress import (
    LINK_TOL,
    float_to_file_quarters,
    link_notes,
    match_seconds,
    parse_match,
    quarters_to_seconds,
    ticks_to_seconds,
)
from features import assemble_timing, note_features
from preprocess import flatten_notes
from score_structure import match_voices, meter_arrays, xml_path_for

warnings.filterwarnings("ignore")

ROOT = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize")
ASAP = ROOT / "data" / "asap"
NASAP = ROOT / "data" / "nasap"
PROCESSED = ROOT / "data" / "processed"
CACHE = PROCESSED / "velocity_corrected" / "cache_v1"
VERSION = "corrected-asap-v1"

# Same 30 ms window as the audited performed-note lookup.
SCORE_TOL = 0.030
# A file whose published records do not land on this score is a different
# score version. Individual misses inside a resolved file stay unsupervised.
SCORE_RESOLVE_MIN = 0.90
PERF_LINK_MIN = 0.95
CROP_MAX = 256
CROP_MIN = 64


def group_spans(onset_id):
    onset_id = np.asarray(onset_id)
    n = int(len(onset_id))
    if n == 0:
        return []
    change = np.flatnonzero(onset_id[1:] != onset_id[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def prefix_crop(onset_id, limit=CROP_MAX):
    """Leading notes, whole onset groups, hard cap ``limit``."""
    n = int(len(onset_id))
    if n <= limit:
        return 0, n
    total = 0
    end = 0
    for start, stop in group_spans(onset_id):
        glen = stop - start
        if total > 0 and total + glen > limit:
            break
        if total == 0 and glen > limit:
            return 0, limit
        total += glen
        end = stop
    return 0, int(end)


def sample_crop(onset_id):
    """Random 64..256 note window. Onset groups stay intact. Short pieces are whole."""
    n = int(len(onset_id))
    if n <= CROP_MAX:
        return 0, n
    spans = group_spans(onset_id)
    g = len(spans)
    target = int(np.random.randint(CROP_MIN, CROP_MAX + 1))
    gi = int(np.random.randint(0, g))
    gj = gi + 1
    total = spans[gi][1] - spans[gi][0]
    if total > CROP_MAX:
        return spans[gi][0], min(spans[gi][0] + CROP_MAX, n)

    def can_add(forward):
        if forward:
            if gj >= g:
                return False
            return total + (spans[gj][1] - spans[gj][0]) <= CROP_MAX
        if gi <= 0:
            return False
        return total + (spans[gi - 1][1] - spans[gi - 1][0]) <= CROP_MAX

    def add(forward):
        nonlocal gi, gj, total
        if forward:
            total += spans[gj][1] - spans[gj][0]
            gj += 1
        else:
            gi -= 1
            total += spans[gi][1] - spans[gi][0]

    for floor in (target, CROP_MIN):
        guard = 0
        while total < floor and guard < g + 2:
            forward_ok = can_add(True)
            back_ok = can_add(False)
            if not forward_ok and not back_ok:
                break
            if forward_ok and (not back_ok or np.random.rand() < 0.75):
                add(True)
            else:
                add(False)
            guard += 1
    return int(spans[gi][0]), int(spans[gj - 1][1])


def excerpt_bounds(starts, onset_id, lo=30.0, hi=45.0):
    """Score-time span of 30–45 s from the first note, extended only by whole chords."""
    starts = np.asarray(starts, dtype=np.float64)
    n = int(len(starts))
    if n == 0:
        return None
    t0 = float(starts[0])
    if float(starts[-1]) - t0 < lo:
        return None
    end = n
    for i in range(n):
        if float(starts[i]) - t0 >= lo:
            end = i + 1
            break
    for start, stop in group_spans(onset_id):
        if start < end <= stop:
            end = stop
            break
    spans = group_spans(onset_id)
    while end > 1 and float(starts[end - 1]) - t0 > hi:
        group_start = 0
        for start, stop in spans:
            if start <= end - 1 < stop:
                group_start = start
                break
        if group_start == 0:
            break
        if float(starts[group_start - 1]) - t0 < lo:
            break
        end = group_start
    if float(starts[end - 1]) - t0 < lo:
        return None
    return 0, int(end)


def _iter_mido_notes(path, pm):
    """Non-drum notes paired the way pretty_midi pairs them.

    One note-off closes every earlier note-on of that pitch. A note-on and
    note-off on the same tick is not a sounding note. Message indices are kept
    so export can change velocity without moving any other event.
    """
    mid = mido.MidiFile(str(path))
    notes = []
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        open_on = defaultdict(list)
        for mi, msg in enumerate(track):
            abs_tick += int(msg.time)
            if msg.type not in ("note_on", "note_off"):
                continue
            channel = int(getattr(msg, "channel", 0))
            if channel == 9:
                continue
            pitch = int(msg.note)
            if msg.type == "note_on" and int(msg.velocity) > 0:
                open_on[(channel, pitch)].append((abs_tick, mi, int(msg.velocity)))
                continue
            key = (channel, pitch)
            if key not in open_on:
                continue
            pending = open_on[key]
            closing = [item for item in pending if item[0] != abs_tick]
            keeping = [item for item in pending if item[0] == abs_tick]
            for on_tick, on_mi, velocity in closing:
                on_sec = float(ticks_to_seconds(pm, np.array([on_tick]))[0])
                off_sec = float(ticks_to_seconds(pm, np.array([abs_tick]))[0])
                notes.append(
                    {
                        "pitch": pitch,
                        "start": on_sec,
                        "end": off_sec,
                        "duration": float(off_sec - on_sec),
                        "velocity": int(velocity),
                        "track": int(ti),
                        "on_msg": int(on_mi),
                        "off_msg": int(mi),
                        "stable_id": f"{ti:03d}:{on_mi:06d}",
                    }
                )
            if closing and keeping:
                open_on[key] = keeping
            else:
                del open_on[key]
    return notes


def collect_notes(path):
    pm = pmd.PrettyMIDI(str(path))
    raw = _iter_mido_notes(path, pm)
    flat = flatten_notes(pm)
    raw_pitches = [note["pitch"] for note in raw]
    flat_pitches = [int(note.pitch) for note in flat]
    if sorted(raw_pitches) != sorted(flat_pitches):
        raise ValueError(
            f"mido/pretty_midi pitch bags differ for {path} ({len(raw_pitches)} vs {len(flat_pitches)})"
        )
    ordered = sorted(raw, key=lambda note: (note["start"], note["pitch"], note["stable_id"]))
    return pm, ordered


def _unique_score_hit(cands, when):
    """Unique same-pitch score note within SCORE_TOL. None if ambiguous."""
    near = [c for c in cands if abs(c[0] - when) <= SCORE_TOL]
    if not near:
        return None
    near.sort(key=lambda c: abs(c[0] - when))
    if len(near) == 1:
        return near[0][1]
    best = abs(near[0][0] - when)
    second = abs(near[1][0] - when)
    if best + 0.005 < second:
        return near[0][1]
    return None


def align_performance(score_notes, score_pm, perf_pm, parsed):
    """Map match records onto score notes. Returns masks and audit counts."""
    n = len(score_notes)
    available = np.zeros(n, dtype=bool)
    velocity = np.zeros(n, dtype=np.float32)
    by_pitch = defaultdict(list)
    for i, note in enumerate(score_notes):
        by_pitch[int(note["pitch"])].append((float(note["start"]), i))

    matched = parsed["matched"]
    if not matched:
        return available, velocity, {
            "matched": 0,
            "score_candidate": 0,
            "score_unique": 0,
            "supervised": 0,
            "pitch_inconsistent": 0,
            "score_unresolved": 0,
            "perf_unresolved": 0,
            "deletions": len(parsed["deletions"]),
            "insertions": len(parsed["insertions"]),
            "unparsed": int(parsed["unparsed"]),
            "samples": [],
        }

    score_pitch = np.array(
        [-1 if note["score_pitch"] is None else int(note["score_pitch"]) for note in matched],
        dtype=np.int64,
    )
    perf_pitch = np.array([int(note["pitch"]) for note in matched], dtype=np.int64)
    onset_q = np.array([note["on_q"] for note in matched], dtype=np.float64)
    onset_quarters = float_to_file_quarters(onset_q, parsed["time_signatures"])
    written_sec = quarters_to_seconds(score_pm, onset_quarters)
    perf_sec = match_seconds(
        np.array([note["on_tick"] for note in matched], dtype=np.float64),
        parsed["clock"],
        parsed["rate"],
    )
    flat = flatten_notes(perf_pm)
    if flat:
        flat_pitch = np.array([int(note.pitch) for note in flat], dtype=np.int64)
        flat_start = np.array([float(note.start) for note in flat], dtype=np.float64)
        links = link_notes(perf_pitch, perf_sec, flat_pitch, flat_start, tol=LINK_TOL)
    else:
        links = np.full(len(matched), -1, dtype=np.int64)

    claimed = {}
    pending = []
    pitch_inconsistent = 0
    score_unresolved = 0
    for row, rec in enumerate(matched):
        sp = score_pitch[row]
        if sp < 0 or sp != int(perf_pitch[row]):
            pitch_inconsistent += 1
            continue
        hit = _unique_score_hit(by_pitch.get(int(sp), []), float(written_sec[row]))
        if hit is None:
            score_unresolved += 1
            continue
        pending.append((row, hit))
        claimed.setdefault(hit, []).append(row)

    ambiguous = {idx for idx, rows in claimed.items() if len(rows) > 1}
    supervised = 0
    perf_unresolved = 0
    score_unique = 0
    samples = []
    for row, hit in pending:
        if hit in ambiguous:
            score_unresolved += 1
            continue
        score_unique += 1
        link = int(links[row])
        if link < 0:
            perf_unresolved += 1
            continue
        perf_note = flat[link]
        if int(perf_note.pitch) != int(perf_pitch[row]):
            pitch_inconsistent += 1
            continue
        if abs(float(perf_note.start) - float(perf_sec[row])) > LINK_TOL + 1e-6:
            perf_unresolved += 1
            continue
        vel = int(perf_note.velocity)
        if not 1 <= vel <= 127:
            perf_unresolved += 1
            continue
        available[hit] = True
        velocity[hit] = float(vel)
        supervised += 1
        samples.append(
            {
                "row": int(row),
                "link": link,
                "pitch": int(perf_pitch[row]),
                "score_pitch": int(score_pitch[row]),
                "perf_start": float(perf_note.start),
                "match_perf_sec": float(perf_sec[row]),
                "hit": int(hit),
                "score_start": float(score_notes[hit]["start"]),
                "written_sec": float(written_sec[row]),
                "velocity": vel,
                "perf_duration": float(perf_note.end - perf_note.start),
                "sid": matched[row]["sid"],
            }
        )
    denom = int(np.sum((score_pitch >= 0) & (score_pitch == perf_pitch)))
    stats = {
        "matched": len(matched),
        "score_candidate": denom,
        "score_unique": score_unique,
        "supervised": supervised,
        "pitch_inconsistent": pitch_inconsistent,
        "score_unresolved": score_unresolved,
        "perf_unresolved": perf_unresolved,
        "deletions": len(parsed["deletions"]),
        "insertions": len(parsed["insertions"]),
        "unparsed": int(parsed["unparsed"]),
        "samples": samples,
    }
    return available, velocity, stats


def _score_features(score_notes, score_pm, ann, midi_score):
    ns = [
        SimpleNamespace(
            pitch=int(note["pitch"]),
            start=float(note["start"]),
            end=float(note["start"]) + float(note["duration"]),
            duration=float(note["duration"]),
        )
        for note in score_notes
    ]
    xml = xml_path_for(midi_score)
    voice, voice_info = match_voices(score_pm, ns, xml)
    bar, known, beats, denom = meter_arrays(ns, ann or {})
    pitch = np.array([note["pitch"] for note in score_notes], dtype=np.int64)
    start = np.array([note["start"] for note in score_notes], dtype=np.float64)
    duration = np.array([max(float(note["duration"]), 0.0) for note in score_notes], dtype=np.float64)
    built = note_features(pitch, start, duration, snap=False)
    structure = {
        "bar_phase": bar,
        "meter_known": known,
        "beats_per_bar": beats,
        "beat_denom": denom,
        "voice": voice.astype(np.int64),
    }
    timing, voice_out = assemble_timing(built, structure)
    onset = np.asarray(built["onset_id"], dtype=np.int64)
    for start_i, stop_i in group_spans(onset):
        if int(onset[start_i]) != int(onset[stop_i - 1]) or np.any(onset[start_i:stop_i] != onset[start_i]):
            raise ValueError("onset group is not contiguous in canonical order")
    return timing, voice_out.astype(np.int16), onset, voice_info


def _ann_for_score(annotations, performances):
    """Score meter only. Performance beat times are not features."""
    for rel in performances:
        ann = annotations.get(rel) or {}
        downs = ann.get("midi_score_downbeats") or []
        sigs = ann.get("midi_score_time_signatures") or {}
        if len(downs) >= 2 and sigs:
            return {
                "midi_score_downbeats": downs,
                "midi_score_time_signatures": sigs,
            }
    return {}


def _reuse_features(midi_score, feature_files):
    """Reload score features already written for this score. None if absent."""
    path = feature_files.get(midi_score)
    if path is None or not path.exists():
        return None
    try:
        with np.load(path) as prev:
            voice = np.array(prev["voice"])
            timing = np.array(prev["timing"])
            onset = np.array(prev["onset_id"])
    except Exception:
        return None
    if timing.ndim != 2 or len(voice) != len(timing) or len(onset) != len(timing):
        return None
    return timing, voice, onset, {"kept": bool(np.any(voice > 0))}


def _save_perf(path, score_notes, timing, voice, onset, available, velocity):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pitch=np.array([note["pitch"] for note in score_notes], dtype=np.int16),
        start=np.array([note["start"] for note in score_notes], dtype=np.float64),
        duration=np.array([note["duration"] for note in score_notes], dtype=np.float32),
        timing=np.asarray(timing, dtype=np.float32),
        voice=np.asarray(voice, dtype=np.int16),
        onset_id=np.asarray(onset, dtype=np.int32),
        velocity=np.asarray(velocity, dtype=np.float32),
        available=np.asarray(available, dtype=bool),
        stable_id=np.array([note["stable_id"] for note in score_notes]),
        src_track=np.array([note["track"] for note in score_notes], dtype=np.int16),
        src_on=np.array([note["on_msg"] for note in score_notes], dtype=np.int32),
        src_off=np.array([note["off_msg"] for note in score_notes], dtype=np.int32),
    )


def build_cache(limit=None, force=False):
    meta_path = CACHE / "index.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("version") == VERSION:
            print(f"reusing {meta_path}", flush=True)
            return meta

    index = pd.read_csv(PROCESSED / "index.csv")
    with open(ASAP / "asap_annotations.json") as handle:
        annotations = json.load(handle)

    pre = {}
    for split, frame in index.groupby("split"):
        pre[str(split)] = int(frame.groupby(["composer", "title"]).ngroups)
    print("pre-filter compositions", pre, flush=True)

    exclusions = []
    performances = []
    audit_pool = {"train": [], "val": [], "test": []}
    feature_cache = {}
    note_cache = {}
    score_perf_paths = defaultdict(list)
    for row in index.itertuples(index=False):
        score_perf_paths[row.midi_score].append(row.midi_performance)
    score_ann = {
        score: _ann_for_score(annotations, paths) for score, paths in score_perf_paths.items()
    }

    rows = list(index.itertuples(index=False))
    if limit is not None:
        rows = rows[:limit]
    feature_files = {}
    for row in rows:
        saved = CACHE / "perf" / row.file
        if row.midi_score not in feature_files and saved.exists():
            feature_files[row.midi_score] = saved
    for n_done, row in enumerate(rows, start=1):
        match_path = NASAP / Path(row.midi_performance).with_suffix(".match")
        rec_base = {
            "file": row.file,
            "split": row.split,
            "composer": row.composer,
            "title": row.title,
            "midi_score": row.midi_score,
            "midi_performance": row.midi_performance,
        }
        if not match_path.exists():
            exclusions.append({**rec_base, "reason": "no_match_file"})
            print("EXCLUDE", row.file, "no_match_file", flush=True)
            continue
        score_path = ASAP / row.midi_score
        perf_path = ASAP / row.midi_performance
        if not score_path.exists() or not perf_path.exists():
            exclusions.append({**rec_base, "reason": "missing_midi"})
            print("EXCLUDE", row.file, "missing_midi", flush=True)
            continue
        try:
            parsed = parse_match(match_path)
            if row.midi_score not in note_cache:
                score_pm, score_notes = collect_notes(score_path)
                note_cache[row.midi_score] = (score_pm, score_notes)
                reused = _reuse_features(row.midi_score, feature_files)
                if reused is None:
                    feature_cache[row.midi_score] = _score_features(
                        score_notes, score_pm, score_ann.get(row.midi_score), row.midi_score
                    )
                else:
                    feature_cache[row.midi_score] = reused
            score_pm, score_notes = note_cache[row.midi_score]
            perf_pm = pmd.PrettyMIDI(str(perf_path))
            available, velocity, stats = align_performance(score_notes, score_pm, perf_pm, parsed)
        except Exception as exc:
            exclusions.append({**rec_base, "reason": "parse_error", "detail": str(exc)})
            print("EXCLUDE", row.file, "parse_error", exc, flush=True)
            continue
        samples = stats.pop("samples", [])
        denom = max(int(stats["score_candidate"]), 1)
        score_rate = stats["score_unique"] / denom if stats["score_candidate"] else 0.0
        link_rate = (
            stats["supervised"] / stats["score_unique"] if stats["score_unique"] else 0.0
        )
        stats["score_rate"] = score_rate
        stats["link_rate"] = link_rate
        if stats["matched"] == 0 or stats["score_candidate"] == 0:
            exclusions.append({**rec_base, "reason": "no_matched_records", **stats})
            print("EXCLUDE", row.file, "no_matched_records", flush=True)
            continue
        if score_rate < SCORE_RESOLVE_MIN:
            exclusions.append({**rec_base, "reason": "score_unresolved", **stats})
            print(
                f"EXCLUDE {row.file} score_unresolved rate {score_rate:.3f} "
                f"link {link_rate:.3f}",
                flush=True,
            )
            continue
        if link_rate < PERF_LINK_MIN:
            exclusions.append({**rec_base, "reason": "perf_link_unresolved", **stats})
            print(f"EXCLUDE {row.file} perf_link_unresolved rate {link_rate:.3f}", flush=True)
            continue
        if stats["supervised"] == 0:
            exclusions.append({**rec_base, "reason": "no_targets", **stats})
            print("EXCLUDE", row.file, "no_targets", flush=True)
            continue
        timing, voice, onset, voice_info = feature_cache[row.midi_score]
        out = CACHE / "perf" / row.file
        _save_perf(out, score_notes, timing, voice, onset, available, velocity)
        performances.append(
            {
                **rec_base,
                "npz": str(out.relative_to(ROOT)),
                "n_notes": int(len(score_notes)),
                "n_targets": int(stats["supervised"]),
                "n_unmatched": int(len(score_notes) - stats["supervised"]),
                "score_rate": score_rate,
                "link_rate": link_rate,
                "deletions": stats["deletions"],
                "insertions": stats["insertions"],
                "pitch_inconsistent": stats["pitch_inconsistent"],
                "score_unresolved": stats["score_unresolved"],
                "perf_unresolved": stats["perf_unresolved"],
                "voice_kept": bool(voice_info.get("kept")),
                "meter_frac": float(np.mean(timing[:, 5] > 0.5)) if len(timing) else 0.0,
            }
        )
        if samples and row.split in audit_pool:
            mid = samples[len(samples) // 2]
            audit_pool[row.split].append({"file": row.file, "midi_performance": row.midi_performance, **mid})
        if n_done % 25 == 0:
            print(f"aligned {n_done}/{len(rows)} kept {len(performances)} excluded {len(exclusions)}", flush=True)

    post = {}
    post_perf = {}
    for split in ("train", "val", "test"):
        kept = [p for p in performances if p["split"] == split]
        post_perf[split] = len(kept)
        post[split] = len({(p["composer"], p["title"]) for p in kept})
    train_vel = []
    for perf in performances:
        if perf["split"] != "train":
            continue
        data = np.load(ROOT / perf["npz"])
        vel = np.asarray(data["velocity"], dtype=np.float64)
        ok = np.asarray(data["available"], dtype=bool)
        if ok.any():
            train_vel.append(vel[ok])
    all_train = np.concatenate(train_vel) if train_vel else np.array([64.0])
    mu = float(all_train.mean())
    sigma = float(all_train.std())
    if not np.isfinite(sigma) or sigma < 1.0:
        sigma = 1.0

    digest = hashlib.sha256()
    digest.update(VERSION.encode())
    digest.update(f"link_tol={LINK_TOL:.3f} score_tol={SCORE_TOL:.3f}".encode())
    for perf in sorted(performances, key=lambda p: p["file"]):
        digest.update(
            f"{perf['split']}|{perf['midi_performance']}|{perf['n_notes']}|{perf['n_targets']}\n".encode()
        )
    for _, row in index.sort_values("file").iterrows():
        digest.update(f"split|{row.file}|{row.split}|{row.composer}|{row.title}\n".encode())

    spot = []
    for split, want in (("train", 70), ("val", 30)):
        pool = audit_pool[split]
        if not pool:
            continue
        pick = np.linspace(0, len(pool) - 1, min(want, len(pool)), dtype=int)
        for i in pick:
            spot.append(pool[int(i)])

    reason_counts = defaultdict(int)
    for item in exclusions:
        reason_counts[item["reason"]] += 1
    meta = {
        "version": VERSION if limit is None else "probe",
        "fingerprint": digest.hexdigest(),
        "link_tol_sec": LINK_TOL,
        "score_tol_sec": SCORE_TOL,
        "score_resolve_min": SCORE_RESOLVE_MIN,
        "perf_link_min": PERF_LINK_MIN,
        "prefilter_compositions": pre,
        "postfilter_compositions": post,
        "postfilter_performances": post_perf,
        "excluded_performances": len(exclusions),
        "exclusion_reasons": dict(reason_counts),
        "exclusions": exclusions,
        "performances": performances,
        "mu_train": mu,
        "sigma_train": sigma,
        "n_train_targets": int(all_train.size if train_vel else 0),
        "unmatched_targets": int(sum(p["n_unmatched"] for p in performances)),
        "supervised_targets": int(sum(p["n_targets"] for p in performances)),
        "spot_checks": spot,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                "prefilter_compositions": pre,
                "postfilter_compositions": post,
                "postfilter_performances": post_perf,
                "excluded_performances": len(exclusions),
                "exclusion_reasons": dict(reason_counts),
                "mu_train": mu,
                "sigma_train": sigma,
                "fingerprint": meta["fingerprint"],
            },
            indent=2,
        ),
        flush=True,
    )
    return meta


def load_meta():
    return json.loads((CACHE / "index.json").read_text())


def load_perf(record):
    data = np.load(ROOT / record["npz"])
    return {key: data[key] for key in data.files}


def spot_check(meta, n=100):
    """Re-run the audited lookup on 100 saved records. Raises if any disagree."""
    samples = meta["spot_checks"]
    if len(samples) < min(n, 1):
        raise RuntimeError("no spot-check records were saved")
    use = samples[:n]
    by_perf = defaultdict(list)
    for sample in use:
        by_perf[sample["midi_performance"]].append(sample)
    checked = 0
    for rel, group in by_perf.items():
        parsed = parse_match(NASAP / Path(rel).with_suffix(".match"))
        perf_pm = pmd.PrettyMIDI(str(ASAP / rel))
        flat = flatten_notes(perf_pm)
        matched = parsed["matched"]
        pitch = np.array([int(note["pitch"]) for note in matched], dtype=np.int64)
        perf_sec = match_seconds(
            np.array([note["on_tick"] for note in matched], dtype=np.float64),
            parsed["clock"],
            parsed["rate"],
        )
        flat_pitch = np.array([int(note.pitch) for note in flat], dtype=np.int64)
        flat_start = np.array([float(note.start) for note in flat], dtype=np.float64)
        links = link_notes(pitch, perf_sec, flat_pitch, flat_start, tol=LINK_TOL)
        for sample in group:
            row = int(sample["row"])
            link = int(links[row])
            if link < 0 or link != int(sample["link"]):
                raise AssertionError(f"{rel} row {row} audited link {link} != stored {sample['link']}")
            note = flat[link]
            if int(note.pitch) != int(sample["pitch"]) or int(note.pitch) != int(sample["score_pitch"]):
                raise AssertionError(f"{rel} row {row} pitch mismatch")
            if abs(float(note.start) - float(sample["match_perf_sec"])) > LINK_TOL:
                raise AssertionError(f"{rel} row {row} performed onset off by more than 30 ms")
            if abs(float(note.start) - float(sample["perf_start"])) > 1e-4:
                raise AssertionError(f"{rel} row {row} stored onset disagrees with lookup")
            if int(note.velocity) != int(sample["velocity"]):
                raise AssertionError(f"{rel} row {row} velocity disagrees with lookup")
            if abs(float(sample["score_start"]) - float(sample["written_sec"])) > SCORE_TOL:
                raise AssertionError(f"{rel} row {row} score onset mapping is outside tolerance")
            checked += 1
    splits = sorted({sample.get("file", "")[:1] for sample in use})
    train = sum(1 for sample in use if sample["file"] and _split_of(meta, sample["file"]) == "train")
    val = sum(1 for sample in use if _split_of(meta, sample["file"]) == "val")
    print(f"spot-check passed {checked} records train {train} val {val}", flush=True)
    return {"checked": checked, "train": train, "val": val, "splits_seen": splits}


def _split_of(meta, file_name):
    for perf in meta["performances"]:
        if perf["file"] == file_name:
            return perf["split"]
    return ""


def select_panel_and_excerpts(meta):
    """Fixed validation panel and demo excerpts. Chosen from the filtered val split."""
    val = [p for p in meta["performances"] if p["split"] == "val"]
    by_comp = defaultdict(list)
    for perf in val:
        by_comp[(perf["composer"], perf["title"])].append(perf)
    compositions = sorted(by_comp)
    panel = []
    for composer, title in compositions:
        if len(panel) >= 8:
            break
        perf = sorted(by_comp[(composer, title)], key=lambda p: p["midi_performance"])[0]
        arrays = load_perf(perf)
        start, end = prefix_crop(arrays["onset_id"], CROP_MAX)
        panel.append(
            {
                "composer": composer,
                "title": title,
                "file": perf["file"],
                "midi_performance": perf["midi_performance"],
                "midi_score": perf["midi_score"],
                "start": int(start),
                "end": int(end),
                "stable_ids": [str(x) for x in arrays["stable_id"][start:end]],
            }
        )
    excerpts = []
    used_composers = []
    for composer, title in compositions:
        if len(excerpts) == 1 and composer == used_composers[0]:
            continue
        perf = sorted(by_comp[(composer, title)], key=lambda p: p["midi_performance"])[0]
        arrays = load_perf(perf)
        bounds = excerpt_bounds(arrays["start"], arrays["onset_id"])
        if bounds is None:
            continue
        start, end = bounds
        excerpts.append(
            {
                "composer": composer,
                "title": title,
                "file": perf["file"],
                "midi_performance": perf["midi_performance"],
                "midi_score": perf["midi_score"],
                "start": int(start),
                "end": int(end),
                "t0": float(arrays["start"][start]),
                "t1": float(arrays["start"][end - 1]),
                "duration_sec": float(arrays["start"][end - 1] - arrays["start"][start]),
                "n_notes": int(end - start),
                "stable_ids": [str(x) for x in arrays["stable_id"][start:end]],
                "unmatched_in_excerpt": int((~arrays["available"][start:end]).sum()),
            }
        )
        used_composers.append(composer)
        if len(excerpts) == 2:
            break
    fixed = []
    for perf in val:
        arrays = load_perf(perf)
        start, end = prefix_crop(arrays["onset_id"], CROP_MAX)
        fixed.append(
            {
                "composer": perf["composer"],
                "title": perf["title"],
                "file": perf["file"],
                "midi_performance": perf["midi_performance"],
                "start": int(start),
                "end": int(end),
            }
        )
    return {"panel": panel, "excerpts": excerpts, "val_crops": fixed}


def apply_note_velocities(src, dst, updates):
    """Copy a MIDI file and change only the listed note-on velocities.

    ``updates`` maps (track, note_on message index) to a MIDI velocity.
    Pitch, delta time, note-off, and every other event stay as they were.
    """
    mid = mido.MidiFile(str(src))
    for ti, track in enumerate(mid.tracks):
        for mi, msg in enumerate(track):
            key = (ti, mi)
            if key not in updates:
                continue
            if msg.type != "note_on" or int(msg.velocity) <= 0:
                raise ValueError(f"update {key} is not a note-on in {src}")
            msg.velocity = int(updates[key])
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(dst))


def write_excerpt(src, dst, notes, velocities, start, end):
    """Excerpt that keeps the original tempo map and the selected notes' timing."""
    keep_on = {}
    keep_off = set()
    for i in range(int(start), int(end)):
        note = notes[i]
        key = (int(note["track"]), int(note["on_msg"]))
        keep_on[key] = int(velocities[i])
        keep_off.add((int(note["track"]), int(note["off_msg"])))
    mid = mido.MidiFile(str(src))
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        kept = []
        for mi, msg in enumerate(track):
            abs_tick += int(msg.time)
            key = (ti, mi)
            if msg.type in ("note_on", "note_off"):
                if msg.type == "note_on" and int(msg.velocity) > 0:
                    if key not in keep_on:
                        continue
                    msg = msg.copy(velocity=int(keep_on[key]))
                else:
                    if key not in keep_off:
                        continue
                    msg = msg.copy()
            else:
                msg = msg.copy()
            kept.append((abs_tick, msg))
        new_track = mido.MidiTrack()
        prev = 0
        for abs_tick, msg in kept:
            msg.time = int(abs_tick - prev)
            prev = abs_tick
            new_track.append(msg)
        if not new_track or new_track[-1].type != "end_of_track":
            new_track.append(mido.MetaMessage("end_of_track", time=0))
        out.tracks.append(new_track)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out.save(str(dst))


def window_spans(onset_id, limit=CROP_MAX):
    """Non-overlapping chord-aligned windows covering every note."""
    spans = group_spans(onset_id)
    if not spans:
        return []
    windows = []
    cur_s = None
    cur_e = None
    for start, stop in spans:
        glen = stop - start
        if glen > limit:
            if cur_s is not None:
                windows.append((int(cur_s), int(cur_e)))
                cur_s = None
            pos = start
            while pos < stop:
                windows.append((int(pos), int(min(pos + limit, stop))))
                pos += limit
            continue
        if cur_s is None:
            cur_s, cur_e = start, stop
            continue
        if stop - cur_s <= limit:
            cur_e = stop
            continue
        windows.append((int(cur_s), int(cur_e)))
        cur_s, cur_e = start, stop
    if cur_s is not None:
        windows.append((int(cur_s), int(cur_e)))
    return windows


if __name__ == "__main__":
    import sys

    cap = int(sys.argv[1]) if len(sys.argv) > 1 else None
    meta = build_cache(limit=cap, force=True)
    if cap is None:
        spot_check(meta)
