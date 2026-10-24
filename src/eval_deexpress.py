"""Does de-expression recover ASAP's written score?

ASAP already pairs a performance with a mechanical score. (n)ASAP adds a
note-to-note alignment. GigaMIDI has performances only, so its mechanical
side is made by setting velocity to 80 and snapping each note onto a beat
grid, without dropping or reordering notes.

This script applies that snap to ASAP performances and scores it against the
written onsets in the (n)ASAP alignment. It does not rebuild training pairs
and it does not re-filter GigaMIDI.

Two onset tests, same notes:

  beat map    The annotated performance beats are known. Each onset is
              converted to beats with that map, then snapped with
              round(q * b) / q. This isolates the grid, not the tempo tracker.
  pipeline    pulse() and snap_notes(), the same functions the GigaMIDI
              feature path uses. The beat map is only the ruler afterwards.

A hit is a snapped onset within 0.02 beats of the written onset. That is
tighter than the gap between adjacent twelfth-notes, so a neighboring cell
does not count. Errors are also reported in beats before any snap.

  PYTHONPATH=src python src/eval_deexpress.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi as pmd

from features import pulse, snap_notes
from preprocess import flatten_notes

warnings.filterwarnings("ignore")

BASE = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/asap")
NASAP = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/nasap")
OUT_DIR = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/processed/checks")

# Subdivisions per beat. 1 is the beat itself, 2 eighths, 3 triplets, 4 sixteenths.
CANDIDATE_Q = (1, 2, 3, 4, 6, 8, 12)
HIT_BEATS = 0.02
LINK_TOL = 0.03
# Insertions plus deletions above this, relative to score notes, stand in for
# a performance of a different score version than the one we aligned.
VERSION_INDEL = 0.05
TEMPO_RATIO = 1.12

PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ALTER = {"n": 0, "#": 1, "b": -1, "x": 2, "bb": -2}

SNOTE_RE = re.compile(
    r"^snote\((?P<sid>[^,]+),\[(?P<step>[A-Ga-g]),(?P<alter>[^\]]+)\],(?P<oct>-?\d+),"
    r"[^,]+,[^,]+,[^,]+,(?P<on>-?\d+(?:\.\d+)?),(?P<off>-?\d+(?:\.\d+)?),"
    r"\[(?P<ann>[^\]]*)\]\)-(?P<rhs>.*)$"
)
NOTE_NUM_RE = re.compile(
    r"^note\((?P<nid>[^,]+),(?P<pitch>\d+),(?P<on>\d+),(?P<off>\d+),(?P<vel>\d+),\d+,\d+\)$"
)
NOTE_SPELL_RE = re.compile(
    r"^note\((?P<nid>[^,]+),\[(?P<step>[A-Ga-g]),(?P<alter>[^\]]+)\],(?P<oct>-?\d+),"
    r"(?P<on>\d+),(?P<off>\d+),\d+,(?P<vel>\d+)\)$"
)
INS_NUM_RE = re.compile(
    r"^insertion-note\((?P<nid>[^,]+),(?P<pitch>\d+),(?P<on>\d+),(?P<off>\d+),(?P<vel>\d+),\d+,\d+\)$"
)
INS_SPELL_RE = re.compile(
    r"^insertion-note\((?P<nid>[^,]+),\[(?P<step>[A-Ga-g]),(?P<alter>[^\]]+)\],(?P<oct>-?\d+),"
    r"(?P<on>\d+),(?P<off>\d+),\d+,(?P<vel>\d+)\)$"
)
CLOCK_RE = re.compile(r"info\(midiClockUnits,(\d+)\)")
RATE_RE = re.compile(r"info\(midiClockRate,(\d+)\)")
REPEAT_RE = re.compile(r"-(\d+)$")
# The float onset in a match file counts denominator-notes (a quarter in 4/4,
# an eighth in 6/8), and the unit changes when the time signature changes.
# Version 1 files: scoreprop(timeSignature,3/8,0:1,0,-1.0000)
# Version 5 files: meta(timeSignature,6/8,92,1093.0)
TS_SCOREPROP_RE = re.compile(
    r"scoreprop\(timeSignature,(\d+)/(\d+),[^,]+,[^,]+,(-?\d+(?:\.\d+)?)\)"
)
TS_META_RE = re.compile(
    r"meta\(timeSignature,(\d+)/(\d+),[^,]+,(-?\d+(?:\.\d+)?)\)"
)
INFO_TS_RE = re.compile(r"info\(timeSignature,\[(\d+)/(\d+)\]\)")


def midi_pitch(step, alter, octave):
    base = PC.get(step.upper())
    shift = ALTER.get(alter)
    if base is None or shift is None:
        return None
    return (int(octave) + 1) * 12 + base + shift


def _rhs(text):
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    return text


def _repeat(sid):
    found = REPEAT_RE.search(sid)
    return int(found.group(1)) if found else 1


def _ornament(ann):
    text = ann.lower()
    return any(token in text for token in ("trill", "mordent", "turn", "tremolo", "ornament", "grace"))


def _parse_performed(text):
    """Performance note on the right-hand side, numeric or spelled."""
    num = NOTE_NUM_RE.match(text) or INS_NUM_RE.match(text)
    if num:
        return {
            "nid": num.group("nid"),
            "pitch": int(num.group("pitch")),
            "on_tick": int(num.group("on")),
            "off_tick": int(num.group("off")),
        }
    spelled = NOTE_SPELL_RE.match(text) or INS_SPELL_RE.match(text)
    if spelled:
        pitch = midi_pitch(spelled.group("step"), spelled.group("alter"), spelled.group("oct"))
        if pitch is None:
            return None
        return {
            "nid": spelled.group("nid"),
            "pitch": pitch,
            "on_tick": int(spelled.group("on")),
            "off_tick": int(spelled.group("off")),
        }
    return None


def parse_match(path):
    matched = []
    deletions = []
    insertions = []
    unparsed = 0
    seen = 0
    clock = 480
    rate = 500000
    signatures = {}
    info_signature = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        found = CLOCK_RE.search(line)
        if found:
            clock = int(found.group(1))
        found = RATE_RE.search(line)
        if found:
            rate = int(found.group(1))
        for pattern in (TS_SCOREPROP_RE, TS_META_RE):
            found = pattern.search(line)
            if found:
                signatures[float(found.group(3))] = (int(found.group(1)), int(found.group(2)))
        found = INFO_TS_RE.search(line)
        if found:
            info_signature = (int(found.group(1)), int(found.group(2)))
        if line.startswith("insertion-note("):
            seen += 1
            note = _parse_performed(_rhs(line))
            if note is None:
                unparsed += 1
            else:
                insertions.append(note)
            continue
        if not line.startswith("snote("):
            continue
        seen += 1
        header = SNOTE_RE.match(line)
        if header is None:
            unparsed += 1
            continue
        rhs = _rhs(header.group("rhs"))
        score_pitch = midi_pitch(header.group("step"), header.group("alter"), header.group("oct"))
        common = {
            "sid": header.group("sid"),
            "on_q": float(header.group("on")),
            "off_q": float(header.group("off")),
            "repeat": _repeat(header.group("sid")),
            "ornament": _ornament(header.group("ann")),
            "score_pitch": score_pitch,
        }
        if rhs == "deletion":
            deletions.append(common)
            continue
        note = _parse_performed(rhs)
        if note is None:
            unparsed += 1
            continue
        note.update(common)
        matched.append(note)
    if info_signature is not None and 0.0 not in signatures:
        signatures[0.0] = info_signature
    return {
        "clock": clock,
        "rate": rate,
        "time_signatures": sorted((pos, num, den) for pos, (num, den) in signatures.items()),
        "matched": matched,
        "deletions": deletions,
        "insertions": insertions,
        "unparsed": unparsed,
        "seen": seen,
    }


def float_to_file_quarters(values, changes):
    """Match-file onset to quarter notes from the start of the score MIDI.

    changes are (position, numerator, denominator). Position 0 of the file is
    the first time signature, which is before the downbeat when the match
    file uses a negative anacrusis.
    """
    values = np.asarray(values, dtype=np.float64)
    if not changes:
        return values.copy()
    pos = np.array([change[0] for change in changes], dtype=np.float64)
    denom = np.array([change[2] for change in changes], dtype=np.float64)
    quarters_at = np.zeros(len(changes), dtype=np.float64)
    for i in range(len(changes) - 1):
        quarters_at[i + 1] = quarters_at[i] + (pos[i + 1] - pos[i]) * 4.0 / denom[i]
    idx = np.clip(np.searchsorted(pos, values, side="right") - 1, 0, len(pos) - 1)
    return quarters_at[idx] + (values - pos[idx]) * 4.0 / denom[idx]


def quarters_to_seconds(pm, quarters):
    """Score MIDI tick 0 is quarter 0 of the file. Negative quarters sit before it."""
    quarters = np.asarray(quarters, dtype=np.float64)
    out = np.empty_like(quarters)
    positive = quarters >= 0
    if positive.any():
        out[positive] = ticks_to_seconds(pm, quarters[positive] * pm.resolution)
    if (~positive).any():
        spq = float(ticks_to_seconds(pm, np.array([float(pm.resolution)]))[0])
        out[~positive] = quarters[~positive] * spq
    return out


def match_seconds(ticks, clock, rate):
    """Match ticks use one clock for the whole file, not the MIDI tempo map."""
    return np.asarray(ticks, dtype=np.float64) * (rate / 1e6) / clock


def ticks_to_seconds(pm, ticks):
    """pretty_midi maps integer ticks. Fractional ticks are interpolated."""
    ticks = np.maximum(np.asarray(ticks, dtype=np.float64), 0.0)
    if ticks.size == 0:
        return ticks
    last = int(np.ceil(float(ticks.max()))) + 1
    pm._update_tick_to_time(last)
    table = pm._PrettyMIDI__tick_to_time
    lo = np.floor(ticks).astype(np.int64)
    frac = ticks - lo
    limit = len(table) - 1
    lo = np.clip(lo, 0, limit)
    hi = np.clip(lo + 1, 0, limit)
    return (1.0 - frac) * table[lo] + frac * table[hi]


def beat_index(times, beat_times):
    """Piecewise-linear beat number. Outside the annotations, keep the end tempo."""
    beats = np.asarray(beat_times, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    grid = np.arange(len(beats), dtype=np.float64)
    out = np.interp(t, beats, grid)
    left = t < beats[0]
    slope = 1.0 / max(beats[1] - beats[0], 1e-6)
    out[left] = (t[left] - beats[0]) * slope
    right = t > beats[-1]
    slope = 1.0 / max(beats[-1] - beats[-2], 1e-6)
    out[right] = (len(beats) - 1) + (t[right] - beats[-1]) * slope
    return out


def cluster_beats(times, tol=0.04):
    if len(times) == 0:
        return np.asarray(times, dtype=np.float64)
    ordered = np.sort(np.asarray(times, dtype=np.float64))
    keep = [float(ordered[0])]
    for t in ordered[1:]:
        if t - keep[-1] >= tol:
            keep.append(float(t))
    return np.asarray(keep, dtype=np.float64)


def _explain(onsets, q):
    if len(onsets) < 8:
        return 0.0
    ioi = np.diff(onsets)
    ioi = ioi[ioi >= 0.03]
    if len(ioi) < 8:
        return 0.0
    cell = 1.0 / q
    steps = np.round(ioi / cell)
    rel = np.abs(ioi - steps * cell) / cell
    good = (steps >= 1) & (steps <= q * 4) & (rel <= 0.25)
    return float(good.mean())


def choose_q(beat_onsets):
    """Coarsest subdivision that explains the beat-normalized gaps.

    A finer grid can always mimic a coarser one, so the finest candidate would
    win every tie and snap quarters onto sixty-fourth notes.
    """
    onsets = cluster_beats(beat_onsets)
    rates = [(q, _explain(onsets, q)) for q in CANDIDATE_Q]
    best = max(rate for _, rate in rates)
    if best <= 0:
        return 4
    for q, rate in rates:
        if rate >= best - 0.02:
            return q
    return 4


def quantize(beats, q):
    return np.round(np.asarray(beats, dtype=np.float64) * q) / q


def link_notes(match_pitch, match_time, midi_pitch, midi_time, tol=LINK_TOL):
    """Greedy same-pitch link. -1 when no MIDI note lies within tol seconds."""
    by_pitch = defaultdict(list)
    for i, (pitch, start) in enumerate(zip(midi_pitch, midi_time)):
        by_pitch[int(pitch)].append((float(start), i))
    for pitch in by_pitch:
        by_pitch[pitch].sort()
    pointer = {pitch: 0 for pitch in by_pitch}
    used = set()
    out = np.full(len(match_pitch), -1, dtype=np.int64)
    order = np.argsort(match_time, kind="mergesort")
    for j in order:
        pitch = int(match_pitch[j])
        start = float(match_time[j])
        bucket = by_pitch.get(pitch)
        if not bucket:
            continue
        k = pointer[pitch]
        while k < len(bucket) and bucket[k][0] < start - tol:
            k += 1
        pointer[pitch] = k
        best = None
        m = k
        while m < len(bucket) and bucket[m][0] <= start + tol:
            midi_i = bucket[m][1]
            if midi_i not in used:
                dt = abs(bucket[m][0] - start)
                if best is None or dt < best[0]:
                    best = (dt, midi_i)
            m += 1
        if best is not None:
            out[j] = best[1]
            used.add(best[1])
    return out


def repeated_mask(pitch, written):
    """Another note of the same pitch within one beat."""
    mask = np.zeros(len(pitch), dtype=bool)
    groups = defaultdict(list)
    for i, p in enumerate(pitch):
        groups[int(p)].append(i)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda i: written[i])
        for a, b in zip(idxs, idxs[1:]):
            if written[b] - written[a] <= 1.0:
                mask[a] = True
                mask[b] = True
    return mask


def near_insertion(pitch, perf_beats, ins_pitch, ins_beats, window=0.5):
    mask = np.zeros(len(pitch), dtype=bool)
    if len(ins_pitch) == 0:
        return mask
    buckets = defaultdict(list)
    for p, b in zip(ins_pitch, ins_beats):
        buckets[int(p)].append(float(b))
    for p in buckets:
        buckets[p].sort()
    for i, p in enumerate(pitch):
        starts = buckets.get(int(p))
        if not starts:
            continue
        b = float(perf_beats[i])
        k = np.searchsorted(starts, b)
        for j in (k - 1, k):
            if 0 <= j < len(starts) and abs(starts[j] - b) <= window:
                mask[i] = True
                break
    return mask


def _increasing(times):
    times = np.asarray(times, dtype=np.float64)
    return len(times) >= 4 and bool(np.all(np.diff(times) > 1e-4))


def attack_stats(onset_q, written, snapped, q):
    """Chord notes that stay together, and separate attacks that collapse."""
    groups = defaultdict(list)
    for i, key in enumerate(np.round(onset_q, 4)):
        groups[float(key)].append(i)
    attacks = sorted(groups.values(), key=lambda idxs: written[idxs[0]])
    n_chords = 0
    n_recovered = 0
    n_gaps = 0
    n_merged = 0
    cell = 1.0 / q
    prev_b = None
    prev_q = None
    for idxs in attacks:
        bq = snapped[idxs]
        if len(idxs) >= 2:
            n_chords += 1
            if float(bq.max() - bq.min()) <= HIT_BEATS:
                n_recovered += 1
        med = float(np.median(bq))
        b = float(written[idxs[0]])
        if prev_b is not None and (b - prev_b) >= 0.9 * cell:
            n_gaps += 1
            if abs(med - prev_q) <= HIT_BEATS:
                n_merged += 1
        prev_b = b
        prev_q = med
    return n_chords, n_recovered, n_gaps, n_merged


def load_score(path, cache):
    if path in cache:
        return cache[path]
    pm = pmd.PrettyMIDI(str(path))
    notes = flatten_notes(pm)
    bag = Counter(note.pitch for note in notes)
    velocities = {note.velocity for note in notes}
    cache[path] = {
        "pm": pm,
        "bag": bag,
        "n": len(notes),
        "unique": len(bag),
        "all_80": velocities == {80},
    }
    return cache[path]


def evaluate_piece(row, ann, parsed, score_cache):
    perf_path = BASE / row.midi_performance
    score_path = BASE / row.midi_score
    if not perf_path.exists() or not score_path.exists():
        return None
    score = load_score(score_path, score_cache)
    perf_pm = pmd.PrettyMIDI(str(perf_path))
    perf_notes = flatten_notes(perf_pm)
    perf_bag = Counter(note.pitch for note in perf_notes)

    matched = parsed["matched"]
    deletions = parsed["deletions"]
    insertions = parsed["insertions"]
    n_score_match = len(matched) + len(deletions)
    n_perf_match = len(matched) + len(insertions)
    indel = len(insertions) + len(deletions)
    bags_equal = score["bag"] == perf_bag
    bag_keys = set(score["bag"]) | set(perf_bag)
    bag_moved = sum(abs(score["bag"][key] - perf_bag[key]) for key in bag_keys) / 2
    bag_l1 = bag_moved / max(score["n"], len(perf_notes), 1)
    info = {
        "path": row.midi_performance,
        "n_score_midi": score["n"],
        "n_perf_midi": len(perf_notes),
        "n_score_unique": score["unique"],
        "n_perf_unique": len(perf_bag),
        "bags_equal": bags_equal,
        "bag_l1_frac": bag_l1,
        "score_velocity_80": score["all_80"],
        "n_matched": len(matched),
        "n_deletions": len(deletions),
        "n_insertions": len(insertions),
        "unparsed": parsed["unparsed"],
        "seen": parsed["seen"],
        "indel_rate": indel / n_score_match if n_score_match else 0.0,
        "equal_bag_with_indel": bool(bags_equal and indel > 0),
    }

    sb = np.asarray((ann or {}).get("midi_score_beats") or [], dtype=np.float64)
    pb = np.asarray((ann or {}).get("performance_beats") or [], dtype=np.float64)
    if not matched or not _increasing(sb) or not _increasing(pb) or len(sb) != len(pb):
        info["onset"] = False
        return info

    on_tick = np.array([note["on_tick"] for note in matched], dtype=np.float64)
    off_tick = np.array([note["off_tick"] for note in matched], dtype=np.float64)
    pitch = np.array([note["pitch"] for note in matched], dtype=np.int64)
    onset_q = np.array([note["on_q"] for note in matched], dtype=np.float64)
    offset_q = np.array([note["off_q"] for note in matched], dtype=np.float64)
    perf_on = match_seconds(on_tick, parsed["clock"], parsed["rate"])
    perf_off = match_seconds(off_tick, parsed["clock"], parsed["rate"])
    changes = parsed["time_signatures"]
    onset_quarters = float_to_file_quarters(onset_q, changes)
    offset_quarters = float_to_file_quarters(offset_q, changes)
    written = beat_index(quarters_to_seconds(score["pm"], onset_quarters), sb)
    written_off = beat_index(quarters_to_seconds(score["pm"], offset_quarters), sb)
    played = beat_index(perf_on, pb)
    played_off = beat_index(perf_off, pb)

    ins_on = np.zeros(0, dtype=np.float64)
    ins_pitch = np.zeros(0, dtype=np.int64)
    if insertions:
        ins_tick = np.array([note["on_tick"] for note in insertions], dtype=np.float64)
        ins_on = match_seconds(ins_tick, parsed["clock"], parsed["rate"])
        ins_pitch = np.array([note["pitch"] for note in insertions], dtype=np.int64)
    grid_onsets = played if len(ins_on) == 0 else np.concatenate([played, beat_index(ins_on, pb)])
    q = choose_q(grid_onsets)
    snapped_a = quantize(played, q)
    snapped_oracle_q = max(CANDIDATE_Q, key=lambda candidate: np.mean(np.abs(quantize(played, candidate) - written) <= HIT_BEATS))
    snapped_oracle = quantize(played, snapped_oracle_q)
    snapped_q4 = quantize(played, 4)

    err_pre = np.abs(played - written)
    err_a = np.abs(snapped_a - written)
    err_oracle = np.abs(snapped_oracle - written)
    err_q4 = np.abs(snapped_q4 - written)

    midi_pitch = np.array([note.pitch for note in perf_notes], dtype=np.int64)
    midi_start = np.array([note.start for note in perf_notes], dtype=np.float64)
    midi_dur = np.array([note.duration for note in perf_notes], dtype=np.float64)
    if len(perf_notes):
        _beat, t0, quantum = pulse(midi_start)
        quantum = max(float(quantum), 1e-3)
        grid_start, grid_dur = snap_notes(midi_start, midi_dur, t0, quantum)
        formula = t0 + np.round((midi_start - t0) / quantum) * quantum
        formula_diff = float(np.max(np.abs(grid_start - formula))) if len(grid_start) else 0.0
        end = midi_start + midi_dur
        partitura_end = t0 + np.round((end - t0) / quantum) * quantum
        partitura_start = formula
        zero_dur = int(np.sum(partitura_end <= partitura_start + 1e-9))
    else:
        beat, t0, quantum = 0.5, 0.0, 0.5
        grid_start = np.zeros(0)
        grid_dur = np.zeros(0)
        formula_diff = 0.0
        zero_dur = 0
    links = link_notes(pitch, perf_on, midi_pitch, midi_start) if len(perf_notes) else np.full(len(pitch), -1)
    linked = links >= 0
    err_b = np.full(len(pitch), np.nan, dtype=np.float64)
    dur_b = np.full(len(pitch), np.nan, dtype=np.float64)
    if linked.any():
        snapped_sec = grid_start[links[linked]]
        snapped_end = snapped_sec + grid_dur[links[linked]]
        err_b[linked] = np.abs(beat_index(snapped_sec, pb) - written[linked])
        dur_b[linked] = beat_index(snapped_end, pb) - beat_index(snapped_sec, pb)

    dur_written = written_off - written
    dur_played = played_off - played
    dur_ok = dur_written > 0.01
    dur_q = np.maximum(quantize(dur_played, q), 1.0 / q)
    err_dur = np.full(len(pitch), np.nan)
    err_dur_raw = np.full(len(pitch), np.nan)
    err_dur[dur_ok] = np.abs(dur_q[dur_ok] - dur_written[dur_ok])
    err_dur_raw[dur_ok] = np.abs(dur_played[dur_ok] - dur_written[dur_ok])
    err_dur_b = np.full(len(pitch), np.nan)
    both = dur_ok & linked & np.isfinite(dur_b)
    err_dur_b[both] = np.abs(dur_b[both] - dur_written[both])

    beat_lengths = np.diff(pb)
    beat_lengths = beat_lengths[beat_lengths > 1e-3]
    if len(beat_lengths):
        tempo_maxmin = float(beat_lengths.max() / beat_lengths.min())
        lo, hi = np.percentile(beat_lengths, [10, 90])
        tempo_ratio = float(hi / max(lo, 1e-6))
    else:
        tempo_maxmin = 1.0
        tempo_ratio = 1.0
    version_mismatch = info["indel_rate"] >= VERSION_INDEL
    # Shorter than a triplet eighth. The raw float is in denominator-notes,
    # so convert before comparing.
    grace = (offset_quarters - onset_quarters) <= (1.0 / 12.0)
    ornament = np.array([note["ornament"] for note in matched], dtype=bool) | near_insertion(
        pitch, played, ins_pitch, beat_index(ins_on, pb) if len(ins_on) else np.zeros(0)
    )
    flags = {
        "repeated": repeated_mask(pitch, written),
        "ornament": ornament,
        # Compound-meter eighths land on thirds of the beat. A tuplet here is
        # a 5 or 7 that sits on neither the binary grid nor that ternary grid.
        "tuplet": (np.abs(written - quantize(written, 4)) > HIT_BEATS)
        & (np.abs(written - quantize(written, 3)) > HIT_BEATS)
        & (np.abs(written - quantize(written, 6)) > HIT_BEATS)
        & (
            (np.abs(written - quantize(written, 5)) <= HIT_BEATS)
            | (np.abs(written - quantize(written, 7)) <= HIT_BEATS)
        ),
        "pickup": written < -HIT_BEATS,
        "tempo_change": np.full(len(pitch), tempo_ratio > TEMPO_RATIO),
        "repeat_diff": np.array([note["repeat"] >= 2 for note in matched], dtype=bool),
        "score_version": np.full(len(pitch), version_mismatch),
        "bag_diff": np.full(len(pitch), not bags_equal),
        "grace": grace,
        "off_grid": np.abs(written - quantize(written, q)) > HIT_BEATS,
    }
    miss = err_a > HIT_BEATS
    chords_a = attack_stats(onset_q, written, snapped_a, q)
    # Unlinked notes stay out of the pipeline chord count. Treating them as
    # already correct would hide a missed MIDI link.
    if linked.any():
        chords_b = attack_stats(
            onset_q[linked],
            written[linked],
            beat_index(grid_start[links[linked]], pb),
            q,
        )
    else:
        chords_b = (0, 0, 0, 0)

    info.update(
        {
            "onset": True,
            "q": int(q),
            "q_oracle": int(snapped_oracle_q),
            "tempo_ratio": tempo_ratio,
            "tempo_maxmin": tempo_maxmin,
            "version_mismatch": version_mismatch,
            "n_linked": int(linked.sum()),
            "formula_diff": formula_diff,
            "partitura_zero": zero_dur,
            "n_snapped": int(len(midi_start)),
            "ids_preserved": int(len(grid_start) == len(midi_start)),
            "median_pre": float(np.median(err_pre)),
            "hit_a": float(np.mean(err_a <= HIT_BEATS)),
            "hit_b": float(np.mean(err_b[linked] <= HIT_BEATS)) if linked.any() else None,
            "hit_oracle": float(np.mean(err_oracle <= HIT_BEATS)),
            "hit_q4": float(np.mean(err_q4 <= HIT_BEATS)),
            "chords_a": chords_a,
            "chords_b": chords_b,
            "err_pre": err_pre.astype(np.float32),
            "err_a": err_a.astype(np.float32),
            "err_b": err_b.astype(np.float32),
            "err_oracle": err_oracle.astype(np.float32),
            "err_q4": err_q4.astype(np.float32),
            "err_dur": err_dur.astype(np.float32),
            "err_dur_raw": err_dur_raw.astype(np.float32),
            "err_dur_b": err_dur_b.astype(np.float32),
            "flags": {name: mask.astype(bool) for name, mask in flags.items()},
            "miss_a": miss,
            "timeline_ok": bool(np.median(err_pre) <= 0.25),
        }
    )
    return info


def _cat():
    return {"n": 0, "miss": 0}


def _add_abs(chunks, values):
    keep = values[np.isfinite(values)]
    if len(keep):
        chunks.append(keep.astype(np.float32))


def summarize_abs(chunks):
    if not chunks:
        return {"n": 0}
    err = np.concatenate(chunks)
    return {
        "n": int(err.size),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "within_0.001": float(np.mean(err <= 0.001)),
        "within_0.02": float(np.mean(err <= HIT_BEATS)),
        "within_0.05": float(np.mean(err <= 0.05)),
        "within_0.125": float(np.mean(err <= 0.125)),
        "within_0.25": float(np.mean(err <= 0.25)),
        "within_0.5": float(np.mean(err <= 0.5)),
        "within_1": float(np.mean(err <= 1.0)),
    }


def sanity_check(annotations):
    """Bach prelude: tick 480 is 0.5s, and the written eighth sits half a beat early."""
    path = NASAP / "Bach/Fugue/bwv_846/Shi05M.match"
    parsed = parse_match(path)
    first = parsed["matched"][0]
    if first["pitch"] != 60 or first["on_tick"] != 480 or abs(first["on_q"] - 0.5) > 1e-6:
        raise SystemExit(f"Shi05M parse mismatch: {first}")
    perf = pmd.PrettyMIDI(str(BASE / "Bach/Fugue/bwv_846/Shi05M.mid"))
    score = pmd.PrettyMIDI(str(BASE / "Bach/Fugue/bwv_846/midi_score.mid"))
    when = float(match_seconds(np.array([first["on_tick"]]), parsed["clock"], parsed["rate"])[0])
    if abs(when - 0.5) > 0.005:
        raise SystemExit(f"Shi05M tick conversion gave {when}, expected 0.5")
    ann = annotations["Bach/Fugue/bwv_846/Shi05M.mid"]
    sb = np.asarray(ann["midi_score_beats"], dtype=np.float64)
    written = beat_index(
        quarters_to_seconds(score, float_to_file_quarters(np.array([0.5, 1.0]), parsed["time_signatures"])),
        sb,
    )
    if abs(written[0] + 0.5) > 0.02 or abs(written[1]) > 0.02:
        raise SystemExit(f"Shi05M written beats {written}, expected [-0.5, 0]")
    spelled = parse_match(NASAP / "Balakirev/Islamey/Shi11.match")
    if spelled["matched"][0]["pitch"] != midi_pitch("B", "b", 3):
        raise SystemExit(f"spelled pitch mismatch: {spelled['matched'][0]}")
    if spelled["time_signatures"][:2] != [(0.0, 12, 16), (1093.0, 6, 8)]:
        raise SystemExit(f"Islamey signatures {spelled['time_signatures'][:4]}")
    for rel, limit in (
        ("Bach/Fugue/bwv_856/LuoJ01M.mid", 0.05),
        ("Bach/Fugue/bwv_860/Ko04M.mid", 0.08),
    ):
        med = _median_onset_error(rel, annotations[rel])
        if med > limit:
            raise SystemExit(f"{rel} median onset error {med:.3f} beats")
    print(
        f"sanity Shi05M pitch {first['pitch']} at {when:.4f}s, written beats {written[0]:.3f} {written[1]:.3f}",
        flush=True,
    )


def _median_onset_error(rel, ann):
    """Median |played beat − written beat| for one performance. Used as a gate."""
    parsed = parse_match(NASAP / Path(rel).with_suffix(".match"))
    perf = pmd.PrettyMIDI(str(BASE / rel))
    score_rel = str(Path(rel).parent / "midi_score.mid")
    score = pmd.PrettyMIDI(str(BASE / score_rel))
    sb = np.asarray(ann["midi_score_beats"], dtype=np.float64)
    pb = np.asarray(ann["performance_beats"], dtype=np.float64)
    notes = parsed["matched"]
    onset = np.array([note["on_q"] for note in notes], dtype=np.float64)
    ticks = np.array([note["on_tick"] for note in notes], dtype=np.float64)
    written = beat_index(quarters_to_seconds(score, float_to_file_quarters(onset, parsed["time_signatures"])), sb)
    played = beat_index(match_seconds(ticks, parsed["clock"], parsed["rate"]), pb)
    return float(np.median(np.abs(played - written)))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    metadata = pd.read_csv(BASE / "metadata.csv")
    with open(BASE / "asap_annotations.json") as handle:
        annotations = json.load(handle)
    sanity_check(annotations)

    score_cache = {}
    pieces = []
    chunks = {name: [] for name in ("pre", "a", "b", "oracle", "q4", "dur", "dur_raw", "dur_b")}
    ok_chunks = {name: [] for name in ("pre", "a", "b", "dur", "dur_raw", "dur_b")}
    cats = defaultdict(_cat)
    ok_cats = defaultdict(_cat)
    tempo_slice = {"fast": _cat(), "stable": _cat(), "fast_b": _cat(), "stable_b": _cat()}
    chords = {"a_n": 0, "a_ok": 0, "a_gaps": 0, "a_merge": 0, "b_n": 0, "b_ok": 0, "b_gaps": 0, "b_merge": 0}
    q_hist = Counter()
    totals = Counter()
    failures = []
    bag_equal = bag_equal_indel = bag_diff = 0
    bag_l1 = []
    tempo_maxmin_over = 0
    score_80 = 0
    n_scores_seen = set()

    rows = list(metadata.itertuples(index=False))
    if limit is not None:
        rows = rows[:limit]
    for n_done, row in enumerate(rows, start=1):
        match_path = NASAP / Path(row.midi_performance).with_suffix(".match")
        totals["metadata"] += 1
        if not match_path.exists():
            totals["no_match"] += 1
            continue
        totals["with_match"] += 1
        try:
            parsed = parse_match(match_path)
            result = evaluate_piece(row, annotations.get(row.midi_performance), parsed, score_cache)
        except Exception as exc:
            totals["errors"] += 1
            if len(failures) < 8:
                failures.append(f"{row.midi_performance}: {exc}")
            continue
        if result is None:
            totals["missing_midi"] += 1
            continue
        pieces.append(result)
        totals["unparsed"] += result["unparsed"]
        totals["seen"] += result["seen"]
        totals["matched"] += result["n_matched"]
        totals["deletions"] += result["n_deletions"]
        totals["insertions"] += result["n_insertions"]
        totals["score_midi"] += result["n_score_midi"]
        totals["perf_midi"] += result["n_perf_midi"]
        if result["bags_equal"]:
            bag_equal += 1
            if result["equal_bag_with_indel"]:
                bag_equal_indel += 1
        else:
            bag_diff += 1
        bag_l1.append(result["bag_l1_frac"])
        if result["score_velocity_80"]:
            score_80 += 1
        n_scores_seen.add(row.midi_score)
        if not result["onset"]:
            totals["no_beats"] += 1
            continue
        totals["onset_pieces"] += 1
        totals["onset_notes"] += result["n_matched"]
        totals["linked"] += result["n_linked"]
        totals["snapped"] += result["n_snapped"]
        totals["ids_preserved"] += result["ids_preserved"] * result["n_snapped"]
        totals["formula_bad"] += result["formula_diff"] > 1e-6
        totals["partitura_zero"] += result["partitura_zero"]
        q_hist[result["q"]] += 1
        if result["tempo_maxmin"] > TEMPO_RATIO:
            tempo_maxmin_over += 1
        _add_abs(chunks["pre"], result["err_pre"])
        _add_abs(chunks["a"], result["err_a"])
        _add_abs(chunks["b"], result["err_b"])
        _add_abs(chunks["oracle"], result["err_oracle"])
        _add_abs(chunks["q4"], result["err_q4"])
        _add_abs(chunks["dur"], result["err_dur"])
        _add_abs(chunks["dur_raw"], result["err_dur_raw"])
        _add_abs(chunks["dur_b"], result["err_dur_b"])
        if result["timeline_ok"]:
            totals["timeline_ok"] += 1
            _add_abs(ok_chunks["pre"], result["err_pre"])
            _add_abs(ok_chunks["a"], result["err_a"])
            _add_abs(ok_chunks["b"], result["err_b"])
            _add_abs(ok_chunks["dur"], result["err_dur"])
            _add_abs(ok_chunks["dur_raw"], result["err_dur_raw"])
            _add_abs(ok_chunks["dur_b"], result["err_dur_b"])
        else:
            totals["timeline_bad"] += 1
        miss = result["miss_a"]
        for name, mask in result["flags"].items():
            cats[name]["n"] += int(mask.sum())
            cats[name]["miss"] += int(np.sum(mask & miss))
            if result["timeline_ok"]:
                ok_cats[name]["n"] += int(mask.sum())
                ok_cats[name]["miss"] += int(np.sum(mask & miss))
        totals["miss_a"] += int(miss.sum())
        none = ~np.zeros(len(miss), dtype=bool)
        for name in (
            "repeated",
            "ornament",
            "tuplet",
            "pickup",
            "tempo_change",
            "repeat_diff",
            "score_version",
            "grace",
            "off_grid",
        ):
            none &= ~result["flags"][name]
        cats["none_of_those"]["n"] += int(none.sum())
        cats["none_of_those"]["miss"] += int(np.sum(none & miss))
        if result["timeline_ok"]:
            ok_cats["none_of_those"]["n"] += int(none.sum())
            ok_cats["none_of_those"]["miss"] += int(np.sum(none & miss))
            totals["miss_a_ok"] += int(miss.sum())
            totals["onset_notes_ok"] += result["n_matched"]
        tempo_name = "fast" if result["tempo_ratio"] > TEMPO_RATIO else "stable"
        tempo_slice[tempo_name]["n"] += result["n_matched"]
        tempo_slice[tempo_name]["miss"] += int(miss.sum())
        if result["hit_b"] is not None:
            # Reconstruct pipeline misses from the stored error array.
            err_b = result["err_b"]
            linked = np.isfinite(err_b)
            tempo_slice[tempo_name + "_b"]["n"] += int(linked.sum())
            tempo_slice[tempo_name + "_b"]["miss"] += int(np.sum(linked & (err_b > HIT_BEATS)))
        ca, ra, ga, ma = result["chords_a"]
        cb, rb, gb, mb = result["chords_b"]
        chords["a_n"] += ca
        chords["a_ok"] += ra
        chords["a_gaps"] += ga
        chords["a_merge"] += ma
        chords["b_n"] += cb
        chords["b_ok"] += rb
        chords["b_gaps"] += gb
        chords["b_merge"] += mb
        # Drop bulky arrays before keeping the piece row.
        for key in ("err_pre", "err_a", "err_b", "err_oracle", "err_q4", "err_dur", "err_dur_raw", "err_dur_b", "flags", "miss_a"):
            del result[key]
        if n_done % 50 == 0:
            hit = 1.0 - totals["miss_a"] / max(totals["onset_notes"], 1)
            print(
                f"{n_done} files  hit {hit:.3f}  linked {totals['linked'] / max(totals['onset_notes'], 1):.3f}",
                flush=True,
            )

    def rates(block):
        out = {}
        for name, rec in block.items():
            out[name] = {
                "n": rec["n"],
                "miss": rec["miss"],
                "miss_rate": rec["miss"] / rec["n"] if rec["n"] else None,
            }
        return out

    onset_pieces = [piece for piece in pieces if piece.get("onset")]
    def macro(key):
        vals = [piece[key] for piece in onset_pieces if piece.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "hit_beats": HIT_BEATS,
        "metadata_rows": totals["metadata"],
        "with_match": totals["with_match"],
        "no_match": totals["no_match"],
        "errors": totals["errors"],
        "error_samples": failures,
        "unparsed": totals["unparsed"],
        "alignment_lines": totals["seen"],
        "pitch_bags": {
            "performances": len(pieces),
            "unique_scores": len(n_scores_seen),
            "identical": bag_equal,
            "identical_fraction": bag_equal / len(pieces) if pieces else None,
            "identical_but_indel": bag_equal_indel,
            "different": bag_diff,
            "median_bag_l1_frac": float(np.median(bag_l1)) if bag_l1 else None,
            "bag_l1_under_1pct": int(sum(value <= 0.01 for value in bag_l1)),
            "bag_l1_under_5pct": int(sum(value <= 0.05 for value in bag_l1)),
            "score_notes": totals["score_midi"],
            "performance_notes": totals["perf_midi"],
            "scores_velocity_all_80": score_80,
        },
        "discrepancies": {
            "matched": totals["matched"],
            "deletions": totals["deletions"],
            "insertions": totals["insertions"],
            "deletion_fraction_of_score": totals["deletions"] / max(totals["matched"] + totals["deletions"], 1),
            "insertion_fraction_of_performance": totals["insertions"] / max(totals["matched"] + totals["insertions"], 1),
        },
        "compatibility": {
            "onset_pieces": totals["onset_pieces"],
            "onset_notes": totals["onset_notes"],
            "linked_to_local_midi": totals["linked"],
            "link_fraction": totals["linked"] / max(totals["onset_notes"], 1),
            "no_beat_map": totals["no_beats"],
            "timeline_ok_pieces": totals["timeline_ok"],
            "timeline_bad_pieces": totals["timeline_bad"],
        },
        "id_preservation": {
            "snapped_notes": totals["snapped"],
            "preserved": totals["ids_preserved"],
            "fraction": totals["ids_preserved"] / max(totals["snapped"], 1),
            "formula_mismatches": totals["formula_bad"],
            "partitura_zero_duration": totals["partitura_zero"],
            "partitura_zero_fraction": totals["partitura_zero"] / max(totals["snapped"], 1),
        },
        "grid_choices": dict(sorted(q_hist.items())),
        "tempo_maxmin_over_1_12": tempo_maxmin_over,
        "macro_hit_beat_map": macro("hit_a"),
        "macro_hit_pipeline": macro("hit_b"),
        "macro_hit_oracle": macro("hit_oracle"),
        "before_snap": summarize_abs(chunks["pre"]),
        "onset_beat_map": summarize_abs(chunks["a"]),
        "onset_oracle_grid": summarize_abs(chunks["oracle"]),
        "onset_fixed_sixteenths": summarize_abs(chunks["q4"]),
        "onset_pipeline": summarize_abs(chunks["b"]),
        "duration_raw": summarize_abs(chunks["dur_raw"]),
        "duration_beat_map": summarize_abs(chunks["dur"]),
        "duration_pipeline": summarize_abs(chunks["dur_b"]),
        "timeline_ok": {
            "before_snap": summarize_abs(ok_chunks["pre"]),
            "onset_beat_map": summarize_abs(ok_chunks["a"]),
            "onset_pipeline": summarize_abs(ok_chunks["b"]),
            "duration_raw": summarize_abs(ok_chunks["dur_raw"]),
            "duration_beat_map": summarize_abs(ok_chunks["dur"]),
            "duration_pipeline": summarize_abs(ok_chunks["dur_b"]),
        },
        "chords": {
            "beat_map_chords": chords["a_n"],
            "beat_map_recovered": chords["a_ok"],
            "beat_map_recovery": chords["a_ok"] / chords["a_n"] if chords["a_n"] else None,
            "beat_map_separate_attacks": chords["a_gaps"],
            "beat_map_merged": chords["a_merge"],
            "beat_map_merge_rate": chords["a_merge"] / chords["a_gaps"] if chords["a_gaps"] else None,
            "pipeline_chords": chords["b_n"],
            "pipeline_recovered": chords["b_ok"],
            "pipeline_recovery": chords["b_ok"] / chords["b_n"] if chords["b_n"] else None,
            "pipeline_separate_attacks": chords["b_gaps"],
            "pipeline_merged": chords["b_merge"],
            "pipeline_merge_rate": chords["b_merge"] / chords["b_gaps"] if chords["b_gaps"] else None,
        },
        "failure_categories": rates(cats),
        "failure_categories_timeline_ok": rates(ok_cats),
        "tempo_slices": {
            name: {
                "n": rec["n"],
                "miss_rate": rec["miss"] / rec["n"] if rec["n"] else None,
            }
            for name, rec in tempo_slice.items()
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "deexpress_eval.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(OUT_DIR / "deexpress_pieces.csv", "w", newline="") as handle:
        fieldnames = [
            "path",
            "n_score_midi",
            "n_perf_midi",
            "bags_equal",
            "n_matched",
            "n_deletions",
            "n_insertions",
            "indel_rate",
            "onset",
            "q",
            "tempo_ratio",
            "median_pre",
            "hit_a",
            "hit_b",
            "hit_oracle",
            "hit_q4",
            "timeline_ok",
            "n_linked",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for piece in pieces:
            writer.writerow(piece)
    print(json.dumps({k: summary[k] for k in (
        "pitch_bags", "discrepancies", "compatibility", "id_preservation",
        "before_snap", "onset_beat_map", "onset_pipeline", "chords",
        "duration_beat_map", "duration_pipeline", "tempo_slices",
    )}, indent=2))


if __name__ == "__main__":
    main()
