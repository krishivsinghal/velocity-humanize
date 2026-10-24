"""
Score clues that the note list alone does not have.

Measure position and time signature come from the ASAP score annotations.
Voice and staff come from the MusicXML, and only where the XML notes line up
with the MIDI score. A miss is voice 0, not a guessed part.

Chord size, octave doubling, and register or density changes are computed
later from the notes themselves, in features.extra_timing.

  PYTHONPATH=src python src/score_structure.py
  PYTHONPATH=src python src/score_structure.py 8
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pretty_midi as pmd

from features import pack_voice
from preprocess import (
    align_by_pitch,
    flatten_notes,
    piece_length,
    round_note,
    shift_to_zero,
    stretch_notes,
)

BASE = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/asap")
PROCESSED = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/processed")
STRUCT_DIR = PROCESSED / "structure"

# A 64th note is 0.0625 quarters. Stay inside half of that so neighbors cannot swap.
QUARTER_TOL = 0.03
# Shorter than this, an unmatched MIDI note is treated as a grace and ignored.
GRACE_QUARTERS = 0.12
# Below this, onsets are not really the same score, so every voice is dropped.
# Above it, unmatched notes stay unknown and the unique matches are kept.
VOICE_MATCH_MIN = 0.50

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def local(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def child_text(elem, name):
    for child in elem:
        if local(child.tag) == name and child.text:
            return child.text.strip()
    return None


def xml_pitched_notes(path: Path):
    """(onset in quarters, pitch, staff or None, voice or None) for every pitched note."""
    root = ET.parse(path).getroot()
    notes = []
    for part in root:
        if local(part.tag) != "part":
            continue
        divisions = 1.0
        staves = 1
        cursor = 0.0
        last_onset = 0.0
        # (staff, voice, pitch) of a tie that has started and not yet stopped.
        open_ties = set()
        for measure in part:
            if local(measure.tag) != "measure":
                continue
            for elem in measure:
                name = local(elem.tag)
                if name == "attributes":
                    div = child_text(elem, "divisions")
                    if div:
                        divisions = max(float(div), 1.0)
                    st = child_text(elem, "staves")
                    if st:
                        staves = int(st)
                    continue
                if name == "backup":
                    dur = child_text(elem, "duration")
                    cursor -= float(dur or 0.0) / divisions
                    continue
                if name == "forward":
                    dur = child_text(elem, "duration")
                    cursor += float(dur or 0.0) / divisions
                    continue
                if name != "note":
                    continue
                tags = {local(child.tag) for child in elem}
                # Grace notes sit on the following attack. Cue notes in this
                # corpus are real notes (MuseScore wrote inner voices that way).
                if "grace" in tags:
                    continue
                dur = child_text(elem, "duration")
                dur_q = float(dur or 0.0) / divisions
                if "chord" in tags:
                    onset = last_onset
                else:
                    onset = cursor
                    last_onset = onset
                    cursor += dur_q
                if "rest" in tags:
                    continue
                tie_types = {
                    child.get("type")
                    for child in elem
                    if local(child.tag) == "tie"
                }
                pitch_el = next((child for child in elem if local(child.tag) == "pitch"), None)
                if pitch_el is None:
                    continue
                step = child_text(pitch_el, "step")
                octave = child_text(pitch_el, "octave")
                if step not in STEP or octave is None:
                    continue
                alter = child_text(pitch_el, "alter")
                alt = int(round(float(alter))) if alter is not None else 0
                pitch = (int(octave) + 1) * 12 + STEP[step] + alt
                staff_txt = child_text(elem, "staff")
                voice_txt = child_text(elem, "voice")
                if staff_txt is None and staves == 1:
                    staff = 1
                elif staff_txt is None:
                    staff = None
                else:
                    staff = int(staff_txt)
                voice = int(voice_txt) if voice_txt is not None else None
                key = (staff, voice, pitch)
                # Merge a tie into the note that started it. A stop with no
                # start (the start was a grace we skipped) is a real attack.
                if "stop" in tie_types and key in open_ties:
                    open_ties.discard(key)
                    if "start" in tie_types:
                        open_ties.add(key)
                    continue
                if "start" in tie_types:
                    open_ties.add(key)
                notes.append((onset, pitch, staff, voice))
    return notes


def match_voices(score_pm: pmd.PrettyMIDI, score_notes, xml_path: Path):
    """Voice id per raw score note. All zeros when the XML does not line up."""
    n = len(score_notes)
    voice = np.zeros(n, dtype=np.int16)
    info = {"xml_notes": 0, "match_rate": 0.0, "labeled": 0.0, "kept": False}
    if n == 0 or not xml_path.exists():
        return voice, info
    try:
        xml_notes = xml_pitched_notes(xml_path)
    except (ET.ParseError, OSError, ValueError):
        return voice, info
    info["xml_notes"] = len(xml_notes)
    if not xml_notes:
        return voice, info

    res = float(score_pm.resolution or 480)
    by_pitch = defaultdict(list)
    for onset, pitch, staff, voice_n in xml_notes:
        by_pitch[int(pitch)].append([float(onset), staff, voice_n, False])

    matched = 0
    counted = 0
    for i, note in enumerate(score_notes):
        q = score_pm.time_to_tick(float(note.start)) / res
        dur_q = (score_pm.time_to_tick(float(note.end)) - score_pm.time_to_tick(float(note.start))) / res
        cands = by_pitch.get(int(note.pitch), [])
        best = None
        best_dt = QUARTER_TOL
        tie = False
        for cand in cands:
            if cand[3]:
                continue
            dt = abs(cand[0] - q)
            if dt > QUARTER_TOL:
                continue
            if best is None or dt < best_dt - 1e-6:
                best = cand
                best_dt = dt
                tie = False
            elif abs(dt - best_dt) <= 1e-6:
                tie = True
        grace = dur_q < GRACE_QUARTERS
        if best is None or tie:
            if not grace:
                counted += 1
            continue
        best[3] = True
        if not grace:
            counted += 1
            matched += 1
        else:
            matched += 1
            counted += 1
        voice[i] = pack_voice(best[1], best[2])

    rate = matched / counted if counted else 0.0
    info["match_rate"] = float(rate)
    if rate < VOICE_MATCH_MIN:
        voice[:] = 0
        info["kept"] = False
        info["labeled"] = 0.0
    else:
        info["kept"] = True
        info["labeled"] = float((voice > 0).mean()) if n else 0.0
    return voice, info


def _time_signatures(ann):
    sigs = []
    raw = ann.get("midi_score_time_signatures") or {}
    for key, val in raw.items():
        try:
            when = float(key)
            text = str(val[0])
            beats = int(val[1])
        except (TypeError, ValueError, IndexError):
            continue
        denom = 4
        if "/" in text:
            try:
                denom = int(text.split("/")[1])
            except ValueError:
                denom = 4
        if beats <= 0 or denom <= 0:
            continue
        sigs.append((when, beats, denom))
    sigs.sort()
    return sigs


def _signature_at(sigs, t):
    chosen = None
    for when, beats, denom in sigs:
        if when <= t + 1e-4:
            chosen = (beats, denom)
        else:
            break
    return chosen


def _bar_phase(t, downbeats):
    """Fraction through the annotated measure. None when the bar is not reliable."""
    if len(downbeats) < 2:
        return None
    i = int(np.searchsorted(downbeats, t, side="right") - 1)
    if i < 0:
        bar_len = float(downbeats[1] - downbeats[0])
        if bar_len <= 1e-3:
            return None
        phase = 1.0 + (float(t) - float(downbeats[0])) / bar_len
        if 0.0 <= phase < 1.0:
            return phase
        return None
    if i >= len(downbeats) - 1:
        bar_len = float(downbeats[-1] - downbeats[-2])
        if bar_len <= 1e-3:
            return None
        phase = (float(t) - float(downbeats[-1])) / bar_len
        if phase < 0.0 or phase > 1.25:
            return None
        return min(phase, 0.999)
    bar_len = float(downbeats[i + 1] - downbeats[i])
    if bar_len <= 1e-3:
        return None
    phase = (float(t) - float(downbeats[i])) / bar_len
    if phase < 0.0 or phase >= 1.0:
        return None
    return phase


def meter_arrays(score_notes, ann):
    n = len(score_notes)
    bar = np.zeros(n, dtype=np.float32)
    known = np.zeros(n, dtype=np.float32)
    beats = np.zeros(n, dtype=np.float32)
    denom = np.zeros(n, dtype=np.float32)
    if not ann:
        return bar, known, beats, denom
    downbeats = np.array(sorted(float(t) for t in (ann.get("midi_score_downbeats") or [])), dtype=np.float64)
    sigs = _time_signatures(ann)
    for i, note in enumerate(score_notes):
        phase = _bar_phase(float(note.start), downbeats)
        sig = _signature_at(sigs, float(note.start))
        if phase is None or sig is None:
            continue
        bar[i] = np.float32(phase)
        known[i] = 1.0
        beats[i] = np.float32(sig[0])
        denom[i] = np.float32(sig[1])
    return bar, known, beats, denom


def align_annotated(score_pm, perf_pm, ann, xml_path):
    """Same pairing as prepare_pair, with score clues attached before the stretch."""
    raw = flatten_notes(score_pm)
    voice, voice_info = match_voices(score_pm, raw, xml_path)
    bar, known, beats, denom = meter_arrays(raw, ann)
    for i, note in enumerate(raw):
        note.voice_id = int(voice[i])
        note.bar_phase = float(bar[i])
        note.meter_known = float(known[i])
        note.beats_per_bar = float(beats[i])
        note.beat_denom = float(denom[i])

    score_notes = shift_to_zero(raw)
    perf_notes = shift_to_zero(flatten_notes(perf_pm))
    if not score_notes or not perf_notes:
        raise ValueError("Score or performance has no notes")
    score_len = piece_length(score_notes)
    perf_len = piece_length(perf_notes)
    if score_len <= 0:
        raise ValueError("Score length is zero")
    score_notes = stretch_notes(score_notes, perf_len / score_len)
    score_notes, perf_notes = align_by_pitch(score_notes, perf_notes)
    score_notes = [round_note(note) for note in score_notes]
    perf_notes = [round_note(note) for note in perf_notes]
    return score_notes, perf_notes, voice_info


def arrays_from_alignment(score_notes, perf_notes):
    n = len(score_notes)
    return {
        "voice": np.array([int(getattr(note, "voice_id", 0)) for note in score_notes], dtype=np.int16),
        "bar_phase": np.array([float(getattr(note, "bar_phase", 0.0)) for note in score_notes], dtype=np.float32),
        "meter_known": np.array([float(getattr(note, "meter_known", 0.0)) for note in score_notes], dtype=np.float32),
        "beats_per_bar": np.array([float(getattr(note, "beats_per_bar", 0.0)) for note in score_notes], dtype=np.float32),
        "beat_denom": np.array([float(getattr(note, "beat_denom", 0.0)) for note in score_notes], dtype=np.float32),
        "perf_duration": np.array([float(note.duration) for note in perf_notes], dtype=np.float32),
    }, n


def load_structure(file_name: str):
    path = STRUCT_DIR / file_name
    if not path.exists():
        return None
    data = np.load(path)
    return {key: data[key] for key in data.files}


def xml_path_for(midi_score: str) -> Path:
    return BASE / Path(midi_score).parent / "xml_score.musicxml"


def build(limit=None):
    index = pd.read_csv(PROCESSED / "index.csv")
    with open(BASE / "asap_annotations.json") as f:
        annotations = json.load(f)
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        index = index.iloc[:limit]

    rows = []
    written = 0
    for row in index.itertuples(index=False):
        path = PROCESSED / "pairs" / row.file
        with np.load(path) as data:
            stored_pitch = np.array(data["pitch"])
        ann = annotations.get(row.midi_performance, {})
        try:
            score_pm = pmd.PrettyMIDI(str(BASE / row.midi_score))
            perf_pm = pmd.PrettyMIDI(str(BASE / row.midi_performance))
            score_notes, perf_notes, voice_info = align_annotated(
                score_pm, perf_pm, ann, xml_path_for(row.midi_score)
            )
        except Exception as exc:
            print("SKIP", row.file, exc, flush=True)
            rows.append({"file": row.file, "status": "skip", "error": str(exc)})
            continue
        pitch = np.array([note.pitch for note in score_notes], dtype=np.int16)
        if len(pitch) != len(stored_pitch) or not np.array_equal(pitch, stored_pitch):
            print("SKIP", row.file, "pitch alignment does not match the saved pair", flush=True)
            rows.append({"file": row.file, "status": "pitch_mismatch"})
            continue
        arrays, n = arrays_from_alignment(score_notes, perf_notes)
        np.savez(STRUCT_DIR / row.file, **arrays)
        written += 1
        rec = {
            "file": row.file,
            "status": "ok",
            "composer": row.composer,
            "title": row.title,
            "split": row.split,
            "n_notes": int(n),
            "voice_match_rate": voice_info["match_rate"],
            "voice_kept": bool(voice_info["kept"]),
            "voice_labeled": voice_info["labeled"],
            "meter_frac": float(arrays["meter_known"].mean()) if n else 0.0,
        }
        rows.append(rec)
        if written % 25 == 0:
            print(f"wrote {written}", flush=True)

    report = {
        "written": written,
        "files": rows,
    }
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        report["summary"] = {
            "n": len(ok),
            "voice_kept_files": int(sum(r["voice_kept"] for r in ok)),
            "mean_voice_labeled_when_kept": float(
                np.mean([r["voice_labeled"] for r in ok if r["voice_kept"]] or [0.0])
            ),
            "mean_meter_frac": float(np.mean([r["meter_frac"] for r in ok])),
            "mean_match_rate": float(np.mean([r["voice_match_rate"] for r in ok])),
        }
        print(json.dumps(report["summary"], indent=2), flush=True)
    out = STRUCT_DIR / ("coverage_partial.json" if limit is not None else "coverage.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {written} structure files, report {out}", flush=True)
    return report


if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build(cap)
