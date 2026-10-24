"""
Per-note inputs.

The original four timing columns come from the notes alone. A test file only
has to contain notes:

  pitch            MIDI key, 0–127
  duration_beats   how long the grid holds the note
  beat_phase       where it sits inside the beat, 0 to 1
  onset_gap_beats  how long the grid says the gap is since the previous attack
  chord_height     pitch rank inside the attack, 0 at the bass to 1 at the top (0.5 alone)

The structure experiment adds clues. They are not rules: a downbeat or an
octave doubling is not told to be louder.

  bar_phase        position inside the measure, 0 at the downbeat, 1 at the next
  meter_known      1 when bar_phase and the time signature are real, else 0
  beats_per_bar    ASAP's beat count for the bar (2 in 6/8, 4 in 4/4), / 12
  beat_denom       log2 of the time-signature denominator, / 4; 0 if unknown
  chord_size       log1p(notes in the attack) / log1p(12)
  octave_partner   1 when another note in the attack is 12, 24, or 36 away
  register_change  this attack's mean pitch minus the previous, in octaves, clipped
  density_change   this attack's size minus the previous, / 6, clipped to [-1, 1]

bar_phase is 0 when meter_known is 0, so a missing bar is not a fake downbeat.
beat_phase already separates on-beats from off-beats. bar_phase says which beat
in the bar that is. Voice identity is a separate integer, not one of these
columns: 0 means unknown.

Notes that land together share one gap, measured back to the previous attack.
A chord is not a run of zeros. The small spread inside the chord is the onset
offset, which is a target, not an input.
"""

import numpy as np

from grid_stability import candidate_quanta, cluster_onsets, explain_rate

# MIDI pitch % 12. C is 0.
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Notes closer than this are one attack when no subdivision is known.
CHORD_TOL = 0.035

# Columns after the original four. Order is part of the checkpoint.
EXTRA_NAMES = (
    "bar_phase",
    "meter_known",
    "beats_per_bar",
    "beat_denom",
    "chord_size",
    "octave_partner",
    "register_change",
    "density_change",
)
N_EXTRA = len(EXTRA_NAMES)
N_TIMING_STRUCT = 4 + N_EXTRA
# 0 = unknown. Staffs 1–4 and voices 1–8 pack into 1..32.
N_VOICE = 33
OCTAVE_STEPS = (12, 24, 36)


def beat_seconds(starts):
    """Length of one beat, in seconds. Falls back to 0.5s when the grid is unclear."""
    onsets = cluster_onsets(starts)
    if len(onsets) < 16:
        return 0.5
    quanta = candidate_quanta(onsets)
    if not quanta:
        return 0.5
    candidates = []
    for quantum in quanta:
        for multiple in (1, 2, 3, 4):
            beat = quantum * multiple
            if 0.3 <= beat <= 1.2:
                candidates.append(beat)
    if not candidates:
        quantum = max(quanta, key=lambda q: explain_rate(onsets, q))
        return float(np.clip(quantum, 0.3, 1.2))

    def score(beat):
        return (explain_rate(onsets, beat), -abs(np.log(beat / 0.5)))

    return float(max(candidates, key=score))


def _subdivision(onsets):
    quanta = candidate_quanta(onsets)
    if not quanta:
        return None
    return float(max(quanta, key=lambda q: explain_rate(onsets, q)))


def _phase_zero(onsets, beat, quantum):
    """Time at which beat phase is 0. Chosen so the most onsets sit on the pulse."""
    if len(onsets) == 0:
        return 0.0
    step = quantum if quantum is not None and 0.02 < quantum < beat else beat / 8.0
    n = int(np.ceil(beat / step))
    n = min(max(n, 1), 32)
    step = beat / n
    base = float(onsets[0])
    best_t0 = base
    best = -1.0
    for i in range(n):
        t0 = base - i * step
        residue = np.mod(onsets - t0, beat)
        dist = np.minimum(residue, beat - residue) / beat
        score = float((dist <= 0.12).mean())
        if score > best:
            best = score
            best_t0 = t0
    return best_t0


def pulse(starts):
    """Beat length in seconds, time of phase 0, and the subdivision length."""
    onsets = cluster_onsets(starts)
    beat = float(beat_seconds(starts))
    if len(onsets) == 0:
        return beat, 0.0, beat
    quantum = _subdivision(onsets)
    if quantum is None or quantum <= 0:
        quantum = beat
    t0 = _phase_zero(onsets, beat, quantum)
    return beat, t0, float(quantum)


def onset_groups(starts, tol):
    """Cluster notes into attacks. Every note in a cluster shares the cluster's start time."""
    n = len(starts)
    onset_id = np.zeros(n, dtype=np.int64)
    onset_time = np.zeros(n, dtype=np.float64)
    if n == 0:
        return onset_id, onset_time
    order = np.argsort(starts, kind="mergesort")
    cid = 0
    cluster_start = float(starts[order[0]])
    members = [int(order[0])]

    def flush(members, cid, cluster_start):
        for j in members:
            onset_id[j] = cid
            onset_time[j] = cluster_start

    for raw in order[1:]:
        i = int(raw)
        if float(starts[i]) - cluster_start >= tol:
            flush(members, cid, cluster_start)
            cid += 1
            cluster_start = float(starts[i])
            members = [i]
        else:
            members.append(i)
    flush(members, cid, cluster_start)
    return onset_id, onset_time


def onset_gap_beats(onset_id, onset_time, beat):
    """Gap from the previous attack, in beats. Shared by every note in the attack."""
    n = len(onset_id)
    gap = np.zeros(n, dtype=np.float32)
    if n == 0:
        return gap
    n_onsets = int(onset_id.max()) + 1
    attack = np.empty(n_onsets, dtype=np.float64)
    attack[onset_id] = onset_time
    gap_sec = np.zeros(n_onsets, dtype=np.float64)
    gap_sec[1:] = np.diff(attack)
    gap[:] = gap_sec[onset_id] / beat
    return gap


def pack_voice(staff, voice):
    """Integer id for a MusicXML staff and voice. 0 means unknown or out of range."""
    if staff is None or voice is None:
        return 0
    staff = int(staff)
    voice = int(voice)
    if not (1 <= staff <= 4 and 1 <= voice <= 8):
        return 0
    return (staff - 1) * 8 + voice


def _attack_values(onset_id, pitch):
    """Per-note chord size, octave doubling, and changes from the previous attack."""
    n = len(onset_id)
    size = np.ones(n, dtype=np.float32)
    octave = np.zeros(n, dtype=np.float32)
    register = np.zeros(n, dtype=np.float32)
    density = np.zeros(n, dtype=np.float32)
    if n == 0:
        return size, octave, register, density

    order = np.argsort(onset_id, kind="mergesort")
    sorted_oid = onset_id[order]
    splits = np.flatnonzero(np.diff(sorted_oid)) + 1
    starts = np.concatenate(([0], splits))
    ends = np.concatenate((splits, [n]))
    n_attacks = len(starts)
    count = np.zeros(n_attacks, dtype=np.float32)
    mean_pitch = np.zeros(n_attacks, dtype=np.float32)
    has_octave = np.zeros(n_attacks, dtype=np.float32)
    for a, (s, e) in enumerate(zip(starts, ends)):
        idx = order[s:e]
        pitches = pitch[idx]
        count[a] = float(len(idx))
        mean_pitch[a] = float(pitches.mean())
        doubled = False
        if len(idx) > 1:
            present = set(int(p) for p in pitches)
            doubled = any((int(p) + step) in present for p in pitches for step in OCTAVE_STEPS)
        has_octave[a] = 1.0 if doubled else 0.0
        size[idx] = count[a]
        octave[idx] = has_octave[a]
    for a, (s, e) in enumerate(zip(starts, ends)):
        idx = order[s:e]
        if a == 0:
            continue
        reg = (mean_pitch[a] - mean_pitch[a - 1]) / 12.0
        den = (count[a] - count[a - 1]) / 6.0
        register[idx] = np.float32(np.clip(reg, -2.0, 2.0))
        density[idx] = np.float32(np.clip(den, -1.0, 1.0))
    return size, octave, register, density


def extra_timing(onset_id, pitch, bar_phase=None, meter_known=None, beats_per_bar=None, beat_denom=None):
    """Eight structure columns. Missing meter is zeros, not a pretend downbeat."""
    n = len(onset_id)
    if bar_phase is None:
        bar_phase = np.zeros(n, dtype=np.float32)
        meter_known = np.zeros(n, dtype=np.float32)
        beats_per_bar = np.zeros(n, dtype=np.float32)
        beat_denom = np.zeros(n, dtype=np.float32)
    bar_phase = np.asarray(bar_phase, dtype=np.float32)
    meter_known = np.asarray(meter_known, dtype=np.float32)
    beats_per_bar = np.asarray(beats_per_bar, dtype=np.float32)
    beat_denom = np.asarray(beat_denom, dtype=np.float32)
    known = meter_known > 0.5
    phase = np.where(known, np.clip(bar_phase, 0.0, 0.999), 0.0).astype(np.float32)
    beats = np.where(known, beats_per_bar / 12.0, 0.0).astype(np.float32)
    denom = np.zeros(n, dtype=np.float32)
    if np.any(known):
        safe = np.maximum(beat_denom, 1.0)
        denom = np.where(known, np.log2(safe) / 4.0, 0.0).astype(np.float32)
    size, octave, register, density = _attack_values(onset_id, pitch)
    chord = (np.log1p(size) / np.log1p(12.0)).astype(np.float32)
    extra = np.stack(
        [phase, known.astype(np.float32), beats, denom, chord, octave, register, density],
        axis=-1,
    )
    return np.nan_to_num(extra).astype(np.float32)


def assemble_timing(built, structure=None):
    """Original four timing columns plus the eight structure clues. Voice is separate."""
    pitch = built["pitch"]
    onset_id = built["onset_id"]
    n = len(pitch)
    if structure is None:
        extra = extra_timing(onset_id, pitch)
        voice = np.zeros(n, dtype=np.int64)
    else:
        extra = extra_timing(
            onset_id,
            pitch,
            structure["bar_phase"],
            structure["meter_known"],
            structure["beats_per_bar"],
            structure["beat_denom"],
        )
        voice = np.asarray(structure["voice"], dtype=np.int64)
        if len(voice) != n:
            raise ValueError(f"voice length {len(voice)} != {n} notes")
    timing = np.concatenate([built["timing"], extra], axis=-1).astype(np.float32)
    return timing, voice


def chord_height(onset_id, pitch):
    """Pitch rank inside each attack: 0 at the bass, 1 at the top, 0.5 when alone."""
    n = len(onset_id)
    height = np.full(n, 0.5, dtype=np.float32)
    if n == 0:
        return height
    order = np.lexsort((pitch, onset_id))
    sorted_oid = onset_id[order]
    splits = np.flatnonzero(np.diff(sorted_oid)) + 1
    starts = np.concatenate(([0], splits))
    ends = np.concatenate((splits, [n]))
    for s, e in zip(starts, ends):
        k = e - s
        if k > 1:
            height[order[s:e]] = np.linspace(0.0, 1.0, k, dtype=np.float32)
    return height


def snap_notes(start, duration, t0, quantum):
    """Move each note onto the subdivision grid. Duration stays at least one subdivision."""
    quantum = max(float(quantum), 1e-3)
    idx = np.round((start - t0) / quantum)
    snapped = t0 + idx * quantum
    end = start + duration
    end_idx = np.maximum(np.round((end - t0) / quantum), idx + 1)
    snapped_dur = (t0 + end_idx * quantum) - snapped
    return snapped, snapped_dur


def ambiguous_arpeggio(starts, pitches, tol=CHORD_TOL):
    """A rolled figure spread across the chord window, not one simultaneous attack.

    Notes already outside the chord tolerance are separate attacks and never
    reach this test. A tight cluster, including a chord voiced low to high
    within a few milliseconds, is not an arpeggio.
    """
    starts = np.asarray(starts, dtype=np.float64)
    pitches = np.asarray(pitches, dtype=np.int64)
    if len(starts) < 3:
        return False
    order = np.argsort(starts, kind="mergesort")
    times = starts[order]
    ordered_pitch = pitches[order]
    span = float(times[-1] - times[0])
    if span < 0.5 * tol:
        return False
    gaps = np.diff(times)
    if np.any(gaps < 0.008):
        return False
    rising = bool(np.all(np.diff(ordered_pitch) > 0))
    falling = bool(np.all(np.diff(ordered_pitch) < 0))
    return rising or falling


def snap_shared_attacks(start, duration, pitch, t0, quantum, tol=CHORD_TOL):
    """Give each confident attack one grid onset. Keep every note in place.

    Attack groups are the same 35 ms clusters used on the diagnostic clips.
    The shared onset is the grid point nearest the group's median performed
    onset. A later distinct attack that would land on an occupied grid point
    moves to the next free point instead of merging. Ambiguous arpeggios keep
    the independent snap. Pitch, order, and length are unchanged.
    """
    start = np.asarray(start, dtype=np.float64)
    duration = np.maximum(np.asarray(duration, dtype=np.float64), 0.0)
    pitch = np.asarray(pitch, dtype=np.int64)
    quantum = max(float(quantum), 1e-3)
    n = len(start)
    grid_start, grid_dur = snap_notes(start, duration, t0, quantum)
    if n == 0:
        return grid_start, grid_dur
    onset_id, _onset_time = onset_groups(start, tol)
    claimed = set()
    groups = []
    for gid in np.unique(onset_id):
        members = np.flatnonzero(onset_id == gid)
        median = float(np.median(start[members]))
        groups.append((median, int(gid), members))
    groups.sort(key=lambda item: (item[0], item[1]))
    for median, _gid, members in groups:
        if len(members) >= 2 and not ambiguous_arpeggio(start[members], pitch[members], tol):
            preferred = int(np.round((median - t0) / quantum))
            idx = preferred
            while idx in claimed:
                idx += 1
            claimed.add(idx)
            shared = t0 + idx * quantum
            for note_i in members:
                end = float(start[note_i] + duration[note_i])
                end_idx = int(np.round((end - t0) / quantum))
                if end_idx <= idx:
                    end_idx = idx + 1
                grid_start[note_i] = shared
                grid_dur[note_i] = (t0 + end_idx * quantum) - shared
            continue
        for note_i in members:
            idx = int(np.round((float(grid_start[note_i]) - t0) / quantum))
            if len(members) == 1:
                while idx in claimed:
                    idx += 1
                shared = t0 + idx * quantum
                end = float(start[note_i] + duration[note_i])
                end_idx = int(np.round((end - t0) / quantum))
                if end_idx <= idx:
                    end_idx = idx + 1
                grid_start[note_i] = shared
                grid_dur[note_i] = (t0 + end_idx * quantum) - shared
            claimed.add(idx)
    if len(grid_start) != n or len(grid_dur) != n:
        raise RuntimeError("shared-attack snap changed the note count")
    return grid_start, grid_dur


def local_onset_offset(score_start, perf_start, beat, half_window_beats=4.0):
    """Onset offset in beats, after removing the local tempo.

    A raw performance-minus-score time drifts by seconds when the pianist
    slows down. The offset kept here is the note's deviation from the median
    drift of the surrounding four beats. That is the early/late of one note,
    not the tempo curve.
    """
    score = np.asarray(score_start, dtype=np.float64)
    perf = np.asarray(perf_start, dtype=np.float64)
    n = len(score)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    dt = perf - score
    half = max(float(beat), 1e-3) * half_window_beats
    order = np.argsort(score, kind="mergesort")
    ordered_score = score[order]
    ordered_dt = dt[order]
    left = 0
    right = 0
    for k in range(n):
        while ordered_score[left] < ordered_score[k] - half:
            left += 1
        while right < n and ordered_score[right] <= ordered_score[k] + half:
            right += 1
        out[order[k]] = (ordered_dt[k] - np.median(ordered_dt[left:right])) / beat
    return out


def note_features(pitch, start, duration, snap=False, shared_attacks=False):
    """Inputs for one piece.

    snap=False: the times are already the grid (a score).
    snap=True: the times are a performance. Build the grid, then measure the
    onset offset from it. That offset is a target. The inputs use the grid.
    shared_attacks=True snaps each confident 35 ms attack to one grid onset.
    The default leaves the independent per-note snap unchanged.
    """
    pitch = np.clip(np.asarray(pitch).astype(np.int64), 0, 127)
    start = np.asarray(start, dtype=np.float64)
    duration = np.maximum(np.asarray(duration, dtype=np.float64), 0.0)
    n = len(pitch)
    if n == 0:
        return {
            "pitch": pitch,
            "timing": np.zeros((0, 4), dtype=np.float32),
            "beat_pos": np.zeros(0, dtype=np.float32),
            "onset_id": np.zeros(0, dtype=np.int64),
            "offset_beats": np.zeros(0, dtype=np.float32),
            "beat_seconds": 0.5,
            "grid_start": np.zeros(0, dtype=np.float64),
            "grid_duration": np.zeros(0, dtype=np.float64),
        }

    beat, t0, quantum = pulse(start)
    beat = max(beat, 1e-3)
    if snap and shared_attacks:
        grid_start, grid_dur = snap_shared_attacks(start, duration, pitch, t0, quantum)
        offset = ((start - grid_start) / beat).astype(np.float32)
    elif snap:
        grid_start, grid_dur = snap_notes(start, duration, t0, quantum)
        offset = ((start - grid_start) / beat).astype(np.float32)
    else:
        grid_start = start
        grid_dur = duration
        offset = np.zeros(n, dtype=np.float32)

    # 35ms is one attack. Anything wider is a real gap, including a short note.
    onset_id, onset_time = onset_groups(grid_start, CHORD_TOL)
    gap = onset_gap_beats(onset_id, onset_time, beat)
    height = chord_height(onset_id, pitch)
    phase = (np.mod(grid_start - t0, beat) / beat).astype(np.float32)
    beat_pos = ((grid_start - t0) / beat).astype(np.float32)
    timing = np.stack(
        [grid_dur / beat, phase, gap, height],
        axis=-1,
    ).astype(np.float32)
    return {
        "pitch": pitch,
        "timing": np.nan_to_num(timing),
        "beat_pos": np.nan_to_num(beat_pos),
        "onset_id": onset_id,
        "offset_beats": np.nan_to_num(offset),
        "beat_seconds": beat,
        "grid_start": grid_start,
        "grid_duration": grid_dur,
    }


if __name__ == "__main__":
    # Quarters at 0.5s, plus a three-note chord 8 beats in.
    starts = list(np.arange(0.0, 8.0, 0.5))
    starts += [8.0, 8.004, 8.008, 8.5]
    pitch = np.arange(len(starts)) % 12 + 60
    duration = np.full(len(starts), 0.4)
    built = note_features(pitch, starts, duration, snap=False)
    chord = built["timing"][-4:-1, 2]
    after = built["timing"][-1, 2]
    print("chord gaps", chord, "next gap", after)
    if np.max(chord) - np.min(chord) > 1e-5:
        raise SystemExit("chord notes did not share a gap")
    if abs(float(chord[0]) - 1.0) > 0.15:
        raise SystemExit(f"expected about 1 beat, got {chord[0]}")

    # Octave doubling inside one attack, then a thicker higher attack.
    pitches = np.array([48, 60, 72, 67, 79])
    starts2 = np.array([0.0, 0.0, 0.01, 0.5, 0.5])
    durs = np.full(5, 0.4)
    built2 = note_features(pitches, starts2, durs, snap=False)
    bar = np.array([0.0, 0.0, 0.0, 0.25, 0.25], dtype=np.float32)
    known = np.ones(5, dtype=np.float32)
    beats = np.full(5, 4.0, dtype=np.float32)
    denom = np.full(5, 4.0, dtype=np.float32)
    extra = extra_timing(built2["onset_id"], built2["pitch"], bar, known, beats, denom)
    if extra.shape != (5, N_EXTRA):
        raise SystemExit(f"extra shape {extra.shape}")
    if extra[:3, 5].min() < 0.5:
        raise SystemExit("octave partner missing on the doubled chord")
    if extra[3, 6] <= 0:
        raise SystemExit("expected a rise in register on the second attack")
    if extra[3, 7] >= 0:
        raise SystemExit("expected a drop in density on the second attack")
    unknown = extra_timing(built2["onset_id"], built2["pitch"])
    if unknown[:, 1].max() != 0 or unknown[:, 0].max() != 0:
        raise SystemExit("missing meter looked like a downbeat")
    # Two notes on opposite sides of a grid line are one attack.
    boundary = note_features(
        np.array([60, 64]),
        np.array([0.37, 0.38]),
        np.array([0.2, 0.2]),
        snap=True,
        shared_attacks=True,
    )
    if abs(float(boundary["grid_start"][0] - boundary["grid_start"][1])) > 1e-9:
        raise SystemExit("confident chord was split across the grid")
    print("ok")
