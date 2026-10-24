"""ASAP score/performance pairing for velocity (then timing) models."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pretty_midi as pmd


def flatten_notes(midi: pmd.PrettyMIDI) -> list:
    """Merge all non-drum tracks into one staff, sorted by onset then pitch."""
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append(
                SimpleNamespace(
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                    duration=float(note.end - note.start),
                    velocity=int(note.velocity),
                )
            )
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _copy_note(note, **updates):
    data = vars(note).copy()
    data.update(updates)
    return SimpleNamespace(**data)


def shift_to_zero(notes: list) -> list:
    if not notes:
        return []
    origin = notes[0].start
    shifted = []
    for note in notes:
        start = note.start - origin
        end = note.end - origin
        shifted.append(_copy_note(note, start=start, end=end, duration=end - start))
    return shifted


def piece_length(notes: list) -> float:
    return max(note.end for note in notes)


def stretch_notes(notes: list, ratio: float) -> list:
    stretched = []
    for note in notes:
        start = note.start * ratio
        end = note.end * ratio
        stretched.append(_copy_note(note, start=start, end=end, duration=end - start))
    return stretched


def _pair_all_short(short_idxs, long_idxs, short_notes, long_notes):
    """Monotone min-|dt| assignment of every short note onto the longer list."""
    n = len(short_idxs)
    m = len(long_idxs)
    if n == 0 or m == 0:
        return []
    if n == m:
        return list(zip(short_idxs, long_idxs))

    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0.0

    for i in range(1, n + 1):
        short_t = short_notes[short_idxs[i - 1]].start
        for j in range(i, m + 1):
            skip = dp[i][j - 1]
            pair = dp[i - 1][j - 1] + abs(short_t - long_notes[long_idxs[j - 1]].start)
            if pair <= skip:
                dp[i][j] = pair
                take[i][j] = True
            else:
                dp[i][j] = skip

    pairs = []
    i, j = n, m
    while i > 0:
        if take[i][j]:
            pairs.append((short_idxs[i - 1], long_idxs[j - 1]))
            i -= 1
            j -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def align_by_pitch(score_notes, perf_notes):
    """One-for-one pitch match: pair each pitch's notes by stretched time."""
    score_by_pitch = defaultdict(list)
    perf_by_pitch = defaultdict(list)
    for idx, note in enumerate(score_notes):
        score_by_pitch[note.pitch].append(idx)
    for idx, note in enumerate(perf_notes):
        perf_by_pitch[note.pitch].append(idx)

    pairs = []
    for pitch, score_idxs in score_by_pitch.items():
        perf_idxs = perf_by_pitch.get(pitch, [])
        if not perf_idxs:
            continue
        if len(score_idxs) <= len(perf_idxs):
            pairs.extend(_pair_all_short(score_idxs, perf_idxs, score_notes, perf_notes))
        else:
            swapped = _pair_all_short(perf_idxs, score_idxs, perf_notes, score_notes)
            pairs.extend((score_idx, perf_idx) for perf_idx, score_idx in swapped)

    if not pairs:
        raise ValueError("No one-for-one pitch matches between score and performance")

    pairs.sort(key=lambda pair: (score_notes[pair[0]].start, score_notes[pair[0]].pitch))
    aligned_score = [score_notes[i] for i, _ in pairs]
    aligned_perf = [perf_notes[j] for _, j in pairs]
    if any(s.pitch != p.pitch for s, p in zip(aligned_score, aligned_perf)):
        raise ValueError("Pitch alignment failed")
    return aligned_score, aligned_perf


def round_note(note):
    start = round(note.start, 4)
    end = round(note.end, 4)
    return _copy_note(note, start=start, end=end, duration=round(end - start, 4))


def prepare_pair(score_midi: pmd.PrettyMIDI, perf_midi: pmd.PrettyMIDI):
    """Merge staves, put first notes at 0, stretch score to perf length, match pitches 1-1."""
    score_notes = shift_to_zero(flatten_notes(score_midi))
    perf_notes = shift_to_zero(flatten_notes(perf_midi))
    if not score_notes or not perf_notes:
        raise ValueError("Score or performance has no notes")

    score_len = piece_length(score_notes)
    perf_len = piece_length(perf_notes)
    if score_len <= 0:
        raise ValueError("Score length is zero; cannot stretch")

    score_notes = stretch_notes(score_notes, perf_len / score_len)
    score_notes, perf_notes = align_by_pitch(score_notes, perf_notes)
    return [round_note(n) for n in score_notes], [round_note(n) for n in perf_notes]
