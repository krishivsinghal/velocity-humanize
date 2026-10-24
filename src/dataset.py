from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/processed")
GIGAMIDI = Path("/Users/krishivsinghal/Desktop/AIR/music_humanize/data/gigamidi")


class VelocityDataset:
    def __init__(self, processed_dir=PROCESSED, split=None):
        self.processed_dir = Path(processed_dir)
        self.index = pd.read_csv(self.processed_dir / "index.csv")
        if split is not None:
            self.index = self.index[self.index["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        row = self.index.iloc[i]
        data = np.load(self.processed_dir / "pairs" / row["file"])
        if "perf_start" not in data.files:
            raise RuntimeError(
                f"{row['file']} has no perf_start. Run: PYTHONPATH=src python src/add_perf_start.py"
            )
        return {
            "pitch": data["pitch"],
            "start": data["start"],
            "duration": data["duration"],
            "velocity": data["velocity"],
            "perf_start": data["perf_start"],
            "snap": False,
            "composer": row["composer"],
            "title": row["title"],
            "split": row["split"],
        }


class SpanDataset:
    """One steady GigaMIDI stretch. The file holds the whole piece; the row is a time window."""

    def __init__(self, split=None):
        self.root = GIGAMIDI / "pairs"
        self.index = pd.read_csv(GIGAMIDI / "index_spans.csv")
        if split is not None:
            self.index = self.index[self.index["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        row = self.index.iloc[i]
        data = np.load(self.root / row["file"])
        start = data["start"]
        keep = (start >= row["t0"]) & (start < row["t1"])
        played = start[keep]
        return {
            "pitch": data["pitch"][keep],
            "start": played,
            "duration": data["duration"][keep],
            "velocity": data["velocity"][keep],
            "perf_start": played,
            "snap": True,
            "composer": row["composer"],
            "title": row["title"],
            "split": row["split"],
        }


class CombinedDataset:
    def __init__(self, parts):
        self.parts = parts
        self.sizes = [len(part) for part in parts]

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, i):
        for part, size in zip(self.parts, self.sizes):
            if i < size:
                return part[i]
            i -= size
        raise IndexError(i)


def all_notes(split):
    """ASAP pairs plus the steady GigaMIDI spans."""
    return CombinedDataset([
        VelocityDataset(split=split),
        SpanDataset(split=split),
    ])


if __name__ == "__main__":
    data = VelocityDataset(split="train")
    example = data[0]
    print(len(data), "train pieces")
    print(example["composer"], example["title"])
    print("n_notes", len(example["pitch"]))
    print("pitch", example["pitch"][:8])
    print("start", example["start"][:8])
    print("duration", example["duration"][:8])
    print("ioi", example["ioi"][:8])
    print("velocity", example["velocity"][:8])

