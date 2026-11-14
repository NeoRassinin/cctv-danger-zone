"""python scripts/eda.py --videos video --output reports/eda"""

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata: {path}")
    samples, seen = [], set()
    decoded = duplicates = 0
    previous = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if decoded % max(1, round(fps)) == 0:
                # Same width for comparing sharpness across different resolutions.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (640, max(1, round(height * 640 / width))))
                digest = hashlib.sha256(frame.tobytes()).hexdigest()
                duplicates += digest in seen
                seen.add(digest)
                samples.append(
                    {
                        "file": path.name,
                        "second": round(decoded / fps, 3),
                        "brightness": round(float(gray.mean()), 2),
                        "contrast": round(float(gray.std()), 2),
                        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
                        "dark_pixel_pct": round(float((gray < 30).mean() * 100), 2),
                        "bright_pixel_pct": round(float((gray > 225).mean() * 100), 2),
                        "change_from_previous": (
                            round(float(cv2.absdiff(gray, previous).mean()), 3)
                            if previous is not None
                            else None
                        ),
                    }
                )
                previous = gray
            decoded += 1
    finally:
        cap.release()
    if not samples:
        raise ValueError(f"No frames decoded: {path}")
    row = {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1_000_000, 3),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "expected_frames": expected,
        "decoded_frames": decoded,
        "duration_s": round(decoded / fps, 3),
        "samples": len(samples),
        "duplicate_samples": duplicates,
        "dark_samples": sum(s["brightness"] < 40 for s in samples),
        "low_sharpness_samples": sum(s["sharpness"] < 50 for s in samples),
        "low_contrast_samples": sum(s["contrast"] < 10 for s in samples),
        "low_change_pairs": sum(
            s["change_from_previous"] is not None and s["change_from_previous"] < 1 for s in samples
        ),
        "mean_brightness": round(float(np.mean([s["brightness"] for s in samples])), 2),
        "median_sharpness": round(float(np.median([s["sharpness"] for s in samples])), 2),
    }
    for key in ("brightness", "contrast", "sharpness", "dark_pixel_pct", "bright_pixel_pct"):
        values = [s[key] for s in samples]
        row[f"{key}_min"] = min(values)
        row[f"{key}_median"] = round(float(np.median(values)), 2)
        row[f"{key}_max"] = max(values)
    changes = [s["change_from_previous"] for s in samples if s["change_from_previous"] is not None]
    row["median_change"] = round(float(np.median(changes)), 3) if changes else None
    return row, samples


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Video inventory and sampled image quality")
    parser.add_argument("--videos", type=Path, default=Path("video"))
    parser.add_argument("--files", nargs="+", default=["Camera_0000.mp4", "input.mp4"])
    parser.add_argument("--output", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    paths = [args.videos / name for name in args.files]
    if len(paths) != len(set(paths)) or any(not path.is_file() for path in paths):
        parser.error("Input names must be unique existing video files")
    rows, samples = [], []
    for path in paths:
        row, frames = analyze(path)
        rows.append(row)
        samples.extend(frames)
        print(f"{path.name}: {row['decoded_frames']} frames, {row['duration_s']} s", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "videos.csv", rows)
    write_csv(args.output / "samples.csv", samples)

    names = [r["file"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axes[0, 0].bar(names, [r["duration_s"] for r in rows])
    axes[0, 0].set(title="Video duration", ylabel="Seconds")
    axes[0, 1].bar(names, [r["width"] * r["height"] / 1e6 for r in rows])
    axes[0, 1].set(title="Frame resolution", ylabel="Megapixels")
    for ax, key, title, label in [
        (axes[1, 0], "brightness", "Brightness: one frame per second", "Mean gray (0-255)"),
        (axes[1, 1], "sharpness", "Sharpness at width 640", "Laplacian variance"),
    ]:
        ax.boxplot([[s[key] for s in samples if s["file"] == name] for name in names])
        ax.set_xticks(range(1, len(names) + 1), names)
        ax.set(title=title, ylabel=label)
    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=30)
    fig.savefig(args.output / "overview.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, key, title in zip(
        axes.flat,
        ("brightness", "sharpness", "contrast", "change_from_previous"),
        ("Brightness", "Sharpness at width 640", "Contrast (gray std)", "Pixel change at ~1 s"),
        strict=True,
    ):
        for name in names:
            values = [s for s in samples if s["file"] == name and s[key] is not None]
            ax.plot([s["second"] for s in values], [s[key] for s in values], label=name)
        ax.set(title=title, xlabel="Video time, seconds")
        ax.legend(fontsize=8)
    fig.savefig(args.output / "quality_timeline.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
