"""Audit the selected videos: visual counts, object sizes, tracking and PPE gaps."""

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def diagnostics(root, manifest, audit, report):
    predicted, object_rows, track_rows, frame_rows, timeline_rows = {}, [], [], [], []
    for video in manifest["videos"]:
        name = video["video"]
        wanted = {int(row["frame"]) for row in audit if row["video"] == name}
        observations, buckets, by_class = defaultdict(list), defaultdict(Counter), defaultdict(list)
        frames_without_person = untracked = 0
        with gzip.open(root / (Path(name).stem + ".jsonl.gz"), "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                counts = Counter(obj["label"] for obj in record["objects"])
                if record["frame"] in wanted:
                    predicted[name, record["frame"]] = counts
                bucket = buckets[int(record["frame"] / video["fps"])]
                bucket.update(counts)
                bucket["frames"] += 1
                frames_without_person += counts["person"] == 0
                for obj in record["objects"]:
                    by_class[obj["label"]].append(obj)
                    if obj["label"] != "person":
                        continue
                    for kind in ("helmet", "vest"):
                        bucket[kind + "_unknown"] += obj[kind] == "unknown"
                    if obj["track_id"] is None:
                        untracked += 1
                    else:
                        observations[obj["track_id"]].append(
                            (record["frame"], obj["helmet"], obj["vest"])
                        )
        for label in manifest["classes"].values():
            objects = by_class[label]
            heights = [obj["bbox"][3] - obj["bbox"][1] for obj in objects]
            object_rows.append(
                {
                    "video": name,
                    "class": label,
                    "boxes": len(objects),
                    "mean_confidence": round(float(np.mean([o["confidence"] for o in objects])), 3)
                    if objects
                    else None,
                    "median_height_px": round(float(np.median(heights)), 2) if objects else None,
                    "height_below_32_pct": round(
                        sum(h < 32 for h in heights) / len(heights) * 100, 2
                    )
                    if heights
                    else None,
                    "unassociated_ppe_boxes": sum("person_index" not in o for o in objects)
                    if label != "person"
                    else None,
                }
            )
        for track_id, values in sorted(observations.items()):
            runs, longest, previous = [0, 0], [0, 0], -2
            for index, helmet, vest in values:
                for kind, state in enumerate((helmet, vest)):
                    runs[kind] = (
                        (runs[kind] + 1 if index == previous + 1 else 1) if state == "absent" else 0
                    )
                    longest[kind] = max(longest[kind], runs[kind])
                previous = index
            span = values[-1][0] - values[0][0] + 1
            track_rows.append(
                {
                    "video": name,
                    "track_id": track_id,
                    "first_s": round(values[0][0] / video["fps"], 3),
                    "last_s": round(values[-1][0] / video["fps"], 3),
                    "observations": len(values),
                    "coverage_pct": round(len(values) / span * 100, 2),
                    "helmet_unknown_pct": round(
                        sum(v[1] == "unknown" for v in values) / len(values) * 100, 2
                    ),
                    "vest_unknown_pct": round(
                        sum(v[2] == "unknown" for v in values) / len(values) * 100, 2
                    ),
                    "max_helmet_absent_s": round(longest[0] / video["fps"], 3),
                    "max_vest_absent_s": round(longest[1] / video["fps"], 3),
                }
            )
        frame_rows.append(
            {
                "video": name,
                "frames": video["frames"],
                "frames_without_person": frames_without_person,
                "untracked_person_boxes": untracked,
                "confirmed_track_ids": len(observations),
            }
        )
        for second, bucket in sorted(buckets.items()):
            timeline_rows.append(
                {
                    "video": name,
                    "second": second,
                    "frames": bucket["frames"],
                    **{
                        kind + "_mean": round(bucket[kind] / bucket["frames"], 3)
                        for kind in ("person", "helmet", "vest")
                    },
                    **{
                        kind + "_unknown_pct": round(
                            bucket[kind + "_unknown"] / bucket["person"] * 100, 2
                        )
                        if bucket["person"]
                        else None
                        for kind in ("helmet", "vest")
                    },
                }
            )
    for filename, rows in [
        ("object_stats.csv", object_rows),
        ("tracks.csv", track_rows),
        ("frame_stats.csv", frame_rows),
        ("timeline.csv", timeline_rows),
    ]:
        write_csv(report / filename, rows)
    fig, axes = plt.subplots(
        2, len(manifest["videos"]), figsize=(11, 7), squeeze=False, layout="constrained"
    )
    for column, video in enumerate(manifest["videos"]):
        values = [row for row in timeline_rows if row["video"] == video["video"]]
        for kind in ("person", "helmet", "vest"):
            axes[0, column].plot(
                [r["second"] for r in values], [r[kind + "_mean"] for r in values], label=kind
            )
        for kind in ("helmet", "vest"):
            axes[1, column].plot(
                [r["second"] for r in values],
                [r[kind + "_unknown_pct"] for r in values],
                label=kind,
            )
        axes[0, column].set(title=video["video"], ylabel="Mean detections per frame")
        axes[1, column].set(
            xlabel="Video time, seconds", ylabel="Unknown PPE status, %", ylim=(0, 105)
        )
        axes[0, column].legend(fontsize=8)
        axes[1, column].legend(fontsize=8)
    fig.savefig(report / "detection_timeline.png", dpi=140)
    plt.close(fig)
    return predicted


def main():
    root = Path("data/annotations")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    audit = list(csv.DictReader((root / "visual_audit.csv").open(encoding="utf-8")))
    names = {row["video"] for row in manifest["videos"]}
    if not audit or any(row["video"] not in names for row in audit):
        raise ValueError("Visual audit must cover only videos in the current manifest")
    report = Path("reports/annotations")
    report.mkdir(parents=True, exist_ok=True)
    predicted = diagnostics(root, manifest, audit, report)
    rows = []
    for label in ("person", "helmet", "vest"):
        refs = [int(row[label]) for row in audit]
        preds = [predicted[row["video"], int(row["frame"])][label] for row in audit]
        rows.append(
            {
                "class": label,
                "audit_frames": len(audit),
                "visual_count": sum(refs),
                "auto_count": sum(preds),
                "count_mae": round(
                    sum(abs(a - b) for a, b in zip(refs, preds, strict=True)) / len(audit), 3
                ),
            }
        )
    write_csv(report / "audit_summary.csv", rows)
    by_video = []
    for name in sorted(names):
        selected = [row for row in audit if row["video"] == name]
        for label in ("person", "helmet", "vest"):
            refs = [int(row[label]) for row in selected]
            preds = [predicted[name, int(row["frame"])][label] for row in selected]
            by_video.append(
                {
                    "video": name,
                    "class": label,
                    "audit_frames": len(selected),
                    "visual_count": sum(refs),
                    "auto_count": sum(preds),
                    "count_mae": round(
                        sum(abs(a - b) for a, b in zip(refs, preds, strict=True)) / len(selected), 3
                    ),
                }
            )
    write_csv(report / "audit_by_video.csv", by_video)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    names = [row["class"] for row in rows]
    axes[0].bar(
        [i - 0.18 for i in range(3)],
        [row["visual_count"] for row in rows],
        width=0.36,
        label="Visual audit (assistant)",
    )
    axes[0].bar(
        [i + 0.18 for i in range(3)],
        [row["auto_count"] for row in rows],
        width=0.36,
        label="Automatic prelabels",
    )
    axes[0].set_xticks(range(3), names)
    axes[0].set(title=f"{len(audit)} sampled frames", ylabel="Object appearances")
    axes[0].legend(fontsize=8)
    axes[1].bar(names, [row["count_mae"] for row in rows])
    axes[1].set(title="Count difference per frame", ylabel="Mean absolute difference")
    fig.savefig(report / "quality.png", dpi=140)
    plt.close(fig)
    print(rows)


if __name__ == "__main__":
    main()
