"""Full-frame demonstration prelabels; no prediction is presented as ground truth."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".venv/ultralytics").resolve()))
os.environ["YOLO_AUTOINSTALL"] = "false"

import cv2
import numpy as np
from ultralytics import YOLO, settings
from ultralytics.trackers.byte_tracker import BYTETracker

LABELS = {0: "person", 1: "helmet", 2: "no_helmet", 3: "vest", 4: "no_vest"}
MODEL_REPO = "constructelligence/construction-site-safety-hazards"
MODEL_REVISION = "79673cff5a8bb249fc8ec9cca3daa19e2ef95191"


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def associate(objects):
    """Attach each PPE observation to at most one geometrically compatible person."""
    people = [o for o in objects if o["label"] == "person"]
    for person in people:
        person.update(helmet="unknown", vest="unknown", head_visibility="unknown")
        person["torso_visibility"] = "unknown"
    evidence = {}
    for item in objects:
        if item["label"] == "person":
            continue
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        kind = "helmet" if "helmet" in item["label"] else "vest"
        candidates = []
        for index, person in enumerate(people):
            px1, py1, px2, py2 = person["bbox"]
            pw, ph = px2 - px1, py2 - py1
            rel_y = (cy - py1) / ph
            low, high, center_y = (-0.10, 0.40, 0.12) if kind == "helmet" else (0.15, 0.80, 0.45)
            if px1 - 0.1 * pw <= cx <= px2 + 0.1 * pw and low <= rel_y <= high:
                cost = abs(cx - (px1 + px2) / 2) / pw + abs(rel_y - center_y)
                candidates.append((cost, index))
        if candidates:
            _, index = min(candidates)
            item["person_index"] = index
            state = "absent" if item["label"].startswith("no_") else "present"
            key = (index, kind)
            evidence.setdefault(key, {}).setdefault(state, []).append(item["confidence"])
    for (index, kind), states in evidence.items():
        # Contradictory observations stay unknown, even with high scores.
        if len(states) == 1:
            people[index][kind] = next(iter(states))
            part = "head_visibility" if kind == "helmet" else "torso_visibility"
            people[index][part] = "model_evidence"
    return people


def finish_event(events, key, start, end, fps):
    if (end - 1 - start) / fps > 5:
        track_id, kind = key
        events.append(
            {
                "track_id": track_id,
                "type": kind,
                "start_s": round(start / fps, 3),
                "end_s": round(end / fps, 3),
                "trigger_s": round((start + int(5 * fps) + 1) / fps, 3),
                "source": "derived_auto_demo",
                "verified": False,
            }
        )


def render(frame, polygon, objects, people, index, fps):
    canvas = frame.copy()
    tint = canvas.copy()
    cv2.fillPoly(tint, [polygon.astype(np.int32)], (0, 180, 240))
    canvas = cv2.addWeighted(tint, 0.15, canvas, 0.85, 0)
    cv2.polylines(canvas, [polygon.astype(np.int32)], True, (0, 220, 255), 3)
    for obj in objects:
        x1, y1, x2, y2 = map(round, obj["bbox"])
        color = (0, 60, 255) if obj["label"].startswith("no_") else (0, 220, 0)
        if obj["label"] == "person":
            color = (255, 180, 0)
            label = f"P{obj['track_id']} H:{obj['helmet']} V:{obj['vest']}"
        else:
            label = f"{obj['label']} {obj['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            label,
            (max(0, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"AUTO / DEMO ZONE | {index / fps:.2f}s",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    return canvas


def annotate(model, path, zone, args, archive):
    cap = cv2.VideoCapture(str(path))
    fps, expected = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    cap.release()
    polygon = np.array(zone, dtype=np.float32) * [width, height]
    tracker = BYTETracker(
        SimpleNamespace(
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.35,
            track_buffer=round(fps),
            match_thresh=0.8,
            fuse_score=True,
        )
    )
    preview = args.previews / f"{path.stem}.mp4"
    out_size = (960, round(height * 960 / width / 2) * 2)
    writer = cv2.VideoWriter(str(preview), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        raise ValueError(f"Cannot write {preview}")
    counts, statuses, active, events, tracks = Counter(), Counter(), {}, [], set()
    audit_indices = set(np.linspace(0, expected - 1, 10, dtype=int).tolist())
    audit_dir = Path(".venv/review") / path.stem
    audit_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    stream = model.predict(
        str(path),
        stream=True,
        imgsz=args.imgsz,
        conf=0.25,
        classes=list(LABELS),
        device=args.device,
        verbose=False,
    )
    with gzip.open(args.output / f"{path.stem}.jsonl.gz", "wt", encoding="utf-8") as output:
        for index, result in enumerate(stream):
            boxes = result.boxes.cpu().numpy()
            persons = boxes[boxes.cls == 0]
            matched = tracker.update(persons)
            ids = {int(row[-1]): int(row[4]) for row in matched}
            objects, person_index = [], 0
            for box, confidence, cls in zip(boxes.xyxy, boxes.conf, boxes.cls, strict=True):
                obj = {
                    "label": LABELS[int(cls)],
                    "bbox": [round(float(x), 2) for x in box],
                    "confidence": round(float(confidence), 3),
                }
                if int(cls) == 0:
                    obj["track_id"] = ids.get(person_index)
                    person_index += 1
                objects.append(obj)
                counts[obj["label"]] += 1
            people = associate(objects)
            conditions = set()
            for person in people:
                track_id = person["track_id"]
                x1, _, x2, y2 = person["bbox"]
                inside = (
                    cv2.pointPolygonTest(polygon.astype(np.float32), ((x1 + x2) / 2, y2), False)
                    >= 0
                )
                person["in_demo_zone"] = inside
                for kind in ("helmet", "vest"):
                    statuses[f"{kind}_{person[kind]}"] += 1
                if track_id is None:
                    continue
                tracks.add(track_id)
                for kind, condition in (
                    ("zone", inside),
                    ("no_helmet", person["helmet"] == "absent"),
                    ("no_vest", person["vest"] == "absent"),
                ):
                    if condition:
                        conditions.add((track_id, kind))
            for key in list(active):
                if key not in conditions:
                    finish_event(events, key, active.pop(key), index, fps)
            for key in conditions:
                active.setdefault(key, index)
            record = {
                "frame": index,
                "time_s": round(index / fps, 4),
                "source": "auto",
                "reviewed": False,
                "objects": objects,
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            lines = []
            for obj in objects:
                x1, y1, x2, y2 = obj["bbox"]
                cls = next(k for k, label in LABELS.items() if label == obj["label"])
                values = [
                    (x1 + x2) / 2 / width,
                    (y1 + y2) / 2 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ]
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in values))
            archive.writestr(
                f"{path.stem}/frame_{index:06d}.txt", "\n".join(lines) + ("\n" if lines else "")
            )
            canvas = render(result.orig_img, polygon, objects, people, index, fps)
            writer.write(cv2.resize(canvas, out_size))
            if index in audit_indices:
                cv2.imwrite(str(audit_dir / f"{index:06d}.jpg"), canvas)
                cv2.imwrite(str(audit_dir / f"{index:06d}_raw.jpg"), result.orig_img)
                (audit_dir / f"{index:06d}.json").write_text(json.dumps(record), encoding="utf-8")
            processed = index + 1
            if processed % 300 == 0:
                print(f"{path.name}: {processed}/{expected}", flush=True)
    writer.release()
    if processed != expected:
        raise ValueError(f"Incomplete annotation: {path.name}: {processed}/{expected}")
    for key, start in active.items():
        finish_event(events, key, start, processed, fps)
    for event in events:
        event["video"] = path.name
    return {
        "video": path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": processed,
        "track_ids": len(tracks),
        **dict(counts),
        **dict(statuses),
    }, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, default=Path("video"))
    parser.add_argument("--files", nargs="+", default=["Camera_0000.mp4", "input.mp4"])
    parser.add_argument("--weights", type=Path, default=Path("models/ppe.pt"))
    parser.add_argument("--zones", type=Path, default=Path("data/demo_zones.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations"))
    parser.add_argument("--previews", type=Path, default=Path("reports/annotations/previews"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    settings.update({"sync": False})
    args.output.mkdir(parents=True, exist_ok=True)
    args.previews.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    if any(
        model.names[k] != name
        for k, name in {
            0: "person",
            1: "hardhat",
            2: "no-hardhat",
            3: "safety vest",
            4: "no-safety vest",
        }.items()
    ):
        raise ValueError("Unexpected checkpoint class mapping")
    zones = json.loads(args.zones.read_text(encoding="utf-8"))["videos"]
    rows, events = [], []
    paths = [args.videos / name for name in args.files]
    if len(paths) != len(set(paths)) or any(not p.is_file() or p.name not in zones for p in paths):
        parser.error("Input names must be unique existing files with demo zones")
    for path in paths:
        with ZipFile(
            args.output / f"{path.stem}_yolo.zip", "w", compression=ZIP_DEFLATED
        ) as archive:
            row, found = annotate(model, path, zones[path.name], args, archive)
            rows.append(row)
            events.extend(found)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    rows = [{column: row.get(column, 0) for column in columns} for row in rows]
    write_csv(args.output / "summary.csv", rows, columns)
    write_csv(
        args.output / "events.csv",
        events,
        ["video", "track_id", "type", "start_s", "end_s", "trigger_s", "source", "verified"],
    )
    manifest = {
        "kind": "automatic_prelabels",
        "verified_ground_truth": False,
        "classes": LABELS,
        "model_repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "weight_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "imgsz": args.imgsz,
        "confidence": 0.25,
        "device": args.device,
        "tracker": "ByteTrack",
        "event_rule": "continuous evidence for more than 5 seconds; missing/unknown resets",
        "bbox": "xyxy in original pixels",
        "frame_index": "zero_based",
        "videos": rows,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done: {sum(r['frames'] for r in rows)} frames, {len(events)} demo events", flush=True)


if __name__ == "__main__":
    main()
