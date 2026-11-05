import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".bmp", ".png"}
LABELS = ["mh", "mp", "ma", "fh", "fp", "fa"]
TIMESTAMP_PATTERN = re.compile(r"vlcsnap-(\d{4})-(\d{2})-(\d{2})-(\d{2})h(\d{2})m(\d{2})s(\d{3})", re.I)


def scan_directory(root):
    images = defaultdict(list)
    annotations = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = (path.parent.relative_to(root).as_posix() or ".", path.stem)
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            images[key].append(path)
        elif suffix == ".json":
            annotations[key] = path
    return images, annotations


def parse_filename(stem):
    match = TIMESTAMP_PATTERN.search(stem)
    if match:
        year, month, day, hour, minute, second, millisecond = (int(v) for v in match.groups())
        try:
            return "ts", np.nan, datetime(year, month, day, hour, minute, second, millisecond * 1000)
        except ValueError:
            return "other", np.nan, None
    if stem.isdigit():
        return "frame", float(int(stem)), None
    return "other", np.nan, None


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_annotation(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        raise ValueError("no 'shapes' list")
    points, duplicates = {}, []
    for shape in data["shapes"]:
        label = str(shape.get("label", "")).strip().lower()
        if label not in LABELS:
            continue
        x, y = np.asarray(shape["points"], dtype=float).reshape(-1, 2).mean(axis=0)
        if label in points:
            duplicates.append(label)
        else:
            points[label] = (x, y)
    return points, duplicates


def build_table(root, images, annotations):
    rows, problems = [], []
    keys = sorted(set(images) | set(annotations), key=lambda k: (k[0], natural_key(k[1])))
    for folder, stem in keys:
        image_path = sorted(images[(folder, stem)])[0] if (folder, stem) in images else None
        json_path = annotations.get((folder, stem))
        if image_path is None:
            problems.append(("json_without_image", str(json_path)))
        if json_path is None:
            problems.append(("image_without_json", str(image_path)))
        kind, frame_id, timestamp = parse_filename(stem)
        row = {
            "sample_id": f"{folder}/{stem}",
            "folder": folder,
            "stem": stem,
            "image_path": str(image_path) if image_path else "",
            "image_rel": image_path.relative_to(root).as_posix() if image_path else "",
            "has_image": image_path is not None,
            "has_json": False,
            "name_kind": kind,
            "frame_id": frame_id,
            "timestamp": timestamp,
        }
        for label in LABELS:
            row[f"{label}_x"] = row[f"{label}_y"] = np.nan
        if json_path is not None:
            try:
                points, duplicates = parse_annotation(json_path)
                row["has_json"] = True
                for label, (x, y) in points.items():
                    row[f"{label}_x"], row[f"{label}_y"] = x, y
                for label in LABELS:
                    if label not in points:
                        problems.append(("missing_label", f"{json_path.name}: {label}"))
                for label in duplicates:
                    problems.append(("duplicate_label", f"{json_path.name}: {label}"))
            except Exception as error:
                problems.append(("malformed_json", f"{json_path}: {error!r}"))
        rows.append(row)
    return pd.DataFrame(rows), problems


def add_sequence_order(table):
    table["seq_id"] = table["folder"] + "::" + table["name_kind"]
    table["time_s"] = np.nan
    table["order_in_seq"] = 0
    for _, group in table.groupby("seq_id"):
        kind = group["name_kind"].iloc[0]
        if kind == "ts":
            order = group.sort_values(["timestamp", "stem"]).index
            start = group["timestamp"].min()
            table.loc[order, "time_s"] = [(t - start).total_seconds() for t in table.loc[order, "timestamp"]]
        elif kind == "frame":
            order = group.sort_values("frame_id").index
        else:
            order = group.sort_values("stem", key=lambda s: s.map(natural_key)).index
        table.loc[order, "order_in_seq"] = np.arange(len(order))
    table["timestamp"] = table["timestamp"].astype(str).replace("NaT", "")
    return table.sort_values(["folder", "seq_id", "order_in_seq"]).reset_index(drop=True)


def print_report(table, problems):
    parsed = table[table["has_json"]]
    print(f"images: {int(table['has_image'].sum())}   json parsed: {len(parsed)}   folders: {table['folder'].nunique()}")
    print("images per sequence:")
    for sequence, count in table["seq_id"].value_counts().sort_index().items():
        print(f"  {sequence}: {count}")
    print("annotated labels present:")
    for label in LABELS:
        print(f"  {label}: {int(parsed[f'{label}_x'].notna().sum())}/{len(parsed)}")
    both = parsed[["mh_x", "ma_x", "fh_x", "fa_x"]].notna().all(axis=1).sum()
    print(f"images with head and abdomen of both flies: {both}")
    print("problems:", dict(Counter(kind for kind, _ in problems)) or "none")
    for kind in ("image_without_json", "json_without_image", "malformed_json", "duplicate_label"):
        for _, detail in [p for p in problems if p[0] == kind][:5]:
            print(f"  {kind}: {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="images")
    parser.add_argument("--out", default="processed")
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    if not root.is_dir():
        sys.exit(f"data root not found: {root}")
    images, annotations = scan_directory(root)
    table, problems = build_table(root, images, annotations)
    if table.empty:
        sys.exit("no images or annotations found")
    table = add_sequence_order(table)
    print_report(table, problems)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "annotations.csv", index=False)
    print(f"saved {out / 'annotations.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
