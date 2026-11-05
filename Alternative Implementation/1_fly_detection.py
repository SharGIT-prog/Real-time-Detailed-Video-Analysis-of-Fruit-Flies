import argparse
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 0
FLY_LENGTH_PX = 48.0
MIN_AREA_FRACTION = 0.03
MAX_AREA_FRACTION = 0.15
OPEN_KERNEL = 7
WELL_RADIUS_FRACTION = 0.46
SPLIT_RATIO = 1.7
EXPECTED_FLIES = 2
NMS_RADIUS = 0.35
MATCH_RADIUS = 0.5
INNER_FOLDS = 3
SVM_GRID = {"svm__C": [1, 10, 100, 1000], "svm__gamma": ["scale", 0.01, 0.03, 0.1, 0.3]}
FEATURES = ["area_rel", "perimeter_rel", "width_rel", "height_rel", "aspect_ratio", "circularity",
            "solidity", "extent", "eccentricity", "cx_norm", "cy_norm", "mean_intensity", "std_intensity",
            "contrast"]


def resolve_path(row, data_root):
    path = Path(row["image_path"])
    if path.exists():
        return str(path)
    return str(Path(data_root) / row["image_rel"])


def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def fly_keypoints(row, prefix):
    points = {}
    for part in ("h", "p", "a"):
        x, y = row[f"{prefix}{part}_x"], row[f"{prefix}{part}_y"]
        if np.isfinite(x) and np.isfinite(y):
            points[part] = (x, y)
    return points


def fly_centre(points):
    if "p" in points:
        return points["p"]
    if "h" in points and "a" in points:
        return ((points["h"][0] + points["a"][0]) / 2, (points["h"][1] + points["a"][1]) / 2)
    return next(iter(points.values()))


def load_data(annotations, data_root):
    df = pd.read_csv(annotations)
    df = df[df["has_image"] & df["has_json"]].reset_index(drop=True)
    flies = []
    for _, row in df.iterrows():
        per_image = [fly_keypoints(row, prefix) for prefix in ("m", "f")]
        flies.append([points for points in per_image if points])
    keep = [i for i, f in enumerate(flies) if f]
    df = df.iloc[keep].reset_index(drop=True)
    flies = [flies[i] for i in keep]
    df["image_path"] = [resolve_path(row, data_root) for _, row in df.iterrows()]
    print(f"images: {len(df)}   annotated flies: {sum(len(f) for f in flies)}   folders: {df['folder'].nunique()}")
    return df, flies


def reference_length(df):
    lengths = []
    for prefix in ("m", "f"):
        length = np.hypot(df[f"{prefix}h_x"] - df[f"{prefix}a_x"], df[f"{prefix}h_y"] - df[f"{prefix}a_y"]).dropna()
        lengths.append(float(length.median()))
    return float(np.mean(lengths))


def learn_polarity(df, flies):
    offsets = []
    for (_, row), image_flies in zip(df.iterrows(), flies):
        gray = read_gray(row["image_path"])
        median = float(np.median(gray))
        for points in image_flies:
            x, y = (int(round(v)) for v in fly_centre(points))
            patch = gray[max(y - 6, 0):y + 7, max(x - 6, 0):x + 7]
            if patch.size:
                offsets.append(float(np.median(patch)) - median)
    return "dark" if np.median(offsets) < 0 else "light"


def fill_outside_well(gray):
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    inside = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= (WELL_RADIUS_FRACTION * min(height, width)) ** 2
    filled = gray.copy()
    filled[~inside] = int(np.median(gray[inside]))
    return filled


def outer_contours(mask, min_area, max_area):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    return [c for c, h in zip(contours, hierarchy[0]) if h[3] == -1 and min_area <= cv2.contourArea(c) <= max_area]


def foreground_mask(gray, polarity):
    image = fill_outside_well(gray).astype(np.float32)
    background = cv2.GaussianBlur(image, (0, 0), max(gray.shape) * 0.1)
    image = np.clip(image - background + background.mean(), 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (binary if polarity == "light" else 255 - binary).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def preprocess_images(df, flies, scale, polarity):
    min_area = MIN_AREA_FRACTION * FLY_LENGTH_PX ** 2
    samples = {}
    for (_, row), image_flies in zip(df.iterrows(), flies):
        gray = cv2.resize(read_gray(row["image_path"]), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        max_area = MAX_AREA_FRACTION * gray.shape[0] * gray.shape[1]
        contours = outer_contours(foreground_mask(gray, polarity), min_area, max_area)
        samples[row["sample_id"]] = {
            "gray": gray,
            "contours": contours,
            "keypoints": [np.array(list(p.values())) * scale for p in image_flies],
            "centres": [tuple(np.array(fly_centre(p)) * scale) for p in image_flies],
            "folder": row["folder"],
        }
    return samples


def split_touching_flies(mask, parts, radius):
    distance = cv2.GaussianBlur(cv2.distanceTransform(mask, cv2.DIST_L2, 5), (0, 0), 1.0)
    peaks = peak_local_max(distance, min_distance=max(3, int(0.7 * radius)), num_peaks=parts,
                           labels=(mask > 0).astype(int), exclude_border=False)
    if len(peaks) < 2:
        return [mask]
    markers = np.zeros(mask.shape, np.int32)
    for index, (row, col) in enumerate(peaks, 1):
        markers[row, col] = index
    labels = watershed(-distance, markers, mask=mask > 0)
    pieces = [(labels == i).astype(np.uint8) * 255 for i in range(1, len(peaks) + 1)]
    pieces = [p for p in pieces if p.sum() / 255 >= MIN_AREA_FRACTION * FLY_LENGTH_PX ** 2]
    return pieces if len(pieces) >= 2 else [mask]


def segment_features(contour, mask, gray, offset, area_ref, shape):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    hull = cv2.contourArea(cv2.convexHull(contour))
    moments = cv2.moments(contour)
    cx = moments["m10"] / moments["m00"] + offset[0] if moments["m00"] > 0 else x + w / 2 + offset[0]
    cy = moments["m01"] / moments["m00"] + offset[1] if moments["m00"] > 0 else y + h / 2 + offset[1]
    eccentricity = 0.0
    if len(contour) >= 5:
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), max(min(axis_a, axis_b), 1e-6)
        eccentricity = math.sqrt(max(0.0, 1 - (minor / major) ** 2))
    inside = mask > 0
    values = gray[inside]
    ring = (cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0) & ~inside
    ring_mean = float(gray[ring].mean()) if ring.any() else float(values.mean())
    unit = math.sqrt(area_ref)
    return {
        "area_rel": area / area_ref,
        "perimeter_rel": perimeter / unit,
        "width_rel": w / unit,
        "height_rel": h / unit,
        "aspect_ratio": max(rw, rh) / max(min(rw, rh), 1e-6),
        "circularity": 4 * math.pi * area / (perimeter ** 2 + 1e-9),
        "solidity": area / hull if hull > 0 else 0.0,
        "extent": area / (w * h + 1e-9),
        "eccentricity": eccentricity,
        "cx_norm": cx / shape[1],
        "cy_norm": cy / shape[0],
        "mean_intensity": float(values.mean()),
        "std_intensity": float(values.std()),
        "contrast": float(values.mean()) - ring_mean,
        "area": area,
        "cx": cx,
        "cy": cy,
    }


def label_segment(mask, offset, keypoints):
    flies_inside = 0
    for points in keypoints:
        hits = []
        for x, y in points:
            col, row = int(round(x - offset[0])), int(round(y - offset[1]))
            window = mask[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]
            hits.append(bool(window.size and window.any()))
        flies_inside += int(np.mean(hits) >= 0.5)
    return flies_inside


def make_segments(sample, area_ref, use_watershed):
    gray = sample["gray"]
    segments = []
    radius = math.sqrt(area_ref / math.pi)
    for contour in sample["contours"]:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0 = max(x - 4, 0), max(y - 4, 0)
        x1, y1 = min(x + w + 4, gray.shape[1]), min(y + h + 4, gray.shape[0])
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(mask, [contour - np.array([[x0, y0]])], -1, 255, -1)
        pieces = [mask]
        if use_watershed and cv2.contourArea(contour) > SPLIT_RATIO * area_ref:
            parts = int(min(3, max(2, round(cv2.contourArea(contour) / area_ref))))
            pieces = split_touching_flies(mask, parts, radius)
        for piece in pieces:
            found, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not found:
                continue
            piece_contour = max(found, key=cv2.contourArea)
            if cv2.contourArea(piece_contour) < MIN_AREA_FRACTION * FLY_LENGTH_PX ** 2:
                continue
            segment = segment_features(piece_contour, piece, gray[y0:y1, x0:x1], (x0, y0), area_ref, gray.shape)
            segment["flies_inside"] = label_segment(piece, (x0, y0), sample["keypoints"])
            segment["label"] = int(segment["flies_inside"] > 0)
            segment["contour"] = piece_contour + np.array([[x0, y0]])
            segments.append(segment)
    return segments


def single_fly_area(sample_ids, samples):
    areas = []
    for sample_id in sample_ids:
        segments = make_segments(samples[sample_id], 1.0, False)
        areas.extend(s["area"] for s in segments if s["flies_inside"] == 1)
    return float(np.median(areas)) if areas else 0.3 * FLY_LENGTH_PX ** 2


def segment_table(sample_ids, samples, area_ref):
    rows = []
    for sample_id in sample_ids:
        for segment in make_segments(samples[sample_id], area_ref, True):
            segment["sample_id"] = sample_id
            segment["folder"] = samples[sample_id]["folder"]
            rows.append(segment)
    return pd.DataFrame(rows)


def train_classifier(table):
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=SEED))])
    folds = GroupKFold(n_splits=min(INNER_FOLDS, table["folder"].nunique()))
    search = GridSearchCV(pipeline, SVM_GRID, scoring="f1", cv=folds)
    search.fit(table[FEATURES].to_numpy(float), table["label"].to_numpy(int), groups=table["folder"].to_numpy())
    return search.best_estimator_


def cross_validate(samples):
    ids = list(samples)
    folders = {i: samples[i]["folder"] for i in ids}
    predictions = []
    for held_out in sorted(set(folders.values())):
        train_ids = [i for i in ids if folders[i] != held_out]
        test_ids = [i for i in ids if folders[i] == held_out]
        area_ref = single_fly_area(train_ids, samples)
        train = segment_table(train_ids, samples, area_ref)
        test = segment_table(test_ids, samples, area_ref)
        model = train_classifier(train)
        test["prediction"] = model.predict(test[FEATURES].to_numpy(float))
        test["score"] = model.decision_function(test[FEATURES].to_numpy(float))
        predictions.append(test)
        print(f"  held-out folder {held_out}: single-fly area {area_ref:.0f} px^2, "
              f"{len(train)} training segments, {len(test)} test segments")
    return pd.concat(predictions, ignore_index=True)


def evaluate(segments, samples):
    labels, predictions = segments["label"], segments["prediction"]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    
    records = []
    for sample_id, sample in samples.items():
        detected_count = len(segments[(segments["sample_id"] == sample_id) & (segments["prediction"] == 1)])
        records.append({
            "folder": sample["folder"],
            "detected": detected_count,
            "truth": len(sample["centres"]),
        })
    records = pd.DataFrame(records)
    records["count_error"] = records["detected"] - records["truth"]
    
    metrics = {
        "segments": len(segments),
        "segment_accuracy": accuracy_score(labels, predictions),
        "fly_precision": precision[0],
        "fly_recall": recall[0],
        "fly_f1": f1[0],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "count_mae": records["count_error"].abs().mean(),
        "count_rmse": float(np.sqrt((records["count_error"] ** 2).mean())),
        "count_bias": records["count_error"].mean(),
        "count_exact_match": (records["count_error"] == 0).mean(),
    }
    
    per_folder = records.groupby("folder").apply(lambda g: pd.Series({
        "images": len(g),
        "count_mae": g["count_error"].abs().mean(),
        "exact_match": (g["count_error"] == 0).mean(),
    }))
    
    print("\n==================== RESULTS: CLASSIFICATION & COUNT ====================")
    print("SVM contour classifier (fly vs non-fly)")
    print(f"  segments {metrics['segments']}   accuracy {metrics['segment_accuracy']:.3f}")
    print(f"  fly class: precision {precision[0]:.3f}  recall {recall[0]:.3f}  F1 {f1[0]:.3f}")
    print(f"  confusion matrix: TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    print("Fly count per image")
    print(f"  MAE {metrics['count_mae']:.3f}  RMSE {metrics['count_rmse']:.3f}  "
          f"bias {metrics['count_bias']:+.3f}  exactly correct {metrics['count_exact_match']:.1%}")
    print("Per held-out folder")
    print(per_folder.round(3).to_string())
    return metrics


def plot_confusion(segments, path):
    matrix = confusion_matrix(segments["label"], segments["prediction"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["non-fly", "fly"])
    ax.set_yticks([0, 1], ["non-fly", "fly"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("annotated")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_decision_scores(segments, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(segments["score"].quantile(0.01), segments["score"].quantile(0.99), 50)
    ax.hist(segments.loc[segments["label"] == 0, "score"], bins=bins, alpha=0.6, color="tab:red", label="annotated non-fly")
    ax.hist(segments.loc[segments["label"] == 1, "score"], bins=bins, alpha=0.6, color="tab:green", label="annotated fly")
    ax.axvline(0, color="k", ls="--", label="SVM decision boundary")
    ax.set_xlabel("SVM decision value (> 0 predicted fly)")
    ax.set_ylabel("segments")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_detections(segments, samples, path):
    rng = np.random.RandomState(SEED)
    chosen = rng.choice(list(samples), size=min(6, len(samples)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, sample_id in zip(axes.ravel(), chosen):
        canvas = cv2.cvtColor(samples[sample_id]["gray"], cv2.COLOR_GRAY2RGB)
        for _, segment in segments[segments["sample_id"] == sample_id].iterrows():
            color = (0, 200, 0) if segment["prediction"] == 1 else (220, 0, 0)
            cv2.drawContours(canvas, [segment["contour"]], -1, color, 1)
        ax.imshow(canvas)
        ax.set_title(sample_id, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="processed/annotations.csv")
    parser.add_argument("--data-root", default="images")
    parser.add_argument("--out", default="outputs/1_fly_detection")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1. load data")
    df, flies = load_data(args.annotations, args.data_root)
    length = reference_length(df)
    scale = FLY_LENGTH_PX / length
    polarity = learn_polarity(df, flies)
    print(f"  reference body length {length:.1f} px, scale {scale:.3f}, flies are {polarity}")

    print("2. preprocess images")
    samples = preprocess_images(df, flies, scale, polarity)

    print("3. segment, extract features, classify (leave-one-folder-out)")
    segments = cross_validate(samples)

    print("4. evaluate")
    metrics = evaluate(segments, samples)

    print("5. save results")
    segments.drop(columns=["contour"]).to_csv(out / "segment_predictions.csv", index=False)
    pd.Series(metrics).to_csv(out / "metrics.csv", header=["value"])
    plot_confusion(segments, out / "confusion_matrix.png")
    plot_decision_scores(segments, out / "svm_decision_scores.png")
    plot_detections(segments, samples, out / "detections.png")
    print(f"  results written to {out}")


if __name__ == "__main__":
    main()