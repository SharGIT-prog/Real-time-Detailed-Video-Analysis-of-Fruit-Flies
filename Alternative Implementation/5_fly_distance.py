import argparse
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
FLY_LENGTH_PX = 48.0
MIN_AREA_FRACTION = 0.03
MAX_AREA_FRACTION = 0.15
OPEN_KERNEL = 7
WELL_RADIUS_FRACTION = 0.46
MAX_FLIES = 2
NMS_RADIUS = 0.35
INNER_FOLDS = 3
LOGISTIC_GRID = {"logistic__C": [0.1, 1, 10, 100]}
COMPONENT_FEATURES = ["area_rel", "perimeter_rel", "aspect_ratio", "circularity", "solidity", "extent", "contrast"]
GEOMETRY_FEATURES = ["separation", "hull_diameter", "spread_major", "box_diagonal"]


def resolve_path(row, data_root):
    path = Path(row["image_path"])
    if path.exists():
        return str(path)
    return str(Path(data_root) / row["image_rel"])


def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def load_data(annotations, data_root):
    df = pd.read_csv(annotations)
    df = df[df["has_image"] & df["has_json"]]
    lengths = []
    for prefix in ("m", "f"):
        length = np.hypot(df[f"{prefix}h_x"] - df[f"{prefix}a_x"], df[f"{prefix}h_y"] - df[f"{prefix}a_y"]).dropna()
        lengths.append(float(length.median()))
    reference = float(np.mean(lengths))
    df = df[df[["mp_x", "mp_y", "fp_x", "fp_y"]].notna().all(axis=1)].reset_index(drop=True)
    df["image_path"] = [resolve_path(row, data_root) for _, row in df.iterrows()]
    print(f"image pairs: {len(df)}   reference body length: {reference:.1f} px   folders: {df['folder'].nunique()}")
    return df, reference


def true_distance(df, reference):
    return (np.hypot(df["fp_x"] - df["mp_x"], df["fp_y"] - df["mp_y"]) / reference).to_numpy()


def learn_polarity(df):
    offsets = []
    for _, row in df.iterrows():
        gray = read_gray(row["image_path"])
        median = float(np.median(gray))
        for prefix in ("m", "f"):
            x, y = int(round(row[f"{prefix}p_x"])), int(round(row[f"{prefix}p_y"]))
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


def foreground_mask(gray, polarity):
    image = fill_outside_well(gray).astype(np.float32)
    background = cv2.GaussianBlur(image, (0, 0), max(gray.shape) * 0.1)
    image = np.clip(image - background + background.mean(), 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (binary if polarity == "light" else 255 - binary).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def outer_contours(mask, min_area, max_area):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    return [c for c, h in zip(contours, hierarchy[0]) if h[3] == -1 and min_area <= cv2.contourArea(c) <= max_area]


def preprocess_images(df, scale, polarity):
    min_area = MIN_AREA_FRACTION * FLY_LENGTH_PX ** 2
    samples = []
    for _, row in df.iterrows():
        gray = cv2.resize(read_gray(row["image_path"]), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        max_area = MAX_AREA_FRACTION * gray.shape[0] * gray.shape[1]
        samples.append({
            "gray": gray,
            "contours": outer_contours(foreground_mask(gray, polarity), min_area, max_area),
            "body_points": [(row["mp_x"] * scale, row["mp_y"] * scale), (row["fp_x"] * scale, row["fp_y"] * scale)],
        })
    return samples


def contains_point(contour, point):
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True) >= -2


def component_features(contour, gray, area_ref):
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0 = max(x - 4, 0), max(y - 4, 0)
    x1, y1 = min(x + w + 4, gray.shape[1]), min(y + h + 4, gray.shape[0])
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.drawContours(mask, [contour - np.array([[x0, y0]])], -1, 255, -1)
    patch = gray[y0:y1, x0:x1]
    inside = mask > 0
    ring = (cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0) & ~inside
    contrast = float(patch[inside].mean()) - (float(patch[ring].mean()) if ring.any() else float(patch[inside].mean()))
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.contourArea(cv2.convexHull(contour))
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    return [area / area_ref, perimeter / math.sqrt(area_ref), max(rw, rh) / max(min(rw, rh), 1e-6),
            4 * math.pi * area / (perimeter ** 2 + 1e-9), area / hull if hull > 0 else 0.0,
            area / (w * h + 1e-9), contrast]


def single_fly_area(indices, samples):
    areas = []
    for index in indices:
        sample = samples[index]
        for contour in sample["contours"]:
            if sum(contains_point(contour, p) for p in sample["body_points"]) == 1:
                areas.append(cv2.contourArea(contour))
    return float(np.median(areas)) if areas else 0.3 * FLY_LENGTH_PX ** 2


def component_table(indices, samples, area_ref, folders):
    rows = []
    for index in indices:
        sample = samples[index]
        for contour in sample["contours"]:
            moments = cv2.moments(contour)
            row = dict(zip(COMPONENT_FEATURES, component_features(contour, sample["gray"], area_ref)))
            row.update({
                "image": index,
                "folder": folders[index],
                "label": int(any(contains_point(contour, p) for p in sample["body_points"])),
                "cx": moments["m10"] / max(moments["m00"], 1e-9),
                "cy": moments["m01"] / max(moments["m00"], 1e-9),
                "contour": contour,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def train_component_classifier(table):
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("logistic", LogisticRegression(class_weight="balanced", max_iter=2000))])
    folds = GroupKFold(n_splits=min(INNER_FOLDS, table["folder"].nunique()))
    search = GridSearchCV(pipeline, LOGISTIC_GRID, scoring="f1", cv=folds)
    search.fit(table[COMPONENT_FEATURES].to_numpy(float), table["label"].to_numpy(int),
               groups=table["folder"].to_numpy())
    return search.best_estimator_


def select_detections(components):
    ranked = components.sort_values("score", ascending=False)
    candidates = ranked[ranked["score"] > 0]
    if candidates.empty:
        candidates = ranked.head(1)
    kept = []
    for _, component in candidates.iterrows():
        if all(math.hypot(component["cx"] - k["cx"], component["cy"] - k["cy"]) >= NMS_RADIUS * FLY_LENGTH_PX
               for k in kept):
            kept.append(component)
        if len(kept) == MAX_FLIES:
            break
    return kept


def distance_features(detections):
    if not detections:
        return np.zeros(len(GEOMETRY_FEATURES) + 1)
    outline = np.vstack([d["contour"][:, 0, :] for d in detections]).astype(np.float32)
    hull = cv2.convexHull(outline)[:, 0, :]
    canvas_x, canvas_y = outline[:, 0], outline[:, 1]
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(outline.T.astype(np.float64))))[::-1]
    separation = math.hypot(detections[0]["cx"] - detections[1]["cx"], detections[0]["cy"] - detections[1]["cy"]) \
        if len(detections) == 2 else 0.0
    box = math.hypot(canvas_x.max() - canvas_x.min(), canvas_y.max() - canvas_y.min())
    unit = FLY_LENGTH_PX
    return np.array([separation / unit, (pdist(hull).max() if len(hull) > 1 else 0.0) / unit,
                     math.sqrt(max(eigenvalues[0], 0.0)) / unit, box / unit, float(len(detections))])


def distance_matrix(indices, table):
    rows = []
    for index in indices:
        components = table[table["image"] == index]
        rows.append(distance_features(select_detections(components)) if len(components) else
                    np.zeros(len(GEOMETRY_FEATURES) + 1))
    rows = np.array(rows)
    geometry, separate = rows[:, :-1], (rows[:, -1] == 2)[:, None]
    return np.hstack([geometry * separate, geometry * ~separate, separate])


def cross_validate(samples, distances, folders):
    predictions = np.zeros(len(samples))
    tables = []
    for held_out in sorted(set(folders)):
        train = [i for i in range(len(samples)) if folders[i] != held_out]
        test = [i for i in range(len(samples)) if folders[i] == held_out]
        area_ref = single_fly_area(train, samples)
        train_table = component_table(train, samples, area_ref, folders)
        test_table = component_table(test, samples, area_ref, folders)
        classifier = train_component_classifier(train_table)
        for table in (train_table, test_table):
            table["score"] = classifier.decision_function(table[COMPONENT_FEATURES].to_numpy(float))
            table["prediction"] = classifier.predict(table[COMPONENT_FEATURES].to_numpy(float))
        regressor = LinearRegression().fit(distance_matrix(train, train_table), distances[train])
        predictions[test] = np.maximum(regressor.predict(distance_matrix(test, test_table)), 0.0)
        tables.append(test_table)
        print(f"  held-out folder {held_out}: {len(test)} images, "
              f"MAE {mean_absolute_error(distances[test], predictions[test]):.3f} BL")
    return predictions, pd.concat(tables, ignore_index=True)


def alternative_distances(df):
    primary = np.hypot(df["fp_x"] - df["mp_x"], df["fp_y"] - df["mp_y"])
    midpoint_m = np.column_stack([(df["mh_x"] + df["ma_x"]) / 2, (df["mh_y"] + df["ma_y"]) / 2])
    midpoint_f = np.column_stack([(df["fh_x"] + df["fa_x"]) / 2, (df["fh_y"] + df["fa_y"]) / 2])

    def mean_point(prefix):
        stack = np.stack([df[[f"{prefix}{p}_x", f"{prefix}{p}_y"]].to_numpy(float) for p in ("h", "p", "a")])
        return np.nanmean(stack, axis=0)

    keypoints_m = np.stack([df[[f"m{p}_x", f"m{p}_y"]].to_numpy(float) for p in ("h", "p", "a")], axis=1)
    keypoints_f = np.stack([df[[f"f{p}_x", f"f{p}_y"]].to_numpy(float) for p in ("h", "p", "a")], axis=1)
    pairwise = np.linalg.norm(keypoints_m[:, :, None, :] - keypoints_f[:, None, :, :], axis=3).reshape(len(df), -1)
    return pd.DataFrame({
        "pivot_points": primary.to_numpy(),
        "head_abdomen_midpoints": np.linalg.norm(midpoint_m - midpoint_f, axis=1),
        "mean_of_head_pivot_abdomen": np.linalg.norm(mean_point("m") - mean_point("f"), axis=1),
        "head_to_head": np.hypot(df["mh_x"] - df["fh_x"], df["mh_y"] - df["fh_y"]).to_numpy(),
        "closest_keypoints": np.nanmin(np.where(np.isfinite(pairwise), pairwise, np.inf), axis=1),
    }).replace(np.inf, np.nan)


def evaluate(distances, predictions, components, folders, alternatives):
    errors = predictions - distances
    precision, recall, f1, _ = precision_recall_fscore_support(
        components["label"], components["prediction"], labels=[1], zero_division=0)
    metrics = {
        "fly_component_accuracy": np.mean(components["label"] == components["prediction"]),
        "fly_component_precision": precision[0],
        "fly_component_recall": recall[0],
        "fly_component_f1": f1[0],
        "distance_mae_bl": mean_absolute_error(distances, predictions),
        "distance_median_ae_bl": float(np.median(np.abs(errors))),
        "distance_rmse_bl": math.sqrt(mean_squared_error(distances, predictions)),
        "distance_r2": r2_score(distances, predictions),
        "distance_pearson_r": float(np.corrcoef(distances, predictions)[0, 1]),
    }
    print("\n==================== RESULTS: INTER-FLY DISTANCE (leave-one-folder-out) ====================")
    print("Logistic regression (fly blob vs non-fly blob)")
    print(f"  precision {precision[0]:.3f}  recall {recall[0]:.3f}  F1 {f1[0]:.3f}")
    print("Linear regression (distance in body lengths, BL)")
    print(f"  MAE {metrics['distance_mae_bl']:.3f}  median AE {metrics['distance_median_ae_bl']:.3f}  "
          f"RMSE {metrics['distance_rmse_bl']:.3f}  R2 {metrics['distance_r2']:.3f}  "
          f"Pearson r {metrics['distance_pearson_r']:.3f}")
    print("Per held-out folder (MAE in BL)")
    for folder in sorted(set(folders)):
        mask = folders == folder
        print(f"  {folder}: images {int(mask.sum())}  MAE {mean_absolute_error(distances[mask], predictions[mask]):.3f}")
    print("Robustness of the annotated distance definition (Pearson r against the pivot-point distance)")
    for name in alternatives.columns[1:]:
        valid = alternatives[name].notna()
        r = np.corrcoef(alternatives.loc[valid, "pivot_points"], alternatives.loc[valid, name])[0, 1]
        difference = np.mean(np.abs(alternatives.loc[valid, "pivot_points"] - alternatives.loc[valid, name]))
        metrics[f"definition_r_{name}"] = r
        print(f"  {name}: r {r:.3f}  mean absolute difference {difference:.1f} px  (n={int(valid.sum())})")
    return metrics


def plot_scatter(distances, predictions, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(distances, predictions, s=14, alpha=0.7)
    limit = max(distances.max(), predictions.max()) * 1.05
    ax.plot([0, limit], [0, limit], "k--", lw=1)
    ax.set_xlabel("annotated distance (BL)")
    ax.set_ylabel("predicted distance (BL)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_overlay(df, distances, predictions, path):
    rng = np.random.RandomState(SEED)
    chosen = rng.choice(len(df), size=min(6, len(df)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, index in zip(axes.ravel(), chosen):
        row = df.iloc[index]
        canvas = cv2.cvtColor(read_gray(row["image_path"]), cv2.COLOR_GRAY2RGB)
        male, female = (int(row["mp_x"]), int(row["mp_y"])), (int(row["fp_x"]), int(row["fp_y"]))
        cv2.line(canvas, male, female, (255, 215, 0), 4)
        cv2.circle(canvas, male, 14, (40, 120, 255), -1)
        cv2.circle(canvas, female, 14, (255, 60, 60), -1)
        ax.imshow(canvas)
        ax.set_title(f"annotated {distances[index]:.2f} BL   predicted {predictions[index]:.2f} BL", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_timeseries(results, path):
    sequences = [(name, group) for name, group in results.groupby("seq_id") if len(group) >= 8]
    sequences = sorted(sequences, key=lambda item: -len(item[1]))[:4]
    fig, axes = plt.subplots(len(sequences), 1, figsize=(10, 2.6 * len(sequences)), squeeze=False)
    for ax, (name, group) in zip(axes[:, 0], sequences):
        group = group.sort_values("order_in_seq")
        if group["time_s"].notna().all():
            x, label = group["time_s"], "time since first frame (s)"
        elif group["frame_id"].notna().all():
            x, label = group["frame_id"], "frame number"
        else:
            x, label = np.arange(len(group)), "sample order"
        ax.plot(x, group["distance_true_bl"], "o-", ms=3, label="annotated")
        ax.plot(x, group["distance_predicted_bl"], "s--", ms=3, label="predicted")
        ax.set_ylabel("distance (BL)")
        ax.set_xlabel(label)
        ax.set_title(name, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_position(df, distances, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    points = ax.scatter(df["mp_x"], df["mp_y"], c=distances, s=16, cmap="viridis")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("male x (px)")
    ax.set_ylabel("male y (px)")
    fig.colorbar(points, ax=ax, label="distance to female (BL)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_definition_robustness(alternatives, path):
    names = list(alternatives.columns[1:])
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.2))
    for ax, name in zip(axes, names):
        valid = alternatives[name].notna()
        x, y = alternatives.loc[valid, "pivot_points"], alternatives.loc[valid, name]
        ax.scatter(x, y, s=8, alpha=0.5)
        limit = max(x.max(), y.max())
        ax.plot([0, limit], [0, limit], "k--", lw=0.8)
        ax.set_xlabel("pivot-point distance (px)")
        ax.set_ylabel(f"{name} (px)")
        ax.set_title(f"Pearson r = {np.corrcoef(x, y)[0, 1]:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="processed/annotations.csv")
    parser.add_argument("--data-root", default="images")
    parser.add_argument("--out", default="outputs/5_fly_distance")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1. load data and annotated distances")
    df, reference = load_data(args.annotations, args.data_root)
    distances = true_distance(df, reference)
    folders = df["folder"].to_numpy()

    print("2. preprocess images (well mask, foreground blobs)")
    samples = preprocess_images(df, FLY_LENGTH_PX / reference, learn_polarity(df))

    print("3-4. detect fly blobs (logistic regression), extract geometry, fit linear regression (leave-one-folder-out)")
    predictions, components = cross_validate(samples, distances, folders)

    print("5. evaluate")
    metrics = evaluate(distances, predictions, components, folders, alternative_distances(df))

    print("6. save results")
    results = df[["sample_id", "folder", "seq_id", "order_in_seq", "frame_id", "time_s"]].copy()
    results["distance_true_bl"] = distances
    results["distance_predicted_bl"] = predictions
    results.to_csv(out / "distance_predictions.csv", index=False)
    pd.Series(metrics).to_csv(out / "metrics.csv", header=["value"])
    plot_scatter(distances, predictions, out / "predicted_vs_true.png")
    plot_overlay(df, distances, predictions, out / "distance_on_flies.png")
    plot_timeseries(results, out / "distance_timeseries.png")
    plot_position(df, distances, out / "male_position_vs_distance.png")
    plot_definition_robustness(alternative_distances(df), out / "definition_robustness.png")
    print(f"   results written to {out}")


if __name__ == "__main__":
    main()
