import argparse
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
CROP_SCALE = 2.4
IMAGE_SIZE = 64
AXIS_SIZE = 96
AUGMENT_OFFSETS = [180.0, 12.0, -12.0, 192.0, 168.0]
PCA_COMPONENTS = 100
KERNEL_GRID = {"model__alpha": [0.03, 0.1, 1.0], "model__gamma": [1e-4, 5e-4, 2e-3]}
INNER_FOLDS = 3


def resolve_path(row, data_root):
    path = Path(row["image_path"])
    if path.exists():
        return str(path)
    return str(Path(data_root) / row["image_rel"])


def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def wrap180(angle):
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def angular_error(a, b):
    return np.abs(wrap180(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def direction_angle(dx, dy):
    return float(np.mod(np.degrees(np.arctan2(-dy, dx)), 360.0))


def load_flies(annotations, data_root):
    df = pd.read_csv(annotations)
    df = df[df["has_image"] & df["has_json"]].reset_index(drop=True)
    df["image_path"] = [resolve_path(row, data_root) for _, row in df.iterrows()]
    flies = []
    for index, row in df.iterrows():
        for sex, prefix in (("male", "m"), ("female", "f")):
            head = (row[f"{prefix}h_x"], row[f"{prefix}h_y"])
            abdomen = (row[f"{prefix}a_x"], row[f"{prefix}a_y"])
            if not np.all(np.isfinite(head + abdomen)):
                continue
            flies.append({
                "image": index,
                "sample_id": row["sample_id"],
                "folder": row["folder"],
                "sex": sex,
                "head": head,
                "abdomen": abdomen,
                "centre": ((head[0] + abdomen[0]) / 2, (head[1] + abdomen[1]) / 2),
                "theta": direction_angle(head[0] - abdomen[0], head[1] - abdomen[1]),
            })
    for fly in flies:
        others = [f for f in flies if f["image"] == fly["image"] and f["sex"] != fly["sex"]]
        fly["other"] = (others[0]["head"], others[0]["abdomen"]) if others else None
    lengths = [math.hypot(f["head"][0] - f["abdomen"][0], f["head"][1] - f["abdomen"][1]) for f in flies]
    medians = [np.median([l for l, f in zip(lengths, flies) if f["sex"] == s]) for s in ("male", "female")]
    reference = float(np.mean(medians))
    print(f"flies: {len(flies)}   reference body length: {reference:.1f} px   folders: {df['folder'].nunique()}")
    return df, flies, reference


def segment_distance(xx, yy, p, q):
    direction = np.asarray(q, float) - np.asarray(p, float)
    length_sq = max(float(direction @ direction), 1e-9)
    t = np.clip(((xx - p[0]) * direction[0] + (yy - p[1]) * direction[1]) / length_sq, 0, 1)
    return np.hypot(xx - (p[0] + t * direction[0]), yy - (p[1] + t * direction[1]))


def extract_crop(image, centre, rotation, side, size, own, other, background, normalise):
    matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), float(rotation), size / side)
    matrix[0, 2] += size / 2 - centre[0]
    matrix[1, 2] += size / 2 - centre[1]
    oversample = 2 if size / side < 0.6 else 1
    crop = cv2.warpAffine(image, matrix * oversample, (size * oversample, size * oversample),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=background).astype(np.float32)
    if oversample > 1:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    if other is not None:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        own_points = [matrix @ np.array([p[0], p[1], 1.0]) for p in own]
        other_points = [matrix @ np.array([p[0], p[1], 1.0]) for p in other]
        closer_to_other = segment_distance(xx, yy, *other_points) < segment_distance(xx, yy, *own_points)
        crop[closer_to_other] = background
    if normalise:
        crop = (crop - crop.mean()) / (crop.std() + 1e-6)
    return crop, matrix


def silhouette_axis(crop, background):
    difference = np.abs(crop - background)
    scaled = np.clip(difference * (255.0 / max(float(difference.max()), 1.0)), 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(cv2.GaussianBlur(scaled, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return 90.0
    centre = crop.shape[0] // 2
    label = labels[centre, centre] if labels[centre, centre] > 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    body = (labels == label).astype(np.uint8)
    if body.sum() < 20:
        return 90.0
    moments = cv2.moments(body, binaryImage=True)
    angle = 0.5 * math.atan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
    return float(np.mod(-math.degrees(angle), 180.0))


def build_dataset(df, flies, reference):
    side = CROP_SCALE * reference
    features, targets, rotations, owners, augmented = [], [], [], [], []
    cache = {"image": None, "pixels": None, "background": None}
    for index, fly in enumerate(flies):
        if cache["image"] != fly["image"]:
            pixels = read_gray(df.loc[fly["image"], "image_path"])
            cache = {"image": fly["image"], "pixels": pixels, "background": float(np.median(pixels))}
        pixels, background = cache["pixels"], cache["background"]
        own = (fly["head"], fly["abdomen"])
        wide, _ = extract_crop(pixels, fly["centre"], 0.0, side, AXIS_SIZE, own, fly["other"], background, False)
        base_rotation = 90.0 - silhouette_axis(wide, background)
        for offset in [0.0] + AUGMENT_OFFSETS:
            rotation = base_rotation + offset
            crop, matrix = extract_crop(pixels, fly["centre"], rotation, side, IMAGE_SIZE, own, fly["other"],
                                        background, True)
            vector = matrix[:, :2] @ (np.array(fly["head"]) - np.array(fly["abdomen"]))
            features.append(hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                block_norm="L2-Hys", feature_vector=True))
            targets.append(direction_angle(vector[0], vector[1]))
            rotations.append(rotation)
            owners.append(index)
            augmented.append(offset != 0.0)
    return (np.array(features), np.array(targets), np.array(rotations), np.array(owners),
            np.array(augmented, dtype=bool))


def angle_to_vector(angle):
    radians = np.radians(angle)
    return np.column_stack([np.sin(radians), np.cos(radians)])


def vector_to_angle(vector):
    unit = vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
    return np.mod(np.degrees(np.arctan2(unit[:, 0], unit[:, 1])), 360.0)


def negative_angular_error(y_true, y_pred):
    return -float(np.mean(angular_error(vector_to_angle(y_true), vector_to_angle(y_pred))))


def validation_splits(folders, augmented):
    splitter = GroupKFold(n_splits=min(INNER_FOLDS, len(set(folders))))
    splits = []
    for train, validate in splitter.split(np.zeros(len(folders)), groups=folders):
        splits.append((train, validate[~augmented[validate]]))
    return splits


def train_model(features, targets, folders, augmented):
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("pca", PCA(n_components=PCA_COMPONENTS, random_state=SEED)),
                         ("model", KernelRidge(kernel="rbf"))])
    search = GridSearchCV(pipeline, KERNEL_GRID, scoring=make_scorer(negative_angular_error),
                          cv=validation_splits(folders, augmented))
    search.fit(features, angle_to_vector(targets))
    return search.best_estimator_


def cross_validate(flies, features, targets, rotations, owners, augmented):
    folders = np.array([f["folder"] for f in flies])
    predictions = np.zeros(len(flies))
    for held_out in sorted(set(folders)):
        train = folders[owners] != held_out
        test = (folders[owners] == held_out) & ~augmented
        model = train_model(features[train], targets[train], folders[owners][train], augmented[train])
        predicted = vector_to_angle(model.predict(features[test]))
        predictions[owners[test]] = np.mod(predicted - rotations[test], 360.0)
        error = angular_error(predictions[owners[test]], [flies[i]["theta"] for i in owners[test]])
        print(f"  held-out folder {held_out}: {int(test.sum())} flies, mean error {error.mean():.2f} deg")
    return predictions


def evaluate(flies, predictions):
    truth = np.array([f["theta"] for f in flies])
    errors = angular_error(predictions, truth)
    sexes = np.array([f["sex"] for f in flies])
    folders = np.array([f["folder"] for f in flies])
    metrics = {
        "flies": len(errors),
        "mean_angular_error": errors.mean(),
        "median_angular_error": np.median(errors),
        "std_angular_error": errors.std(),
        "p75": np.percentile(errors, 75),
        "p90": np.percentile(errors, 90),
        "p95": np.percentile(errors, 95),
        "within_10_deg": (errors <= 10).mean(),
        "within_20_deg": (errors <= 20).mean(),
        "within_45_deg": (errors <= 45).mean(),
        "head_abdomen_flips": (errors > 150).mean(),
    }
    print("\n==================== RESULTS: ORIENTATION (leave-one-folder-out) ====================")
    print("Angular error in degrees (wrap-aware)")
    print(f"  mean {metrics['mean_angular_error']:.2f}   median {metrics['median_angular_error']:.2f}   "
          f"std {metrics['std_angular_error']:.2f}")
    print(f"  75th percentile {metrics['p75']:.2f}   90th {metrics['p90']:.2f}   95th {metrics['p95']:.2f}")
    print(f"  within 10 deg {metrics['within_10_deg']:.1%}   within 20 deg {metrics['within_20_deg']:.1%}   "
          f"within 45 deg {metrics['within_45_deg']:.1%}")
    print(f"  head/abdomen flips (error > 150 deg) {metrics['head_abdomen_flips']:.1%}")
    print("By sex (mean / median error)")
    for sex in ("male", "female"):
        mask = sexes == sex
        print(f"  {sex}: n {int(mask.sum())}   {errors[mask].mean():.2f} / {np.median(errors[mask]):.2f}")
    print("By held-out folder (mean / median error)")
    for folder in sorted(set(folders)):
        mask = folders == folder
        print(f"  {folder}: n {int(mask.sum())}   {errors[mask].mean():.2f} / {np.median(errors[mask]):.2f}")
    return errors, metrics


def plot_prediction_scatter(truth, predictions, errors, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    points = ax.scatter(truth, predictions, c=errors, s=14, cmap="viridis_r", vmin=0, vmax=90)
    ax.plot([0, 360], [0, 360], "k-", lw=1)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel("annotated orientation (deg)")
    ax.set_ylabel("predicted orientation (deg)")
    fig.colorbar(points, ax=ax, label="angular error (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_arrows(df, flies, predictions, reference, path):
    rng = np.random.RandomState(SEED)
    chosen = rng.choice(len(flies), size=min(8, len(flies)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, index in zip(axes.ravel(), chosen):
        fly = flies[index]
        pixels = read_gray(df.loc[fly["image"], "image_path"])
        crop, _ = extract_crop(pixels, fly["centre"], 0.0, CROP_SCALE * reference, 128, None, None,
                               float(np.median(pixels)), False)
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        for angle, color in ((fly["theta"], "lime"), (predictions[index], "red")):
            ax.arrow(64, 64, 45 * math.cos(math.radians(angle)), -45 * math.sin(math.radians(angle)),
                     color=color, width=1.2, head_width=7)
        ax.set_title(f"{fly['sex']}  error {angular_error(predictions[index], fly['theta']):.0f} deg", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="processed/annotations.csv")
    parser.add_argument("--data-root", default="images")
    parser.add_argument("--out", default="outputs/3_orientation")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1. load data")
    df, flies, reference = load_flies(args.annotations, args.data_root)

    print("2. preprocess crops and extract HOG features")
    features, targets, rotations, owners, augmented = build_dataset(df, flies, reference)
    print(f"   feature matrix {features.shape} ({int((~augmented).sum())} original, {int(augmented.sum())} augmented)")

    print("3. train and predict (leave-one-folder-out)")
    predictions = cross_validate(flies, features, targets, rotations, owners, augmented)

    print("4. evaluate")
    errors, metrics = evaluate(flies, predictions)

    print("5. save results")
    results = pd.DataFrame({
        "sample_id": [f["sample_id"] for f in flies],
        "sex": [f["sex"] for f in flies],
        "theta_true": [f["theta"] for f in flies],
        "theta_predicted": predictions,
        "angular_error": errors,
    })
    results.to_csv(out / "orientation_predictions.csv", index=False)
    pd.Series(metrics).to_csv(out / "metrics.csv", header=["value"])
    plot_prediction_scatter(results["theta_true"], predictions, errors, out / "predicted_vs_true.png")
    plot_arrows(df, flies, predictions, reference, out / "orientation_on_flies.png")
    print(f"   results written to {out}")


if __name__ == "__main__":
    main()
