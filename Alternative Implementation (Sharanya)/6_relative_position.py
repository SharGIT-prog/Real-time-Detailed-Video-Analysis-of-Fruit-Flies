import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 0
K_RANGE = range(2, 9)
MIN_CLUSTER_FRACTION = 0.05
VARIANCE_KEPT = 0.95
DISTANCE_RANGE = (0.5, 8.0)
KMEANS_RESTARTS = 50
COLORS = plt.get_cmap("tab10")


def resolve_path(row, data_root):
    path = Path(row["image_path"])
    if path.exists():
        return str(path)
    return str(Path(data_root) / row["image_rel"])


def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def wrap180(angle):
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def heading(df, prefix):
    dx = df[f"{prefix}h_x"] - df[f"{prefix}a_x"]
    dy = df[f"{prefix}h_y"] - df[f"{prefix}a_y"]
    return np.mod(np.degrees(np.arctan2(-dy, dx)), 360.0).to_numpy()


def bearing(reference_xy, other_xy, reference_heading):
    dx = other_xy[:, 0] - reference_xy[:, 0]
    dy = -(other_xy[:, 1] - reference_xy[:, 1])
    radians = np.radians(reference_heading)
    forward = dx * np.cos(radians) + dy * np.sin(radians)
    left = -dx * np.sin(radians) + dy * np.cos(radians)
    return wrap180(np.degrees(np.arctan2(left, forward)))


def sector(alpha):
    magnitude = np.abs(alpha)
    return np.where(magnitude < 45, "front", np.where(magnitude > 135, "behind", np.where(alpha > 0, "left", "right")))


def load_pairs(annotations, data_root):
    df = pd.read_csv(annotations)
    df = df[df["has_image"] & df["has_json"]]
    lengths = []
    for prefix in ("m", "f"):
        length = np.hypot(df[f"{prefix}h_x"] - df[f"{prefix}a_x"], df[f"{prefix}h_y"] - df[f"{prefix}a_y"]).dropna()
        lengths.append(float(length.median()))
    reference = float(np.mean(lengths))
    needed = ["mh_x", "ma_x", "fh_x", "fa_x", "mp_x", "fp_x"]
    df = df[df[needed].notna().all(axis=1)].reset_index(drop=True)
    df["image_path"] = [resolve_path(row, data_root) for _, row in df.iterrows()]
    print(f"image pairs with both headings: {len(df)}   reference body length: {reference:.1f} px")
    return df, reference


def compute_relative_position(df, reference):
    male_xy = df[["mp_x", "mp_y"]].to_numpy(float)
    female_xy = df[["fp_x", "fp_y"]].to_numpy(float)
    male_heading, female_heading = heading(df, "m"), heading(df, "f")
    table = df[["sample_id", "folder", "seq_id", "order_in_seq", "frame_id", "time_s"]].copy()
    table["distance_bl"] = np.hypot(*(female_xy - male_xy).T) / reference
    table["alpha_male"] = bearing(male_xy, female_xy, male_heading)
    table["alpha_female"] = bearing(female_xy, male_xy, female_heading)
    table["delta"] = wrap180(female_heading - male_heading)
    table["sector_female_from_male"] = sector(table["alpha_male"].to_numpy())
    return table


def build_features(table):
    distance = np.log(np.clip(table["distance_bl"].to_numpy(), *DISTANCE_RANGE))
    columns = [distance]
    for name in ("alpha_male", "alpha_female", "delta"):
        radians = np.radians(table[name].to_numpy())
        columns += [np.sin(radians), np.cos(radians)]
    return np.column_stack(columns)


def embed(features):
    scaled = StandardScaler().fit_transform(features)
    return PCA(n_components=VARIANCE_KEPT, random_state=SEED).fit_transform(scaled)


def choose_clusters(embedding):
    candidates = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=KMEANS_RESTARTS, random_state=SEED).fit(embedding)
        candidates.append({
            "k": k,
            "silhouette": silhouette_score(embedding, model.labels_),
            "davies_bouldin": davies_bouldin_score(embedding, model.labels_),
            "calinski_harabasz": calinski_harabasz_score(embedding, model.labels_),
            "smallest_cluster": np.bincount(model.labels_).min() / len(embedding),
            "model": model,
        })
    allowed = [c for c in candidates if c["smallest_cluster"] >= MIN_CLUSTER_FRACTION] or candidates
    best = max(allowed, key=lambda c: c["silhouette"])
    scan = pd.DataFrame([{k: v for k, v in c.items() if k != "model"} for c in candidates])
    return best, scan


def compare_with_predicted_orientation(df, table, predictions_path):
    path = Path(predictions_path)
    if not path.exists():
        print(f"  {path} not found: run 3_orientation.py first to compare with predicted orientations")
        return None, {}
    predicted = pd.read_csv(path)
    headings = {}
    for sex in ("male", "female"):
        series = predicted[predicted["sex"] == sex].set_index("sample_id")["theta_predicted"]
        headings[sex] = table["sample_id"].map(series)
    valid = (headings["male"].notna() & headings["female"].notna()).to_numpy()
    male_xy = df[["mp_x", "mp_y"]].to_numpy(float)
    female_xy = df[["fp_x", "fp_y"]].to_numpy(float)
    male_heading, female_heading = headings["male"].fillna(0).to_numpy(), headings["female"].fillna(0).to_numpy()
    comparison = pd.DataFrame({
        "alpha_male": table["alpha_male"].to_numpy(),
        "alpha_male_predicted": bearing(male_xy, female_xy, male_heading),
        "alpha_female": table["alpha_female"].to_numpy(),
        "alpha_female_predicted": bearing(female_xy, male_xy, female_heading),
        "delta": table["delta"].to_numpy(),
        "delta_predicted": wrap180(female_heading - male_heading),
    })[valid]
    metrics = {}
    print("\nAngles recomputed from predicted orientations (Task 3) versus annotated orientations")
    for name in ("alpha_male", "alpha_female", "delta"):
        error = np.abs(wrap180(comparison[name] - comparison[f"{name}_predicted"]))
        metrics[f"{name}_mean_error_deg"] = error.mean()
        metrics[f"{name}_median_error_deg"] = np.median(error)
        print(f"  {name}: mean error {error.mean():.2f} deg   median {np.median(error):.2f}   "
              f"90th percentile {np.percentile(error, 90):.2f}   (n={len(error)})")
    agreement = (sector(comparison["alpha_male_predicted"].to_numpy()) == sector(comparison["alpha_male"].to_numpy())).mean()
    metrics["sector_agreement"] = agreement
    print(f"  sector (front/left/behind/right) unchanged for {agreement:.1%} of pairs")
    return comparison, metrics


def evaluate(table, embedding, best, scan):
    sizes = np.bincount(table["state"])
    summary = table.groupby("state").agg(
        samples=("state", "size"),
        median_distance_bl=("distance_bl", "median"),
        median_alpha_male=("alpha_male", "median"),
        median_alpha_female=("alpha_female", "median"),
        median_delta=("delta", "median"),
    )
    metrics = {
        "pairs": len(table),
        "k": best["k"],
        "silhouette": best["silhouette"],
        "davies_bouldin": best["davies_bouldin"],
        "calinski_harabasz": best["calinski_harabasz"],
        "smallest_cluster_fraction": sizes.min() / len(table),
        "largest_cluster_fraction": sizes.max() / len(table),
    }
    print("\n==================== RESULTS: RELATIVE POSITION (unsupervised) ====================")
    print("K-Means model selection")
    print(scan.round(3).to_string(index=False))
    print(f"Chosen k = {best['k']}")
    print(f"  silhouette {best['silhouette']:.3f} (higher is better; 0.26-0.5 means weak structure)")
    print(f"  Davies-Bouldin {best['davies_bouldin']:.3f} (lower is better)")
    print(f"  Calinski-Harabasz {best['calinski_harabasz']:.1f} (higher is better)")
    print("States (descriptive statistics only)")
    print(summary.round(2).to_string())
    print("Female relative to male (rule-based sectors):", table["sector_female_from_male"].value_counts().to_dict())
    return metrics


def plot_embedding(embedding, states, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for state in sorted(set(states)):
        mask = states == state
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=16, color=COLORS(state % 10),
                   label=f"state {state} (n={int(mask.sum())})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_predicted_orientation(comparison, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, name in zip(axes, ("alpha_male", "alpha_female", "delta")):
        ax.scatter(comparison[name], comparison[f"{name}_predicted"], s=10, alpha=0.6)
        ax.plot([-180, 180], [-180, 180], "k--", lw=0.8)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel(f"{name} from annotated orientation (deg)")
        ax.set_ylabel(f"{name} from predicted orientation (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_examples(df, table, embedding, model, reference, path):
    states = sorted(set(table["state"]))
    rows = int(np.ceil(len(states) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, state in zip(axes.ravel(), states):
        members = np.where(table["state"].to_numpy() == state)[0]
        index = members[np.argmin(np.linalg.norm(embedding[members] - model.cluster_centers_[state], axis=1))]
        row = df.iloc[index]
        canvas = cv2.cvtColor(read_gray(row["image_path"]), cv2.COLOR_GRAY2RGB)
        for prefix, color in (("m", (40, 120, 255)), ("f", (255, 60, 60))):
            head = (int(row[f"{prefix}h_x"]), int(row[f"{prefix}h_y"]))
            abdomen = (int(row[f"{prefix}a_x"]), int(row[f"{prefix}a_y"]))
            cv2.line(canvas, abdomen, head, color, 5)
            cv2.circle(canvas, head, 14, color, -1)
        cx = (row["mp_x"] + row["fp_x"]) / 2
        cy = (row["mp_y"] + row["fp_y"]) / 2
        half = max(2.2 * reference, 0.8 * abs(row["mp_x"] - row["fp_x"]), 0.8 * abs(row["mp_y"] - row["fp_y"]))
        x0, x1 = int(max(cx - half, 0)), int(min(cx + half, canvas.shape[1]))
        y0, y1 = int(max(cy - half, 0)), int(min(cy + half, canvas.shape[0]))
        ax.imshow(canvas[y0:y1, x0:x1])
        ax.set_title(f"state {state}: {table['distance_bl'].iloc[index]:.1f} BL, "
                     f"female at {table['alpha_male'].iloc[index]:.0f} deg", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="processed/annotations.csv")
    parser.add_argument("--data-root", default="images")
    parser.add_argument("--orientation", default="outputs/3_orientation/orientation_predictions.csv")
    parser.add_argument("--out", default="outputs/6_relative_position")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1. load data")
    df, reference = load_pairs(args.annotations, args.data_root)

    print("2. compute relative position (bearing of each fly in the other's body frame)")
    table = compute_relative_position(df, reference)

    print("3. build features and embed (StandardScaler, PCA)")
    embedding = embed(build_features(table))
    print(f"   {embedding.shape[1]} principal components keep {VARIANCE_KEPT:.0%} of the variance")

    print("4. cluster with K-Means (k chosen by silhouette)")
    best, scan = choose_clusters(embedding)
    table["state"] = best["model"].labels_

    print("5. evaluate")
    metrics = evaluate(table, embedding, best, scan)
    comparison, comparison_metrics = compare_with_predicted_orientation(df, table, args.orientation)
    metrics.update(comparison_metrics)

    print("6. save results")
    table.to_csv(out / "relative_position_states.csv", index=False)
    pd.Series(metrics).to_csv(out / "metrics.csv", header=["value"])
    plot_embedding(embedding, best["model"].labels_, out / "states_embedding.png")
    plot_examples(df, table, embedding, best["model"], reference, out / "states_on_flies.png")
    if comparison is not None:
        plot_predicted_orientation(comparison, out / "angles_from_predicted_vs_annotated_orientation.png")
    print(f"   results written to {out}")


if __name__ == "__main__":
    main()
