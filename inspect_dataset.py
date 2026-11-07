"""Inspect binary mask data and propose patient-safe splits, without training."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import Config
from dataset import make_splits, read_manifest, read_pair


def inspect(manifest: Path, output: Path, provenance: Path, seed: int = 42) -> dict:
    """Scan every mask; inspect three seeded random samples and patient grouping evidence."""
    info = json.loads(provenance.read_text())
    frame = read_manifest(str(manifest))
    if len(frame) < 3:
        raise ValueError("Need at least three image/mask pairs")
    source = info.get("patient_grouping", {})
    if not source.get("basis") or not source.get("source_url"):
        raise ValueError("Provenance must state patient-grouping basis and source URL")
    output.mkdir(parents=True, exist_ok=True)
    stats, labels = [], set()
    # read_manifest additionally rejects duplicate IDs, identical bytes/pixels and missing files.
    for row in frame.itertuples():
        image, mask = read_pair(row.image_path, row.mask_path)
        raw = cv2.imread(row.mask_path, cv2.IMREAD_UNCHANGED)
        values = set(np.unique(raw).tolist())
        if raw.ndim != 2 or raw.dtype != np.uint8 or not values <= {0, 1}:
            raise ValueError(f"Strict 0/1 single-channel uint8 mask required: {row.mask_path}; {values}")
        labels.update(values)
        stats.append(dict(image_id=row.image_id, patient_id=row.patient_id,
                          height=image.shape[0], width=image.shape[1],
                          foreground_pixels=int(mask.sum()), foreground_fraction=float(mask.mean()),
                          labels=','.join(map(str, sorted(values)))))
    if labels != {0, 1}:
        raise ValueError(f"Dataset does not contain both binary labels: {labels}")
    proposed = make_splits(frame, Config(manifest=str(manifest), seed=seed))
    test = set(proposed.loc[proposed.split == "test", "patient_id"])
    dev = set(proposed.loc[proposed.split == "dev", "patient_id"])
    if test & dev:
        raise AssertionError("Patient leakage between development and test")
    folds = []
    for fold in range(5):
        train_patients = set(proposed.loc[(proposed.split == "dev") & (proposed.fold != fold), "patient_id"])
        val_patients = set(proposed.loc[(proposed.split == "dev") & (proposed.fold == fold), "patient_id"])
        if train_patients & val_patients or (train_patients | val_patients) & test:
            raise AssertionError(f"Patient leakage in fold {fold}")
        folds.append(dict(fold=fold, train_patients=len(train_patients),
                          validation_patients=len(val_patients), overlap=0))
    if source.get("basis") == "publisher_one_image_per_patient" and frame.patient_id.duplicated().any():
        raise ValueError("Manifest conflicts with one-image-per-patient grouping basis")
    selected = frame.sample(n=3, random_state=seed)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), squeeze=False)
    samples = []
    for i, row in enumerate(selected.itertuples()):
        image, mask = read_pair(row.image_path, row.mask_path)
        overlay = image.astype(np.float32)/255
        foreground = mask.astype(bool)
        overlay[foreground] = .6*overlay[foreground] + .4*np.array([0., 1., .65])
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 2].imshow(overlay)
        axes[i, 0].set_title(f"{row.image_id}\npatient: {row.patient_id}", fontsize=9)
        axes[i, 1].set_title(f"Binary mask; foreground {mask.mean():.1%}", fontsize=9)
        axes[i, 2].set_title("Mask overlay", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
        samples.append(dict(image_id=row.image_id, patient_id=row.patient_id,
                            shape=list(image.shape[:2]), foreground_fraction=float(mask.mean())))
    fig.suptitle(f"{info['dataset_name']} — three random samples (seed {seed})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "three_sample_overlays.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    stats_frame = pd.DataFrame(stats)
    report = dict(dataset_name=info["dataset_name"], images=len(frame), patients=frame.patient_id.nunique(),
                  labels=sorted(labels), binary_validation="all masks passed strict single-channel uint8 0/1 scan",
                  all_pairs_aligned=True, exact_duplicates=0,
                  empty_masks=int((stats_frame.foreground_pixels == 0).sum()),
                  full_masks=int((stats_frame.foreground_fraction == 1).sum()),
                  dimensions=stats_frame.groupby(["height", "width"]).size().reset_index(name="images").to_dict("records"),
                  foreground_fraction=dict(min=float(stats_frame.foreground_fraction.min()),
                                           median=float(stats_frame.foreground_fraction.median()),
                                           max=float(stats_frame.foreground_fraction.max())),
                  grouping_evidence=source,
                  grouping_limit="Checks confirm manifest consistency; patient identity is not independently recoverable from anonymous pixels.",
                  patient_image_counts=frame.groupby("patient_id").size().value_counts().sort_index().to_dict(),
                  proposed_split_counts=proposed.groupby("split").agg(images=("image_id", "size"), patients=("patient_id", "nunique")).to_dict("index"),
                  development_test_patient_overlap=0, folds=folds, sampled_images=samples,
                  training_started=False, split_status="proposal only; not used for training")
    stats_frame.to_csv(output / "per_image_checks.csv", index=False)
    proposed.to_csv(output / "proposed_splits.csv", index=False)
    (output / "inspection.json").write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inspect(args.manifest, args.output, args.provenance, args.seed)

if __name__ == "__main__":
    main()
