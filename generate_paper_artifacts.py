"""Descriptive patient-bootstrap estimates; no invented results or fold pseudo-replication."""
from itertools import combinations
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config, MODEL_NAMES
from losses_and_metrics import METRICS

LABELS = dict(zip(MODEL_NAMES, ("U-Net / ResNet-34", "U-Net / EfficientNet-B0", "SegFormer / MiT-B0")))
TAGS = dict(zip(MODEL_NAMES, ("Baseline", "CNN", "Transformer")))


def patient_scores(raw: pd.DataFrame) -> pd.DataFrame:
    cols = list(METRICS) + ["positive_dice", "negative_fp_fraction"]
    raw = raw.copy()
    raw["positive_dice"] = raw.dice.where(raw.foreground == 1)
    raw["negative_fp_fraction"] = (raw.fp / (raw.fp + raw.tn)).where(raw.foreground == 0)
    # Equal weight per image, then per patient, then per fold; noise repetitions averaged first.
    image = raw.groupby(["model", "fold", "corruption", "severity", "patient_id", "image_id"], as_index=False)[cols].mean()
    patient_fold = image.groupby(["model", "fold", "corruption", "severity", "patient_id"], as_index=False)[cols].mean()
    return patient_fold.groupby(["model", "corruption", "severity", "patient_id"], as_index=False)[cols].mean()


def estimate(values: np.ndarray, samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return dict(mean=float("nan"), lower=float("nan"), upper=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(samples)])
    return dict(mean=float(values.mean()), lower=float(np.quantile(means, .025)),
                upper=float(np.quantile(means, .975)), n=len(values))


def summarize(scores: pd.DataFrame, cfg: Config, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (model, corruption, severity), group in scores.groupby(["model", "corruption", "severity"]):
        for metric in metrics:
            rows.append(dict(model=model, corruption=corruption, severity=int(severity), metric=metric,
                             **estimate(group[metric].to_numpy(), cfg.bootstrap_samples, cfg.seed)))
    return pd.DataFrame(rows)


def ci_text(row: pd.Series) -> str:
    if not np.isfinite(row["mean"]):
        return "NA"
    return f"{row['mean']:.3f} [{row['lower']:.3f}, {row['upper']:.3f}]"


def ci_cell(row: pd.Series) -> str:
    if not np.isfinite(row["mean"]):
        return "NA"
    return r"\shortstack{"+f"{row['mean']:.3f}"+r"\\{\scriptsize ["+f"{row['lower']:.3f}, {row['upper']:.3f}"+r"]}}"


def latex_table(headers: list[str], rows: list[list[str]], path: Path) -> None:
    body = "\\begin{tabular}{" + "l" + "r"*(len(headers)-1) + "}\n\\toprule\n"
    body += " & ".join(headers) + " \\\\\n\\midrule\n"
    body += "".join(" & ".join(row) + " \\\\\n" for row in rows)
    path.write_text(body + "\\bottomrule\n\\end{tabular}\n")


def generate(root: Path, split: str) -> None:
    cfg = Config.load(root / "config.json")
    evaluation = root / f"evaluation_{split}"
    if not (evaluation / "EVALUATION_COMPLETE").exists():
        raise ValueError("Complete evaluation before generating paper artifacts")
    raw = pd.read_csv(evaluation / "per_image.csv", dtype={"image_id": str, "patient_id": str})
    timing = pd.read_csv(evaluation / "efficiency.csv")
    if set(raw.model) != set(MODEL_NAMES) or raw.duplicated(["model", "fold", "image_id", "corruption", "severity", "repeat"]).any():
        raise ValueError("Incomplete architecture set or duplicate evaluation rows")
    from robustness_eval import conditions
    cohort = pd.read_csv(root / "splits.csv", dtype={"image_id": str, "patient_id": str})
    cohort = cohort[cohort.split == split]
    expected = {(kind, level, repeat) for kind, level, _ in conditions()
                for repeat in range(cfg.noise_repeats if kind == "gaussian_noise" else 1)}
    if set(raw.fold) != set(range(cfg.folds)) or set(raw.image_id) != set(cohort.image_id):
        raise ValueError("Incomplete fold or image coverage")
    for _, group in raw.groupby(["model", "fold", "image_id"]):
        if set(zip(group.corruption, group.severity, group.repeat)) != expected:
            raise ValueError("Incomplete corruption grid")
    if raw.groupby(["model", "fold"]).image_id.nunique().ne(len(cohort)).any():
        raise ValueError("Incomplete model/fold image coverage")
    scores = patient_scores(raw)
    metrics = list(METRICS) + ["positive_dice", "negative_fp_fraction"]
    summary = summarize(scores, cfg, metrics)
    clean = scores[scores.corruption == "clean"][["model", "patient_id"] + metrics]
    paired = scores.merge(clean, on=["model", "patient_id"], suffixes=("", "_clean"), validate="many_to_one")
    for metric in metrics:
        paired[metric] -= paired[f"{metric}_clean"]
    deltas = summarize(paired, cfg, metrics)
    out = evaluation / "paper_artifacts"
    out.mkdir(exist_ok=True)
    scores.to_csv(out / "patient_scores.csv", index=False)
    summary.to_csv(out / "metric_summary.csv", index=False)
    deltas.to_csv(out / "paired_degradation_deltas.csv", index=False)
    contrasts = []
    for left, right in combinations(MODEL_NAMES, 2):
        a, b = scores[scores.model == left], scores[scores.model == right]
        joined = a.merge(b, on=["patient_id", "corruption", "severity"], suffixes=("_a", "_b"), validate="one_to_one")
        for (kind, severity), group in joined.groupby(["corruption", "severity"]):
            contrasts.append(dict(model_a=left, model_b=right, corruption=kind, severity=severity,
                **estimate((group.dice_a-group.dice_b).to_numpy(), cfg.bootstrap_samples, cfg.seed)))
    pd.DataFrame(contrasts).to_csv(out / "paired_architecture_differences.csv", index=False)
    efficiency = timing.groupby("model").mean(numeric_only=True)
    efficiency.to_csv(out / "efficiency_summary.csv")
    rows, macros = [], []
    for name in MODEL_NAMES:
        subset = summary[(summary.model == name) & (summary.corruption == "clean")].set_index("metric")
        row = [LABELS[name]] + [ci_cell(subset.loc[m]) for m in ("dice", "iou", "sensitivity", "specificity")]
        row += [f"{efficiency.loc[name, 'parameters']/1e6:.2f}", f"{efficiency.loc[name, 'latency_ms']:.2f}"]
        rows.append(row)
        for metric, suffix in (("dice", "Dice"), ("iou", "IoU"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity")):
            macros.append(f"\\newcommand{{\\{TAGS[name]}{suffix}}}{{{ci_text(subset.loc[metric])}}}")
        macros.append(f"\\newcommand{{\\{TAGS[name]}Latency}}{{{efficiency.loc[name, 'latency_ms']:.2f}}}")
    latex_table(["Architecture", "Dice [95\\% CI]", "IoU [95\\% CI]", "Sensitivity [95\\% CI]", "Specificity [95\\% CI]", "Params (M)", "ms/image"], rows, out / "table1.tex")
    kinds = ["gaussian_noise", "gaussian_blur", "motion_blur", "underexposure", "resolution"]
    rows = []
    for name in MODEL_NAMES:
        row = [LABELS[name]]
        for kind in kinds:
            s = deltas[(deltas.model == name) & (deltas.corruption == kind) & (deltas.severity == 3) & (deltas.metric == "dice")].iloc[0]
            row.append(ci_cell(s))
        rows.append(row)
    latex_table(["Architecture", "Noise", "Gaussian blur", "Motion blur", "Underexposure", "Resolution"], rows, out / "table2.tex")
    splits = pd.read_csv(root / "splits.csv", dtype={"patient_id": str})
    cohort = splits[splits.split == split]
    macros.extend([f"\\newcommand{{\\TestPatients}}{{{cohort.patient_id.nunique()}}}",
                   f"\\newcommand{{\\TestImages}}{{{len(cohort)}}}",
                   f"\\newcommand{{\\DevelopmentPatients}}{{{splits[splits.split == 'dev'].patient_id.nunique()}}}",
                   f"\\newcommand{{\\DevelopmentImages}}{{{len(splits[splits.split == 'dev'])}}}"])
    (out / "results.tex").write_text("\n".join(macros) + "\n")
    sns.set_theme(style="whitegrid", context="paper")
    fig, grid = plt.subplots(3, 2, figsize=(8, 8), sharey=True)
    axes = grid.flatten()
    colors = sns.color_palette("colorblind", 3)
    for i, (ax, kind) in enumerate(zip(axes, kinds)):
        for name, color in zip(MODEL_NAMES, colors):
            s = summary[(summary.model == name) & (summary.metric == "dice") & summary.corruption.isin(["clean", kind])].sort_values("severity")
            x = s.severity.to_numpy()
            ax.plot(x, s["mean"], "o-", color=color, label=LABELS[name])
            ax.fill_between(x, s.lower.to_numpy(), s.upper.to_numpy(), color=color, alpha=.13)
        ax.set(title=kind.replace("_", " "), xlabel="Severity (0 = clean)", xticks=[0, 1, 2, 3], ylim=(0, 1))
        if i%2 == 0:
            ax.set_ylabel("Patient macro Dice (95% CI)")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=10, frameon=False)
    fig.tight_layout(); fig.savefig(out / "figure1_decay.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    points = []
    for name, color in zip(MODEL_NAMES, colors):
        s = summary[(summary.model == name) & (summary.corruption == "clean") & (summary.metric == "dice")].iloc[0]
        x, y = float(efficiency.loc[name, "latency_ms"]), float(s["mean"])
        ax.errorbar(x, y, yerr=[[y-s.lower], [s.upper-y]], fmt="o", color=color, capsize=4)
        right_side = x > float(efficiency.latency_ms.median())
        ax.annotate(LABELS[name], (x, y), xytext=(-6 if right_side else 6, 8),
                    ha="right" if right_side else "left", textcoords="offset points", fontsize=8)
        points.append((x, y))
    frontier = sorted((x, y) for x, y in points if not any(a <= x and b >= y and (a < x or b > y) for a, b in points))
    if frontier:
        ax.plot(*zip(*frontier), "--", color="gray", label="Descriptive Pareto frontier")
    ax.set(xlabel="Forward latency, batch 1 (ms/image; lower is better)", ylabel="Clean patient macro Dice (higher is better)", ylim=(0, 1))
    ax.margins(x=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "figure2_pareto.png", dpi=300); plt.close(fig)
    shutil.copyfile(Path(__file__).with_name("manuscript.tex"), out / "manuscript.tex")
    (out / "analysis_metadata.json").write_text(json.dumps({"split": split, "bootstrap_samples": cfg.bootstrap_samples,
        "seed": cfg.seed, "preprocessing": "LongestMaxSize plus paired centered zero padding; padding excluded from loss, metrics and corruption", "estimator": "equal patient mean of image/repetition/fold-averaged metrics",
        "uncertainty": "patient percentile bootstrap conditional on fitted models; descriptive, no multiplicity correction",
        "latency_protocol": timing[["warmup", "iterations", "latency_batch_size"]].drop_duplicates().to_dict("records")}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("runs/main"))
    p.add_argument("--split", choices=("test", "external"), default="test")
    args = p.parse_args()
    generate(args.run, args.split)
