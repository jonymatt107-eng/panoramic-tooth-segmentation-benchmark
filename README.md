# Patient-separated segmentation robustness benchmark

A runnable research implementation comparing U-Net/ResNet-34, U-Net/EfficientNet-B0 and SegFormer/MiT-B0 on 2D binary biomedical segmentation. Defaults target dental radiographs, 512 × 512 images, batch eight, FP32 and MPS with CPU fallback. No CUDA selection is used.

**Real-data study completed:** the approved Mendeley secondary-capture whole-tooth experiment completed all 15 MPS fold runs and 21,780 held-out evaluations. The real, populated manuscript and summary are in the sibling `study_results` output directory. `manuscript.tex` in this code directory remains the analysis-linked template; use the empirical manuscript supplied with the results. Synthetic smoke-test scores are not study results. Jonathan Matta confirms sole authorship as an independent researcher with no institutional affiliation. Correspondence: jonymatt.107@gmail.com. No formal institutional ethics review or exemption determination has been obtained; applicable requirements remain unresolved before submission.

## Files

- `config.py`: validated defaults and JSON configuration.
- `dataset.py`: image/mask validation, patient splits and augmentation.
- `models.py`: three SMP factories, checkpoint loading and storage statistics.
- `mps_compat.py`: equivalent MiT key/value extraction and deterministic bilinear interpolation gradients for strict MPS training.
- `losses_and_metrics.py`: BCE/Dice, per-image metrics and synchronized latency.
- `train.py`: grouped CV, AdamW, cosine scheduling, early stopping and CSV histories.
- `robustness_eval.py`: clean and corrupted held-out/external evaluation.
- `explainability.py`: deepest-encoder Grad-CAM and comparative panels.
- `generate_paper_artifacts.py`: patient bootstrap summaries, paired deltas/contrasts, LaTeX tables and figures.
- `manuscript.tex`: full IMRaD draft, references and evidence-dependent fields.
- `test_protocol.py`, `smoke_test.py`: invariant and synthetic integration checks.
- `example_config.json`, `requirements.txt`, `requirements-verified.txt`: configuration, compatible ranges and exact verification environment.

## Install

Use Python 3.11 or newer on a supported macOS/PyTorch combination. Run commands from this directory. Dependency resolution must select matching torch/torchvision releases. Encoder pretraining downloads ImageNet weights on first training use; internet access and adequate disk space are required. Setting `encoder_weights` to `null` trains from scratch and changes the experiment: apply that choice to every architecture and report it.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements-verified.txt` freezes the actual local verification stack, including transitive packages. It is provided for an equivalent Apple Silicon/Python 3.14 environment; it is not a universal cross-platform lock. `environment.json` captures each experiment's stack. Albumentations may check for updates on import; set `NO_ALBUMENTATIONS_UPDATE=1` to disable that network check. These scripts add no telemetry collection.

## Bring your data

The data are not downloaded automatically. Choose one licensed, ethically usable dataset and one binary endpoint. Use lossless single-channel PNG masks with 0/1 or 0/255 encoding. Images may be PNG/JPG/JPEG and must have the same original dimensions as the mask. Multiclass, colored and antialiased masks are rejected; convert intentionally before using the pipeline. Grayscale radiographs become three identical RGB channels. DICOM, 16-bit radiograph intensity calibration and multiclass segmentation are outside this implementation.

Create `data/manifest.csv` with the following header:

```csv
image_id,patient_id,image_path,mask_path,split
img001,patient001,images/img001.png,masks/img001.png,dev
img002,patient002,images/img002.png,masks/img002.png,test
img003,patient003,images/img003.png,masks/img003.png,external
```

The rows above illustrate the schema, not a trainable cohort. Supply enough distinct development patients for five folds. Paths are relative to the manifest directory or absolute. Keep identifiers deidentified and globally unique across sites. The same patient's visits/views/crops must use the same patient_id. Never infer independent patients from image filenames unless the dataset documentation establishes that mapping.

`split` must contain `dev` and `test`, with optional `external`. Every patient must belong to one partition. Prefer an explicit, prespecified test cohort. If the entire split column is omitted, a fixed seed-42 GroupShuffleSplit reserves approximately 20% of patients for testing. External cohorts must be declared before training. Automatic patient grouping cannot be recovered from image content. The loader rejects duplicate image paths, identical file hashes and identical decoded pixels; near-duplicates still need review.

GroupKFold operates only within `dev`. It preserves patient separation but does not stratify by disease, site or sex. Review the saved `cohort_counts.csv` and cohort characteristics. Do not repeatedly select a test split based on model performance. Dataset files and patient assignments are verified against the locked training manifest at evaluation time.

## Run the study

Review `example_config.json` before running. Relative manifest paths are interpreted from the process working directory. Output directories must be new; the code refuses to overwrite an existing training run. Fold indexes are zero-based.

```bash
export NO_ALBUMENTATIONS_UPDATE=1
python train.py --config example_config.json
python robustness_eval.py --run runs/main
python explainability.py --run runs/main --fold 0 --count 5
python generate_paper_artifacts.py --run runs/main
```

Each architecture has its own directory containing `best_model_fold_0.pth` through `best_model_fold_4.pth` and loss/metric history CSVs. A complete five-fold run trains 15 models and can take substantial time. The authorized Mendeley study has completed; use the saved run and empirical results rather than launching duplicate training. Check available unified memory before full-resolution training. Batch size 16 is configurable; lower batch size if necessary, consistently across architectures. Do not silently change image resolution in a scientific run after seeing results.

FP32 is intentional. Automatic mixed precision is not enabled because operator/dtype support varies across MPS versions. Strict deterministic algorithms raise an error on unsupported operations. SegFormer includes an explicit compatibility implementation: native bilinear interpolation is retained in the forward pass, while its backward pass uses the separable interpolation matrix adjoint on MPS instead of atomic scatter accumulation. MiT key/value extraction uses equivalent `unbind`. No gradient computation moves to CPU, and inference retains native interpolation. Output/gradient equivalence is tested; state-dict names and shapes are preserved. See `THIRD_PARTY_NOTICES.txt` for the upstream notice. Preserve the error and document any protocol change. A fixed seed does not guarantee bitwise identity between PyTorch versions or between MPS and CPU. The single-process loader and single CPU thread keep a clear reproducibility/timing baseline. Do not enable MPS CPU-operator fallback without reporting it.

The reserved test data are never used for early stopping. Every best fold model is evaluated separately on the same test set. Scores average across those fold models; this is **not** an inference ensemble. Do not choose a deployment checkpoint or retune parameters based on the test result. An eventual full-development retrain or ensemble is a different estimand and needs a separate prospective evaluation plan.

If an external partition was supplied, evaluate it independently:

```bash
python robustness_eval.py --run runs/main --split external
python generate_paper_artifacts.py --run runs/main --split external
python explainability.py --run runs/main --split external --fold 0 --count 5
```

Document how site/device/time separates the external domain. Without that cohort the study measures internal held-out corruption robustness, not real domain-shift performance. The artifact manuscript uses generic evaluation-count macros; when using external artifacts revise the cohort/results wording and preserve internal results as a separate analysis.

Evaluation writes to a new `evaluation_test` or `evaluation_external` directory. An interrupted evaluation leaves partial output and is not accepted by artifact generation. Preserve or move that incomplete directory before retrying. Training has no resume feature; preserve interrupted runs and use a new output directory. Completed evaluations are not overwritten automatically.

## Degradation definitions

All perturbations occur after deterministic resizing and before ImageNet normalization, on RGB float intensities in [0,1]. The reference mask is never corrupted.

1. Gaussian noise: sigma 0.01, 0.05, 0.10, clipped, three seeded realizations. Seeds are paired across architectures/folds and noise severity.
2. Gaussian blur and horizontal motion blur, evaluated separately: widths 3, 5, 7. Gaussian sigma is chosen by OpenCV from kernel size. Motion uses a normalized central-row kernel.
3. Underexposure proxy: `x_new = c * x ** gamma`, with `(gamma, c)` equal to `(1.25, .9)`, `(1.75, .7)`, `(2.5, .5)`.
4. Resolution: fractions .75, .5, .25 using bilinear downsampling and restoration. At default 512 these are 384, 256, 128 pixels. Three degraded tiers plus clean make four quality tiers.

These are controlled digital perturbations, not physically calibrated clinical sensor simulations. They do not establish diagnostic safety. Square resizing, elastic augmentation and horizontal flips need explicit clinical justification for your target and modality; change the protocol before running if unsuitable.

## Statistics and interpretation

Per-image records include confusion counts and all five requested metrics. Both-empty Dice/IoU are one. Sensitivity with no reference foreground, precision with no predicted foreground, and specificity with no reference background are undefined (`NaN`), not artificially perfect. CSV metric summaries report contributing patient counts. Positive-reference-only Dice and negative-image false-positive pixel fraction are also included.

Noise repeats average within image/fold, images within patient, then folds within patient. Patients receive equal final weight. This avoids treating multiple images, fold models, or perturbation realizations as independent subjects. It is a macro estimator, not pooled pixel accuracy. The validation checkpoint criterion is image-weighted soft Dice loss, which is documented separately from the final patient-weighted estimator. Histories include valid-image counts for each metric.

Confidence intervals are 2,000-resample patient percentile bootstrap intervals conditional on fitted models. Maximum-degradation deltas use the same paired patients; architecture differences are also paired. They are descriptive and do not account for training-set/seed uncertainty, site clustering or multiple comparisons. Do not interpret them as confirmatory significance tests. In small cohorts bootstrap intervals may be uninformative. No power or sample-size adequacy claim is made automatically.

`efficiency.csv` records every fold's parameter/storage counts, mean/SD/median/p95 latency and FPS. Timing defaults are 100 warm-up and 200 measured iterations, batch one. MPS is synchronized around each timed forward. Preprocessing, transfers, postprocessing and explanations are excluded. FPS is reciprocal forward latency, not measured clinical throughput. The Pareto frontier uses point estimates for the three measured systems; report timing conditions and uncertainty before ranking systems.

Grad-CAM targets the mean foreground logit within the detached predicted mask, or the full image if prediction is empty. It uses the deepest encoder spatial tensor for all architectures. Heatmaps are independently normalized and cannot establish biological validity or faithful clinical reasoning. Panels use the first sorted test identifiers; the index CSV records selection. Cyan lines show prediction contours. Only one model is resident at a time.

## Paper outputs

`evaluation_test/paper_artifacts/` contains:

- `table1.tex`: clean metrics with patient CIs, parameters and latency.
- `table2.tex`: maximum-severity paired Dice changes with patient CIs.
- `figure1_decay.png`: all five corruption subconditions and severity curves.
- `figure2_pareto.png`: latency versus clean Dice and a descriptive frontier.
- `patient_scores.csv`, `metric_summary.csv`, `paired_degradation_deltas.csv`, `paired_architecture_differences.csv`, `efficiency_summary.csv`, `analysis_metadata.json`.
- `results.tex` and a copy of `manuscript.tex` that reads those real result macros.

Compile the real-data artifact copy with a LaTeX distribution:

```bash
cd runs/main/evaluation_test/paper_artifacts
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

The source manuscript also compiles without measurements, showing explicit REQUIRED fields and figure/table notices. It intentionally makes no claim that a particular architecture wins. Replace study-specific fields, interpret the real paired outputs, add external evaluation when claiming domain shift, complete ethics/declarations, and adapt to the selected journal. Citations include original architecture/Grad-CAM papers, CLAIM 2024 and official PyTorch documentation.

## Verify without patient data

```bash
python -m pytest test_protocol.py -q
python smoke_test.py --output /tmp/segmentation-smoke-new
```

The smoke test creates a new directory, uses 64-pixel synthetic images, two folds, one epoch, no pretrained download, one noise realization, one warm-up and two timing iterations. It exercises the entire pipeline, including transformer Grad-CAM. These reduced settings must never be used as reported clinical experiment settings. See `VERIFICATION.md` for checks actually performed on the delivered package.

## Current authorized Mendeley study

`study_config.json` freezes the approved 263-development/66-test membership and five development folds from `data/mendeley_teeth_v1/manifest_frozen.csv`. The whole-tooth endpoint uses secondary captures (smartphone photographs of panoramic radiographs) for an engineering evaluation relevant to low-resource tele-dentistry triage preprocessing; clinical triage accuracy is not evaluated.

The current pipeline supersedes the original square-resizing and soft-validation-loss selection described in older scaffold notes. It uses LongestMaxSize and centered zero PadIfNeeded, with an additional validity mask. Padding is excluded from BCE, soft Dice, confusion metrics and Grad-CAM targets. Corruptions affect the content rectangle while preserving the padded border. Checkpoint selection maximizes validation patient-macro hard Dice (threshold .5), with a 35-epoch cap and patience seven. The study configuration requires MPS and prohibits CPU fallback.

The approved sequential runner performs training, robustness evaluation, explanation, artifact generation and compilation:

```bash
python run_study.py --config study_config.json --compiler /absolute/path/to/tectonic
```

A failed process can skip fully completed folds and restart only the interrupted fold from its seeded initialization using `--resume`. The config, frozen splits, data hashes and Python source hashes must match. Do not start a second runner while one is active. Epoch-level optimizer-state resumption is not implemented. Run status is saved beside the experiment directory; all real results remain under `runs/mendeley_tele_dentistry_v1`.

`finalize_manuscript.py` inserts actual CSV-derived observations, creates the training summary, and compiles manuscript.pdf and summary_report.pdf using Tectonic. Investigators still need to verify ethics/reuse obligations and provide author/declaration details before journal submission.

## Independent authorship and public source

Jonathan Matta is the sole author and an independent researcher with no institutional affiliation or sponsorship. Correspondence: jonymatt.107@gmail.com. No college, university or faculty advisor is affiliated with this work. No formal ethics approval or exemption has been obtained or asserted.

Public source: https://github.com/jonymatt107-eng/panoramic-tooth-segmentation-benchmark . No archival DOI has been minted. Raw data remain at https://doi.org/10.17632/jrz4nj82zv.1 under CC BY 4.0. The repository excludes raw images, local run outputs and trained weights. The separately supplied weight archive is not publicly deposited here. The extension quantifies segmentation omissions and boundary errors, not clinical triage utility.
