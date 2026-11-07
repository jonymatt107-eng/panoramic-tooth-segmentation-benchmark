# Verification performed

## Completed real-data study (November 6, 2026)

All 15 MPS fold models and 21,780 held-out records passed final integrity/coverage checks. The frozen 263/66 membership, all source/data hashes, checkpoint selection and exclusion of padding were verified. The 9-page manuscript and 3-page summary were compiled and all pages visually inspected after correcting overlapping Pareto labels. Twelve focused protocol tests passed before launch. Full checkpoint hashes and audit details are in the sibling study_results/verification.json.

## Historical scaffold verification

The earlier limitations below describe the initial software-only checks, before the subsequently authorized real-data study.

Verified on November 5, 2026 (user-local date). See `verification_environment.json` and `requirements-verified.txt` for hardware and dependency versions.

## Passed

- Nine pytest tests: patient partition/fold separation; explicit cross-partition patient overlap rejection; manually calculated confusion metrics and undefined ratios; stable extreme-logit loss/backpropagation; reproducible paired corruptions and bounded image ranges; equal patient weighting; MiT attention forward/gradient/state-key equivalence; native-versus-matrix bilinear forward/gradient equivalence; grayscale image conversion, binary-mask validation and higher-bit-depth rejection.
- Python compilation of all source modules.
- `pip check`: no broken requirements in the verification environment.
- Complete CPU synthetic integration: all three architectures, two folds, one training epoch, checkpoint reload, every corruption condition, timing, patient statistics, both LaTeX tables, both figures, and the five-panel Grad-CAM output.
- Complete MPS synthetic integration with strict deterministic algorithms and FP32: the same three-architecture pipeline passed after the compatibility fix described below.
- Native-resolution MPS checks: all three models accepted a `1 × 3 × 512 × 512` tensor, produced `1 × 1 × 512 × 512` logits, completed loss backward with finite parameter gradients and produced finite normalized 512 × 512 Grad-CAM maps.
- Artifact generation rerun after adding corruption/fold/image coverage validation and fixing plot label placement.
- Generated decay and Pareto figures and comparative Grad-CAM panels visually inspected. LaTeX table row terminators checked.

## MPS compatibility issue resolved

The first strict MPS integration attempt failed in SegFormer's bilinear interpolation backward: PyTorch's decomposition dispatched `UnsafeIndexBackward0`, which called an MPS index-accumulation operation without deterministic support. Both CNN architectures had already trained successfully.

`mps_compat.py` retains native bilinear forward interpolation and replaces its MPS backward with the mathematically equivalent separable matrix adjoint. It also uses equivalent unbind key/value extraction. The compatibility wrapper operates on the selected device without CPU gradient transfers. Forward interpolation, trainable parameters, model outputs, state-dict keys and inference operators are preserved. Numerical gradient equivalence against native interpolation was tested for both alignment conventions and multiple resize ratios. The full MPS pipeline and 512-pixel checks then passed without relaxing `torch.use_deterministic_algorithms(True)`.

## Limits of verification

No patient data, pretrained ImageNet downloads, full five-fold/100-epoch experiment, batch-eight memory stress test or clinical performance validation was run. Synthetic integration used 64-pixel images, two folds, one epoch, random encoder initialization, one noise repetition, 20 bootstrap samples, one warm-up and two timed forwards. Those numbers and scores are software test settings, not study results. Full-resolution checks used batch one and do not establish batch-eight memory capacity.

The final image-loader change was covered by the added validation test. The complete integration run preceded that loader change; it uses the same RGB PNG representation whose decoding is preserved. The subsequent artifact-only coverage/plot changes were exercised against the saved synthetic MPS evaluation.

No LaTeX engine was installed, so `manuscript.tex` has not been compiled or visually inspected as a PDF. Commands for compilation are supplied in README.md. Dataset, ethics, authorship, cohort details and result interpretation remain explicitly marked in the manuscript. External evaluation support is implemented, but no real external dataset was available.

The only warnings in the completed MPS integration were Albumentations' recommendation to replace ShiftScaleRotate with Affine. ShiftScaleRotate was retained because it was explicitly requested. The initial CPU integration also emitted an optional version-check network warning; subsequent runs disabled the check with `NO_ALBUMENTATIONS_UPDATE=1`.

A local Gemini read-only audit helped identify and summarize issues. Its preliminary claims about Albumentations seeding and MiT output incompatibility were rejected based on successful runtime tests. Aider could not access its configured API key, so no Aider edits were accepted and no paid API fallback was used.
