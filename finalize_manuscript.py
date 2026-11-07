"""Populate a real-data manuscript and compile manuscript and summary PDFs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import subprocess
import pandas as pd
from config import Config, MODEL_NAMES
from generate_paper_artifacts import LABELS, ci_text


def finalize(root: Path, compiler: Path) -> None:
    out = root/'evaluation_test'/'paper_artifacts'
    cfg = Config.load(root/'config.json')
    summary = pd.read_csv(out/'metric_summary.csv')
    deltas = pd.read_csv(out/'paired_degradation_deltas.csv')
    timing = pd.read_csv(out/'efficiency_summary.csv').set_index('model')
    training = []
    for name in MODEL_NAMES:
        for fold in range(cfg.folds):
            history = pd.read_csv(root/name/f'history_fold_{fold}.csv')
            best = history.loc[history.val_dice.idxmax()]
            training.append(dict(model=name,fold=fold,epochs_run=len(history),best_epoch=int(best.epoch),best_validation_dice=float(best.val_dice)))
    train = pd.DataFrame(training)
    train.to_csv(out/'training_summary.csv',index=False)
    clean = summary[(summary.corruption=='clean') & (summary.metric=='dice')].set_index('model')
    ordered = clean['mean'].sort_values(ascending=False)
    paragraphs = [r'\subsection{Clean performance and training budget}',
        f'All {3*cfg.folds} fold models completed within the {cfg.epochs}-epoch cap. '
        f'Training stopped after {train.epochs_run.min()}--{train.epochs_run.max()} epochs across folds; '
        'Table~\\ref{tab:clean} summarizes the frozen 66-image held-out evaluation.']
    for name in MODEL_NAMES:
        t = train[train.model==name]
        paragraphs.append(f'{LABELS[name]} obtained clean Dice {ci_text(clean.loc[name])}, '
                          f'with mean forward latency {timing.loc[name,"latency_ms"]:.2f} ms/image. '
                          f'Its folds ran for {t.epochs_run.min()}--{t.epochs_run.max()} epochs.')
    paragraphs += [f'The highest clean Dice point estimate was observed for {LABELS[ordered.index[0]]}. '
                    'This ranking is descriptive; the paired architecture contrasts are supplied separately and no confirmatory significance test is claimed.',
                   r'\subsection{Paired degradation response}']
    names={'gaussian_noise':'Gaussian noise','gaussian_blur':'Gaussian blur','motion_blur':'horizontal motion blur','underexposure':'joint intensity/contrast attenuation','resolution':'resolution reduction'}
    for name in MODEL_NAMES:
        rows=deltas[(deltas.model==name)&(deltas.metric=='dice')&(deltas.severity==3)]
        descriptions=[f'{names[row.corruption]}: {ci_text(row)}'for _,row in rows.iterrows()]
        paragraphs.append(f'{LABELS[name]} had maximum-severity Dice changes (corrupted minus clean) of '+ '; '.join(descriptions)+'.')
    paragraphs.append('Table~\\ref{tab:robust} reports paired intervals; Figure~\\ref{fig:decay} shows all severity levels. '
                      'Changes reflect controlled digital perturbations of secondary captures and cannot be interpreted as calibrated detector-noise tolerances.')
    paragraphs += [r'\subsection{Efficiency and attribution}',
                   'Figure~\\ref{fig:pareto} presents the point-estimate Pareto frontier for the measured implementation. '
                   'Figure~\\ref{fig:cam} shows the first prespecified held-out example; all five selected comparisons are archived. '
                   'Heatmaps are normalized qualitative attributions and do not quantify explanation fidelity or diagnostic validity.']
    narrative='\n\n'.join(paragraphs)+'\n'
    (out/'empirical_results.tex').write_text(narrative)
    panel=root/'explainability_test_fold_0'/'panel_001.png'
    shutil.copyfile(panel,out/'gradcam_example.png')
    shutil.copyfile(Path(__file__).with_name('manuscript.tex'),out/'manuscript.tex')
    report=r'''\documentclass[11pt]{article}
\usepackage[margin=.8in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,graphicx,hyperref}
\input{results.tex}
\title{Secondary-capture panoramic whole-tooth segmentation: empirical summary}
\author{Apple Silicon MPS benchmark}
\date{}
\begin{document}\maketitle
\noindent\textbf{Scope.} Mendeley version 1, 329 smartphone-photographed panoramic radiographs; whole-tooth foreground masks. Intended application: preprocessing in low-resource tele-dentistry triage workflows. Clinical triage or diagnostic performance was not evaluated.
\paragraph{Frozen protocol.} 263 development images/publisher-supported patient groups, five grouped folds, 66 held-out images/groups, three architectures. ImageNet encoders; aspect-preserving longest-side resize with matched zero padding to $512^2$; padding excluded from loss/metrics; batch eight; FP32 MPS; at most 35 epochs; patience seven on validation Dice. No external validation cohort. Patient identities rely on the publisher's one-image-per-patient statement.
\paragraph{Clean benchmark.} Descriptive 95\% patient-bootstrap intervals are conditional on the fitted models. Forward timings exclude decoding, preprocessing and transfer.
\begin{center}\resizebox{\textwidth}{!}{\input{table1.tex}}\end{center}
\paragraph{Maximum-severity degradation.} Dice differences are corrupted minus clean; negative changes indicate deterioration. Gaussian and motion blur are distinct conditions.
\begin{center}\resizebox{\textwidth}{!}{\input{table2.tex}}\end{center}
\begin{center}\includegraphics[width=\textwidth]{figure1_decay.png}\end{center}
\begin{center}\includegraphics[width=.8\textwidth]{figure2_pareto.png}\end{center}
\clearpage
\paragraph{Prespecified qualitative comparison.} First sorted held-out identifier, fold zero; cyan prediction contours and positive-logit Grad-CAM. All five examples are provided separately.
\begin{center}\includegraphics[width=\textwidth]{gradcam_example.png}\end{center}
\paragraph{Interpretation limits.} One secondary-capture source; no external cohort, independent specialist re-adjudication, prospective triage endpoint or physical calibration of simulated degradation. No entirely tooth-free images. Correlated fold models and one seed family do not establish training-sampling uncertainty. Institutional ethics/reuse obligations and author declarations remain for investigators to verify before submission.
\paragraph{Source.} Chowdhury et al., Dental Panoramic Radiography Dataset for Pixel level Semantic Segmentation of Teeth, Mendeley Data V1, 2026, CC BY 4.0. \url{https://doi.org/10.17632/jrz4nj82zv.1}.
\end{document}
'''
    (out/'summary_report.tex').write_text(report)
    for name in ('manuscript.tex','summary_report.tex'):
        subprocess.run([str(compiler.resolve()),'--keep-logs','--keep-intermediates',name],cwd=out,check=True)
    report_data={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'models':list(MODEL_NAMES),
                 'folds':cfg.folds,'heldout_images':66,'development_images':263,
                 'clean_dice':clean.reset_index().to_dict('records'),'epochs':training,
                 'clinical_triage_evaluated':False,'padding_excluded_from_loss_and_metrics':True}
    (out/'execution_summary.json').write_text(json.dumps(report_data,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    a=p.parse_args();finalize(a.run,a.compiler)
