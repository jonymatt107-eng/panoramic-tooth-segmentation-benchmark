from pathlib import Path
import json
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=Path('outputs/study_results'))
root=parser.parse_args().results
fa=root/'failure_audit'
# Rebuild from preserved pre-extension text; repeated runs do not duplicate sections.
manuscript=Path(__file__).with_name('base_manuscript_v1.tex').read_text()
summary=Path(__file__).with_name('base_summary_v1.tex').read_text()
manuscript=manuscript.replace('\\author{Computational benchmark report\\\\[3pt]\\small Author and affiliation details to be supplied before submission}', '\\author{Jonathan Matta\\\\[3pt]\\small Independent researcher; no institutional affiliation\\thanks{Corresponding author: \\href{mailto:jonymatt.107@gmail.com}{jonymatt.107@gmail.com}.}}')
methods=r'''
\subsection{Post-hoc omission, boundary and interpretability audit}
After inspecting the prespecified benchmark results, an exploratory audit recomputed saved-model predictions for the same 66 held-out groups, without retraining, changing thresholds or replacing checkpoints. It considered clean input and the maximum tier of each degradation family, retaining three Gaussian-noise realizations. An additional 7,920 image/fold/condition/repetition evaluations were recorded. Foreground omission was $FN/(TP+FN)$ and pixel NPV was $TN/(TN+FN)$, excluding padding. These are segmentation quantities; a missed tooth pixel is not a clinically missed actionable case.

Pixel AUROC was calculated separately within each clean image using tooth/background reference labels and unthresholded segmentation probabilities. It was averaged over fitted folds and patients, not pooled over pixels; it does not assess clinical case severity or referral discrimination. Boundary $F_1$ used symmetric boundary matching within two canvas pixels: prediction-boundary precision and reference-boundary recall were combined harmonically. Inner mask boundaries were defined by one-pixel erosion; Euclidean nearest-boundary distance supplied the matching criterion. The two-pixel tolerance is an exploratory engineering convention at the 512-square working scale, not an anatomically calibrated error tolerance. Patient bootstrap intervals used the same 2,000 resamples, conditional on fitted models.

For maximum Gaussian noise, Gaussian blur, motion blur and resolution loss, one case per family was selected by the largest clean-to-corrupted Dice loss after averaging across all three architectures, folds and noise realizations. Selection was deliberately post hoc, not an independent representative safety sample. Illustrative panels use fold zero and noise realization zero. They pair the reference mask and boundary with corrupted-input Grad-CAM, prediction contours, and explicit false-negative/false-positive overlays. Qualitative explanations cannot establish causal mechanisms.

\subsection{Clinical triage estimability}
The supplied release contains only background and tooth-region references; the investigator confirmed that independent high-severity/actionable-case labels are unavailable. None of the three segmentation networks was trained or validated to predict clinical case risk. Therefore, clinical sensitivity, specificity, PPV, NPV and AUROC for flagging actionable cases are not estimable from these data. Corruption severity, segmentation failure, and tooth pixel labels were not substituted for clinical outcomes. No clinical decision curve or triage false-negative rate is reported: net benefit requires defined clinical outcomes, case-risk predictions, clinically justified decision thresholds and suitable calibration \cite{dca}. A separate, input-gated evaluation module is supplied for a future independently labeled cohort; its synthetic unit tests are not clinical study results.
'''
manuscript=manuscript.replace('\\section{Results}',methods+'\n\\section{Results}',1)
result=r'''
\subsection{Model Interpretability \& Failure Mode Analysis}
\input{failure_audit/audit_results.tex}
\begin{table}[htbp]\centering
\caption{Clean pixel-level tooth segmentation metrics and boundary matching, with descriptive 95\% patient-bootstrap intervals. PPV and NPV refer to pixel labels; AUROC discriminates tooth pixels from background within images. None is a clinical case-triage metric. Boundary $F_1$ uses two canvas pixels.}\label{tab:pixel}
\resizebox{\textwidth}{!}{\input{failure_audit/table3_pixel_metrics.tex}}
\end{table}
\begin{table}[htbp]\centering
\caption{Clinical case-triage estimability for high-severity/actionable findings. NE means not estimable: independent clinical outcomes and validated case-risk scores are unavailable. No case-level confusion matrix or clinical AUROC is asserted.}\label{tab:clinical}
\resizebox{\textwidth}{!}{\input{failure_audit/table4_clinical_estimability.tex}}
\end{table}
\begin{figure}[htbp]\centering
\includegraphics[width=.95\textwidth]{failure_audit/figure4_segmentation_fn_boundary.png}
\caption{Post-hoc segmentation omission and boundary audit. Bars are patient means after averaging folds/repetitions; intervals are descriptive 95\% patient-bootstrap intervals. The upper panel reports missed reference tooth pixels, not missed actionable patients.}\label{fig:fn}
\end{figure}
\begin{figure}[p]\centering
\includegraphics[width=\textwidth]{failure_audit/failure_panel_gaussian_noise.png}
\caption{Maximum Gaussian-noise example ($\sigma=0.10$), selected post hoc by the largest architecture-averaged Dice drop. Top: degraded input, reference mask, and fold-zero Grad-CAM/prediction comparisons. Bottom: clean input, reference boundary, and segmentation errors (red FN; cyan FP). Noise realization zero is illustrative.}\label{fig:noisefail}
\end{figure}
\begin{figure}[p]\centering
\includegraphics[width=\textwidth]{failure_audit/failure_panel_motion_blur.png}
\caption{Maximum horizontal motion-blur example ($k=7$), selected with the same post-hoc rule. Boundaries and error overlays show local undercoverage or overcoverage directly. Colored heatmaps are positive-logit attributions, not attention maps or proof of biological reasoning.}\label{fig:motionfail}
\end{figure}
'''
manuscript=manuscript.replace('\\section{Discussion}',result+'\n\\clearpage\n\\section{Discussion}',1)
discussion=r'''
\subsection{Model Interpretability \& Failure Mode Analysis}
The false-negative and boundary audit adds detail beyond global overlap: a network can retain a similar Dice while missing tooth regions or displacing boundaries. Figures~\ref{fig:noisefail} and~\ref{fig:motionfail} visualize precisely where those segmentation errors occur in deliberately selected difficult examples. Grad-CAM should be interpreted alongside the reference and error overlays; greater heat intensity does not establish correct localization, explain causal failure, or show that the network recognizes disease.

Observed boundary degradation is compatible with reduced local edge contrast under blur and loss of fine shape cues under resolution reduction. Noise can alter learned local feature responses. Context aggregation, decoder resolution, pretraining and operator implementations may all contribute to architecture-specific patterns. These are explanatory hypotheses, not experimentally isolated causes. The audit is post hoc, the boundary tolerance is arbitrary, and the panels use one fold; no test-driven retraining was undertaken after their selection. Additional Gaussian-blur and resolution examples are supplied in the supplement.

\subsection{Limits on clinical screening and safety claims}
Neither the original benchmark nor this extension measures the detection of high-severity/actionable cases. Pixel sensitivity and specificity describe tooth/background delineation, and pixel NPV is affected by the abundance of background pixels. They cannot be relabeled as clinical recall, negative reassurance, referral performance or diagnostic safety. Table~\ref{tab:clinical} explicitly records this evidence gap. Clinical triage AUROC, a clinically meaningful false-negative audit and decision-curve net benefit require an independently adjudicated endpoint, patient-level risk predictions, appropriate threshold selection/calibration and external or prospective validation. No synthetic severity labels or fabricated clinical outcomes are introduced here.
'''
manuscript=manuscript.replace('\\section{Conclusion}',discussion+'\n\\section{Conclusion}',1)
a=manuscript.index('\\section*{Data, reproducibility and declarations}')
b=manuscript.index('\\begin{thebibliography}',a)
declarations=r'''
\section*{Ethics \& Institutional Governance}
This study exclusively analyzed an open-access secondary benchmark dataset from Mendeley Data, described by its depositors as fully anonymized \cite{dataset}. No new patient recruitment, direct patient interaction, prospective intervention, identifiable personal health information collection, re-identification, or linkage to clinical records was undertaken. Source attribution and the CC BY 4.0 license were retained.

This independently conducted study exclusively comprised secondary computational analysis of the open-access, de-identified Mendeley Data release. No intervention or interaction with living individuals, access to identifiable private information, or linkage enabling patient identification was undertaken. Applying the human-subject definition in 45 CFR 46.102(e), the author classifies this analysis as Not Human Subjects Research (NHSR) \cite{ohrp}. This is the independent author's assessment, not a determination issued by an institutional review board or other review body. No institutional approval, review waiver or exemption certificate was obtained or is claimed. Jonathan Matta has no institutional affiliation or institutional sponsorship; applicable journal requirements for independent confirmation remain to be addressed.

\section*{Author Contributions}
J.M. (Jonathan Matta): conceptualization; computational methodology design; pipeline implementation/software; experiment execution/investigation; failure-audit analysis/formal analysis; and manuscript drafting/writing---original draft. Jonathan Matta confirms sole authorship and is the independent lead investigator; no institution, faculty advisor or co-author is affiliated with this work.

\section*{Funding}
This research did not receive specific grant funding from agencies in the public, commercial, or not-for-profit sectors.

\section*{Competing Interests}
The author declares no competing financial or non-financial interests.

\section*{Data \& Code Availability}
The raw segmentation dataset is publicly accessible through Mendeley Data, version 1, DOI \href{https://doi.org/10.17632/jrz4nj82zv.1}{10.17632/jrz4nj82zv.1}, under CC BY 4.0. Complete source code, training/evaluation scripts, the evaluation review notebook, requirements, frozen split identifiers, per-image results, and provenance records are packaged as \href{run:SupplementarySoftwareS1.zip}{Supplementary Software S1} and the accompanying result files. All 15 trained fold checkpoints and their SHA-256 manifest are packaged separately as \href{run:../SupplementaryWeightsS2.zip}{Supplementary Weights S2}. The compact result archive intentionally excludes that larger weights archive; it is supplied as a separate file. The deidentified raw images remain obtainable from the original dataset repository rather than being duplicated in these supplements.

Source code is publicly available in the following repository:\begin{flushleft}\href{https://github.com/jonymatt107-eng/panoramic-tooth-segmentation-benchmark}{jonymatt107-eng/panoramic-tooth-segmentation-benchmark}\end{flushleft} The repository URL identifies the public code release; the Zenodo record DOI for the companion results and weights is \href{https://doi.org/10.5281/zenodo.24027462}{10.5281/zenodo.24027462}. The release manifest records the archive publication status. The corresponding author is Jonathan Matta, an independent researcher with no institutional affiliation, at \href{mailto:jonymatt.107@gmail.com}{jonymatt.107@gmail.com}. The formal GitHub software release is \href{https://github.com/jonymatt107-eng/panoramic-tooth-segmentation-benchmark/releases/tag/v1.0.0}{v1.0.0}. The companion Zenodo record contains the separately packaged trained weights and study results after upload and publication; the source repository excludes raw images and checkpoints.

\clearpage
\section*{AI-Assisted Tools and Responsibility}
AI-assisted coding and language tools supported implementation, checking and manuscript preparation. They are not authors. J.M. retains responsibility for verifying the analysis, source attribution, declarations and final submitted content, and for satisfying the selected journal's disclosure requirements.

'''
manuscript=manuscript[:a]+declarations+manuscript[b:]
manuscript=manuscript.replace('\\end{thebibliography}',r'''\bibitem{ohrp} US Department of Health and Human Services, Office for Human Research Protections. Coded Private Information or Biospecimens Used in Research, Guidance (2018). \url{https://www.hhs.gov/ohrp/coded-private-information-or-biospecimens-used-research.html}.
\bibitem{dca} Vickers AJ, van Calster B, Steyerberg EW. A simple, step-by-step guide to interpreting decision curve analysis. Diagnostic and Prognostic Research. 2019;3:18. \url{https://doi.org/10.1186/s41512-019-0064-7}.
\end{thebibliography}''')
manuscript=manuscript.replace('high-severity/actionable','high-severity or actionable').replace('tooth/background','tooth versus background').replace('selection/calibration','selection and calibration').replace('\\begin{document}','\\setlength{\\emergencystretch}{5em}\n\\begin{document}',1)
(root/'manuscript.tex').write_text(manuscript)
summary=summary.replace('\\author{Apple Silicon MPS benchmark}','\\author{Jonathan Matta\\\\[3pt]\\small Independent researcher; no institutional affiliation\\thanks{Corresponding author: \\href{mailto:jonymatt.107@gmail.com}{jonymatt.107@gmail.com}.}}')
extra=r'''
\clearpage
\section*{Segmentation Failure Audit and Clinical Triage Evidence Gap}
\input{failure_audit/audit_results_summary.tex}
\begin{center}\resizebox{\textwidth}{!}{\input{failure_audit/table3_pixel_metrics.tex}}\end{center}
\noindent\small Values are pixel-level tooth/background metrics, not clinical case-triage performance. Boundary $F_1$: two canvas pixels; AUROC: per-image tooth/background ranking.\normalsize
\begin{center}\resizebox{\textwidth}{!}{\input{failure_audit/table4_clinical_estimability.tex}}\end{center}
\noindent NE: not estimable without independent high-severity/actionable-case references and suitable clinical case-risk scores. No clinical decision-curve net benefit or triage safety claim is made.
\begin{center}\includegraphics[width=.93\textwidth]{failure_audit/figure4_segmentation_fn_boundary.png}\end{center}
\clearpage
\section*{Selected Maximum-Severity Failures}
\noindent Post-hoc worst architecture-averaged paired Dice-loss examples; fold zero, noise realization zero. These illustrate segmentation errors, not diagnostic failures or a representative safety sample. Red: missed reference tooth pixels; cyan: false-positive segmentation. Heatmaps are Grad-CAM attributions, not attention or causal explanations.
\begin{center}\includegraphics[width=\textwidth]{failure_audit/failure_panel_gaussian_noise.png}\end{center}
\begin{center}\includegraphics[width=\textwidth]{failure_audit/failure_panel_motion_blur.png}\end{center}
\clearpage
\section*{Declarations and Availability}
\textbf{Author:} Jonathan Matta, sole author and independent investigator; no institutional affiliation. Correspondence: \href{mailto:jonymatt.107@gmail.com}{jonymatt.107@gmail.com}. J.M.: conceptualization, computational methodology, software/pipeline implementation, experiment execution, failure analysis and manuscript drafting.\\
\textbf{Funding:} No specific grant funding from public, commercial or not-for-profit sectors.\\
\textbf{Interests:} The author declares no competing financial or non-financial interests.\\
\textbf{Governance:} Public anonymized secondary Mendeley data; no new patient contact, intervention or PHI collection. Independently conducted; no institutional affiliation or sponsorship. No formal institutional ethics review or exemption determination has been obtained. The author classifies this secondary analysis as Not Human Subjects Research (NHSR) under 45 CFR 46.102(e), because it involved neither interaction/intervention with living individuals nor identifiable private information. This is an author assessment, not a formal review-body determination; no approval, waiver or exemption certificate is claimed.\\
\textbf{Data/code:} Dataset DOI \url{https://doi.org/10.17632/jrz4nj82zv.1}; source, notebook and evaluation artifacts are supplied with \href{run:SupplementarySoftwareS1.zip}{Supplementary Software S1}; all checkpoints are in the separately supplied \href{run:../SupplementaryWeightsS2.zip}{Supplementary Weights S2}. Public source repository: \url{https://github.com/jonymatt107-eng/panoramic-tooth-segmentation-benchmark}. Zenodo companion-record DOI: \href{https://doi.org/10.5281/zenodo.24027462}{10.5281/zenodo.24027462}. GitHub software release: \href{https://github.com/jonymatt107-eng/panoramic-tooth-segmentation-benchmark/releases/tag/v1.0.0}{v1.0.0}. The companion Zenodo record is intended to archive the weights and final results.
'''
summary=summary.replace('\\end{document}',extra+'\n\\end{document}')
summary=summary.replace('\\begin{document}','\\setlength{\\emergencystretch}{5em}\n\\begin{document}',1)
(root/'summary_report.tex').write_text(summary)
# Separate supplementary stress-case panels.
supp=r'''\documentclass[11pt]{article}
\usepackage[margin=.8in]{geometry}\usepackage[T1]{fontenc}\usepackage{lmodern,graphicx,hyperref}\setlength{\emergencystretch}{3em}
\title{Supplementary Failure-Mode Panels}\author{Jonathan Matta}\date{}\begin{document}\maketitle
\noindent Deliberately selected post-hoc worst paired Dice-loss examples, averaged across architectures, folds and repetitions for selection. Panels use fold zero and are qualitative segmentation illustrations, not clinical case-triage evidence. Red denotes missed tooth pixels; cyan denotes false-positive segmentation. Grad-CAM is a logit attribution, not an attention map or causal explanation.
\section*{S1: Maximum Gaussian Blur ($k=7$)}
\includegraphics[width=\textwidth]{failure_audit/failure_panel_gaussian_blur.png}
\section*{S2: Maximum Resolution Loss (width 128)}
\includegraphics[width=\textwidth]{failure_audit/failure_panel_resolution.png}
\end{document}'''
(root/'supplementary_failure_panels.tex').write_text(supp)
(root/'author_statements.json').write_text(json.dumps({'author':'Jonathan Matta','role':'sole author / independent investigator','affiliation':'Independent researcher; no institutional affiliation','correspondence_email':'jonymatt.107@gmail.com',
    'contributions':['Conceptualization','Computational methodology','Pipeline implementation','Experiment execution','Failure audit analysis','Manuscript drafting'],
    'funding':'No specific grant funding, as declared by author','interests':'No competing financial or non-financial interests, as declared by author',
    'ethics_status':'Author-assessed NHSR under 45 CFR 46.102(e); no formal review-body determination; independent researcher',
    'ethics_reference':'No protocol reference or exemption determination obtained',
    'clinical_labels':'Author confirmed unavailable; segmentation audit authorized instead',
    'planned_code_release':'https://github.com/jonymatt107-eng/panoramic-tooth-segmentation-benchmark'},indent=2))
print('Prepared updated manuscripts and declarations; clinical estimability explicitly separated from pixel metrics.')
