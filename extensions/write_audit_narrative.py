from pathlib import Path
import sys
import pandas as pd
root=Path('outputs/study_results');fa=root/'failure_audit'
s=pd.read_csv(fa/'pixel_and_boundary_summary.csv')
models=['unet_resnet34','unet_efficientnet_b0','segformer_b0']
labels=['U-Net/ResNet-34','U-Net/EfficientNet-B0','SegFormer/MiT-B0']
def value(model,kind,metric):return s[(s.model==model)&(s.corruption==kind)&(s.metric==metric)].iloc[0]
def ci(row,scale=1):return f"{row['mean']*scale:.3f} [{row['lower']*scale:.3f}, {row['upper']*scale:.3f}]"
def pct(row):return f"{row['mean']*100:.1f}\\% [{row['lower']*100:.1f}, {row['upper']*100:.1f}]"
paragraphs=['The additional audit retained the original 66 held-out groups and all 15 saved fold models. No retraining, threshold change or clinical label synthesis was performed. Table~\\ref{tab:pixel} summarizes clean pixel metrics and two-pixel boundary matching; Table~\\ref{tab:clinical} records that clinical actionable-case performance is not estimable.']
for name,label in zip(models,labels):
 paragraphs.append(f'{label} had clean pixel NPV {ci(value(name,"clean","pixel_npv"))}, pixel AUROC {ci(value(name,"clean","pixel_auroc"))}, and boundary $F_1$ {ci(value(name,"clean","boundary_f1_2px"))}. These intervals are conditional descriptive patient-bootstrap intervals; the high pixel AUROC/NPV values do not indicate clinical triage discrimination or safety.')
noise='; '.join(f'{label}: {pct(value(name,"gaussian_noise","foreground_fn_fraction"))}'for name,label in zip(models,labels))
motion='; '.join(f'{label}: {pct(value(name,"motion_blur","foreground_fn_fraction"))}'for name,label in zip(models,labels))
paragraphs.append('At maximum Gaussian noise, the mean fraction of reference tooth pixels missed was '+noise+'. Under maximum horizontal motion blur it was '+motion+'. Figure~\\ref{fig:fn} contrasts these omission rates with boundary matching across all maximum-severity conditions.')
for name,label in zip(models,labels):
 clean=value(name,'clean','boundary_f1_2px')['mean'];noisy=value(name,'gaussian_noise','boundary_f1_2px')['mean'];blur=value(name,'motion_blur','boundary_f1_2px')['mean']
 paragraphs.append(f'{label} boundary $F_1$ changed from {clean:.3f} on clean input to {noisy:.3f} under maximum noise and {blur:.3f} under maximum motion blur. This quantifies boundary mismatch at the declared tolerance, rather than identifying which clinical findings would be missed.')
paragraphs.append('The selected noise example is image \\texttt{md1\\_image\\_0175}; the selected motion example is \\texttt{md1\\_image\\_0007}. Figures~\\ref{fig:noisefail} and~\\ref{fig:motionfail} show their reference masks, Grad-CAM maps and spatial error overlays. Their selection maximizes an observed paired Dice drop and is therefore explicitly post hoc; it cannot estimate a population case-failure frequency. Gaussian-blur and resolution examples are supplied separately.')
(fa/'audit_results.tex').write_text('\n\n'.join(paragraphs)+'\n')

with (fa/'audit_results.tex').open('a') as handle: handle.write('\n'+"In the illustrated noise case, narrow lower reference extensions are undersegmented by all three models, while scattered false-positive islands and boundary spill coexist with omissions. In the illustrated motion-blur case, central vertical reference components are partly missing across architectures. These are descriptions of the displayed segmentation geometry, not an adjudication of disease or an explanation of the networks' internal causal reasoning."+'\n')
text=(fa/"audit_results.tex").read_text()
text=text.replace('Table~\\ref{tab:pixel}','the pixel-metric table')
text=text.replace('Table~\\ref{tab:clinical}','the clinical-estimability table')
text=text.replace('Figure~\\ref{fig:fn}','the omission/boundary figure')
text=text.replace('Figures~\\ref{fig:noisefail} and~\\ref{fig:motionfail}','the selected stress-case panels')
(fa/"audit_results_summary.tex").write_text(text)
