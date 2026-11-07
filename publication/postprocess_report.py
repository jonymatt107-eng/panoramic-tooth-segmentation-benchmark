"""Publication layout/text refinement only; no inference, retraining or metric changes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
root=Path('outputs/clinical_segmentation_benchmark/runs/mendeley_tele_dentistry_v1')
out=root/'evaluation_test/paper_artifacts'
models=['unet_resnet34','unet_efficientnet_b0','segformer_b0']
labels=['U-Net / ResNet-34','U-Net / EfficientNet-B0','SegFormer / MiT-B0']
s=pd.read_csv(out/'metric_summary.csv')
e=pd.read_csv(out/'efficiency_summary.csv').set_index('model')
sns.set_theme(style='whitegrid',context='paper')
fig,ax=plt.subplots(figsize=(7,4.5));points=[]
for name,label,color in zip(models,labels,sns.color_palette('colorblind',3)):
 row=s[(s.model==name)&(s.corruption=='clean')&(s.metric=='dice')].iloc[0]
 x,y=float(e.loc[name,'latency_ms']),float(row['mean'])
 ax.errorbar(x,y,yerr=[[y-row.lower],[row.upper-y]],fmt='o',color=color,capsize=4,label=label)
 points.append((x,y))
frontier=sorted((x,y)for x,y in points if not any(a<=x and b>=y and(a<x or b>y)for a,b in points))
ax.plot(*zip(*frontier),'--',color='gray',label='Descriptive Pareto frontier')
ax.set(xlabel='Forward latency, batch 1 (ms/image; lower is better)',ylabel='Clean patient macro Dice (higher is better)',ylim=(.86,.92))
ax.margins(x=.12);ax.legend(loc='upper right',fontsize=9)
fig.tight_layout();fig.savefig(out/'figure2_pareto.png',dpi=300);plt.close(fig)
p=out/'manuscript.tex';text=p.read_text()
text=text.replace('The empirical results quantify segmentation and computational trade-offs under prespecified secondary-capture perturbations. They do not establish caries detection, clinical triage accuracy, deployment safety, or external-domain generalization.', 'ResNet-34 U-Net achieved the highest clean Dice and the smallest mean Dice decrease in each maximum-severity corruption family. EfficientNet-B0 had the lowest measured forward latency. These results do not establish clinical triage accuracy, deployment safety, or external-domain generalization.')
text=text.replace('Hardware and package versions are archived in environment metadata.', 'Execution used an Apple M5 Pro MacBook Pro with 24 GB unified memory and macOS 26.6.1, Python 3.14.7, PyTorch 2.15.0, torchvision 0.29.2, and Albumentations 2.0.8. Full package and hardware metadata are archived.')
if 'The Dice axis is zoomed' not in text:
 text=text.replace('Clean Dice and forward latency, with the descriptive point-estimate Pareto frontier.', 'Clean Dice and forward latency, with the descriptive point-estimate Pareto frontier. The Dice axis is zoomed to 0.86--0.92 to display point differences and intervals; its full possible range is 0--1.')
needle='\\section{Discussion}\n'
addition='''Within this single-source evaluation, ResNet-34 U-Net had the highest clean Dice, \\BaselineDice{}, and the smallest mean decrease in all five maximum-severity corruption subconditions. EfficientNet-B0 used 6.25 million parameters versus 24.44 million for the baseline, with the lowest measured forward latency (\\CNNLatency{} versus \\BaselineLatency{} ms/image), but lower clean overlap. SegFormer had the fewest parameters (3.71 million) and the longest measured latency (\\TransformerLatency{} ms/image). These observations describe the present implementations and device, rather than a general claim about convolution or attention.

Resolution loss produced the largest mean Dice decrease for the baseline (0.055) and EfficientNet (0.119), whereas horizontal motion blur produced the largest decrease for SegFormer (0.099). Relative retention therefore differed by degradation family. Gaussian noise at the maximum prespecified level reduced Dice by 0.014, 0.048 and 0.017 for the baseline, EfficientNet and SegFormer, respectively. These comparisons are descriptive and their clinical importance has not been established.

'''
if 'Within this single-source evaluation' not in text:
 text=text.replace(needle,needle+addition)
text=text.replace('The observed performance and latency quantify implementation-specific trade-offs for a potential low-resource tele-dentistry preprocessing component.', 'The baseline achieved the highest overlap and retained the most Dice under the specified maximum-severity perturbations, while EfficientNet provided the lowest measured forward latency with fewer parameters. These are implementation-specific trade-offs for a potential low-resource tele-dentistry preprocessing component.')
if r'\setlength{\itemsep}{2pt}' not in text:
 text=text.replace(r'\begin{thebibliography}{9}',r'\begin{thebibliography}{9}\small\setlength{\itemsep}{2pt}')
p.write_text(text)
p=out/'summary_report.tex';text=p.read_text().replace('\\usepackage{lmodern,booktabs,graphicx,hyperref}', '\\usepackage{lmodern,booktabs,graphicx,hyperref}\n\\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}\n\\setlength{\\emergencystretch}{3em}')
text=text.replace('\\clearpage\n','')
if '\\paragraph{Observed trade-off.}' not in text:
 text=text.replace('\\paragraph{Maximum-severity degradation.}', '\\paragraph{Observed trade-off.} ResNet-34 U-Net had the highest clean overlap and the smallest mean Dice decrease in every maximum-severity corruption family. EfficientNet-B0 had the lowest measured forward latency. The smallest model, SegFormer, was slower on this MPS implementation.\n\\paragraph{Maximum-severity degradation.}')
if 'Dice axis zoomed to' not in text:
 text=text.replace('\\begin{center}\\includegraphics[width=.8\\textwidth]{figure2_pareto.png}\\end{center}', '\\begin{center}\\includegraphics[width=.8\\textwidth]{figure2_pareto.png}\\end{center}\n\\noindent\\small Dice axis zoomed to 0.86--0.92; full possible range 0--1. Timing: Apple M5 Pro, 24 GB unified memory, FP32 MPS; 100 warm-up and 200 measured iterations.\\normalsize\n')
p.write_text(text)
(out/'publication_refinements.json').write_text(json.dumps({'purpose':'Post-run report refinement only','changes':['Replaced overlapping Pareto annotations with a legend','Zoomed and explicitly labeled Dice axis 0.86 to0.92','Added measured hardware and empirical interpretation to manuscript','Removed forced summary page break'],'metrics_changed':False,'models_retrained':False,'inference_rerun':False},indent=2))
print('Refined plot labels and empirical text; no numerical results or trained sources changed.')
