"""Summarize post-hoc pixel errors and produce explicitly non-clinical figures."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import MODEL_NAMES
from generate_paper_artifacts import LABELS,estimate,ci_cell,latex_table


def summarize(root: Path) -> None:
    out=root/'failure_audit'
    extra=pd.read_csv(out/'per_image_boundary_auc.csv',dtype={'image_id':str,'patient_id':str})
    raw=pd.read_csv(root/'per_image.csv',dtype={'image_id':str,'patient_id':str})
    keys=['model','fold','image_id','patient_id','corruption','severity','repeat']
    merged=extra.merge(raw,on=keys,validate='one_to_one')
    assert len(extra)==len(merged)==7920
    merged['pixel_npv']=merged.tn/(merged.tn+merged.fn)
    merged['foreground_fn_fraction']=merged.fn/(merged.tp+merged.fn)
    merged['pixel_sensitivity']=merged.sensitivity;merged['pixel_specificity']=merged.specificity
    merged['pixel_ppv']=merged.precision
    metrics=['pixel_sensitivity','pixel_specificity','pixel_ppv','pixel_npv','pixel_auroc','boundary_f1_2px','foreground_fn_fraction']
    fold=merged.groupby(['model','fold','patient_id','corruption','severity'],as_index=False)[metrics].mean()
    scores=fold.groupby(['model','patient_id','corruption','severity'],as_index=False)[metrics].mean()
    rows=[]
    for (name,kind,severity),group in scores.groupby(['model','corruption','severity']):
        for metric in metrics:
            values=group[metric].to_numpy()
            if not np.isfinite(values).any():continue
            rows.append(dict(model=name,corruption=kind,severity=int(severity),metric=metric,**estimate(values,2000,42)))
    summary=pd.DataFrame(rows)
    scores.to_csv(out/'patient_auxiliary_metrics.csv',index=False)
    summary.to_csv(out/'pixel_and_boundary_summary.csv',index=False)
    headers=['Architecture','Sensitivity','Specificity','PPV','NPV','Pixel AUROC',r'Boundary $F_1$']
    rows=[]
    for name in MODEL_NAMES:
        s=summary[(summary.model==name)&(summary.corruption=='clean')].set_index('metric')
        rows.append([LABELS[name]]+[ci_cell(s.loc[k])for k in metrics[:6]])
    latex_table(headers,rows,out/'table3_pixel_metrics.tex')
    clinical=[]
    for name in MODEL_NAMES:
        for metric in ['sensitivity','specificity','ppv','npv','auroc','net_benefit']:
            clinical.append(dict(model=name,metric=metric,estimate=None,status='not_estimable',reason='No independent actionable-case outcomes or validated case-risk model supplied'))
    pd.DataFrame(clinical).to_csv(out/'clinical_triage_estimability.csv',index=False)
    latex_table(['Architecture','Clinical sensitivity / specificity','Clinical PPV / NPV','Clinical AUROC'],
        [[LABELS[n],'NE','NE','NE']for n in MODEL_NAMES],out/'table4_clinical_estimability.tex')
    sns.set_theme(style='whitegrid',context='paper')
    kinds=['clean','gaussian_noise','gaussian_blur','motion_blur','underexposure','resolution']
    names=['Clean','Noise\n'+r'$\sigma=0.10$','Gaussian blur\nk=7','Motion blur\nk=7','Exposure\n'+r'$\gamma=2.5,c=0.5$','Resolution\nwidth=128']
    colors=sns.color_palette('colorblind',3)
    fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
    for ax,metric,ylabel,scale in zip(axes,['foreground_fn_fraction','boundary_f1_2px'],['Reference tooth pixels missed (%)','Boundary F1 at 2-pixel tolerance'],[100,1]):
        for j,(name,color)in enumerate(zip(MODEL_NAMES,colors)):
            s=summary[(summary.model==name)&(summary.metric==metric)].set_index('corruption').loc[kinds]
            x=np.arange(len(kinds))+(j-1)*.23
            mean=s['mean'].to_numpy()*scale;lo=s.lower.to_numpy()*scale;hi=s.upper.to_numpy()*scale
            ax.bar(x,mean,width=.22,color=color,label=LABELS[name])
            ax.errorbar(x,mean,yerr=[mean-lo,hi-mean],fmt='none',ecolor='black',capsize=2,lw=.7)
        ax.set_ylabel(ylabel);ax.set_ylim(0,1 if scale==1 else max(25,float(summary[summary.metric==metric].upper.max()*scale+4)))
    axes[0].set_title('Segmentation omission and boundary audit; 66 held-out patient groups (not clinical triage)')
    axes[-1].set_xticks(np.arange(6),names,fontsize=9);axes[-1].set_xlabel('Clean or maximum-severity digital perturbation')
    axes[0].legend(fontsize=8,loc='upper left');fig.tight_layout();fig.savefig(out/'figure4_segmentation_fn_boundary.png',dpi=300);plt.close(fig)
    selected=pd.read_csv(out/'selected_failure_cases.csv');data=np.load(out/'selected_case_arrays.npz')
    panel_metrics=[]
    for j,case in enumerate(selected.itertuples()):
        x=data[f'case{j}_degraded'];image=data[f'case{j}_clean'];target=data[f'case{j}_target']>.5;valid=data[f'case{j}_valid']>.5
        fig,axes=plt.subplots(2,5,figsize=(16,6.6))
        axes[0,0].imshow(x);axes[0,0].set_title('Degraded input')
        axes[0,1].imshow(target,cmap='gray',vmin=0,vmax=1);axes[0,1].set_title('Reference tooth mask')
        axes[1,0].imshow(image);axes[1,0].set_title('Clean input (same case)')
        axes[1,1].imshow(x);axes[1,1].contour(target,levels=[.5],colors='white',linewidths=.6);axes[1,1].set_title('Reference boundary')
        for i,name in enumerate(MODEL_NAMES):
            prob=data[f'case{j}_model{i}_probability'];heat=data[f'case{j}_model{i}_heat']
            pred=(prob>=.5)&valid;fn=(target&~pred)&valid;fp=(~target&pred)&valid
            axes[0,i+2].imshow(x);axes[0,i+2].imshow(np.ma.masked_where(~valid,heat),cmap='inferno',vmin=0,vmax=1,alpha=.45)
            if pred.any()and not pred.all():axes[0,i+2].contour(pred,levels=[.5],colors='cyan',linewidths=.5)
            axes[0,i+2].set_title(LABELS[name]+'\nGrad-CAM + prediction',fontsize=9)
            overlay=x.copy();overlay[fn]=.35*overlay[fn]+.65*np.array([1.,0.,0.]);overlay[fp]=.35*overlay[fp]+.65*np.array([0.,.75,1.])
            axes[1,i+2].imshow(overlay)
            loss=float(fn.sum()/target.sum())
            axes[1,i+2].set_title(f'Missed tooth pixels: {loss:.1%}\nRed: FN; cyan: FP',fontsize=9)
            panel_metrics.append(dict(image_id=case.image_id,corruption=case.corruption,model=name,fold=0,noise_repeat=0,foreground_fn_fraction=loss))
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(case.corruption.replace('_',' ')+f' | {case.image_id} | post-hoc worst paired Dice-loss case; fold 0',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/f'failure_panel_{case.corruption}.png',dpi=300,bbox_inches='tight');plt.close(fig)
    pd.DataFrame(panel_metrics).to_csv(out/'illustrative_panel_metrics.csv',index=False)
    (out/'analysis_scope.json').write_text(json.dumps({'clinical_metrics':'not estimable; user confirmed no clinical labels',
        'computed_metrics':'pixel-level tooth/background metrics, per-image AUROC, and segmentation boundary/foreground omission',
        'patient_groups':66,'patient_bootstrap_samples':2000,'clean_auc_rows':int(extra.pixel_auroc.notna().sum()),'audit_inference_rows':len(extra),
        'decision_curves':'not reported; no independent clinical outcome or validated calibrated case risk',
        'selection':'post-hoc largest Dice degradation, averaged across architectures/folds/repeats; illustrative panels use fold0/repeat0',
        'mechanisms':'visual hypotheses only; no causal attribution or clinical safety claim'},indent=2))
    print(summary[summary.corruption=='clean'][['model','metric','mean','lower','upper']].to_json(orient='records',indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();summarize(a.results)
