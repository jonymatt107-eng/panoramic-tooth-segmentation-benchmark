"""Post-hoc segmentation error audit; not clinical triage validation."""
from pathlib import Path
import sys
import argparse
import csv
import gc
import json
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import Config,MODEL_NAMES
from dataset import read_pair,letterbox_pair
from models import load_checkpoint
from robustness_eval import read_locked,corrupt,corruption_rng,tensor_image
from explainability import GradCAM
from train import seed_everything


def boundary_f1(prediction: np.ndarray,target: np.ndarray,valid: np.ndarray,tolerance: float=2) -> float:
    """Symmetric boundary-match F1 at a declared canvas-pixel tolerance."""
    p=(prediction & valid).astype(np.uint8);y=(target & valid).astype(np.uint8)
    kernel=np.ones((3,3),np.uint8)
    pb=(p-cv2.erode(p,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)).astype(bool)&valid
    yb=(y-cv2.erode(y,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)).astype(bool)&valid
    if not pb.any()and not yb.any():return 1.0
    if not pb.any()or not yb.any():return 0.0
    dp=cv2.distanceTransform((~pb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    dy=cv2.distanceTransform((~yb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    precision=float((dy[pb]<=tolerance).mean());recall=float((dp[yb]<=tolerance).mean())
    return 2*precision*recall/(precision+recall)if precision+recall else 0.0


def prepare_input(row: object,cfg: Config,kind: str,value: float,repeat: int,base: tuple | None=None) -> tuple:
    if base is None:
        image,mask=read_pair(row.image_path,row.mask_path)
        image,mask,valid=letterbox_pair(image,mask,cfg);image=image.astype(np.float32)/255
    else:
        image,mask,valid=base
    ys,xs=np.where(valid>.5);y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    degraded=image.copy()
    degraded[y0:y1,x0:x1]=corrupt(image[y0:y1,x0:x1],kind,value,corruption_rng(cfg.seed,row.image_id,kind,repeat),3 if kind!='clean'else 0)
    return image,degraded,mask,valid


def run_audit(root: Path,output: Path) -> None:
    cfg=Config.load(root/'config.json');seed_everything(cfg.seed)
    frame=read_locked(root,cfg);frame=frame[frame.split=='test']
    output.mkdir(parents=True,exist_ok=False)
    raw=pd.read_csv(root/'evaluation_test/per_image.csv',dtype={'image_id':str,'patient_id':str})
    clean=raw[raw.corruption=='clean'].groupby(['model','image_id'],as_index=False).dice.mean().rename(columns={'dice':'clean_dice'})
    worst=raw[raw.severity==3].groupby(['model','image_id','corruption'],as_index=False).dice.mean().merge(clean,on=['model','image_id'],validate='many_to_one')
    worst['dice_loss_from_clean']=worst.clean_dice-worst.dice
    selected=[]
    for kind in ['gaussian_noise','motion_blur','gaussian_blur','resolution']:
        by_image=worst[worst.corruption==kind].groupby('image_id').dice_loss_from_clean.mean().sort_values(ascending=False)
        selected.append(dict(corruption=kind,image_id=str(by_image.index[0]),selection_mean_dice_loss=float(by_image.iloc[0]),selection='post-hoc largest architecture-averaged fold/repetition-averaged Dice loss'))
    pd.DataFrame(selected).to_csv(output/'selected_failure_cases.csv',index=False)
    selected_map={(x['corruption'],x['image_id'])for x in selected}
    cache={}
    for row in frame.itertuples():
        image,_,mask,valid=prepare_input(row,cfg,'clean',0.0,0)
        cache[row.image_id]=(image,mask,valid)
    panel_data={};fields=['model','fold','image_id','patient_id','corruption','severity','repeat','boundary_f1_2px','pixel_auroc']
    conditions=[('clean',0.0),('gaussian_noise',.1),('gaussian_blur',7),('motion_blur',7),('underexposure',2.5),('resolution',.25)]
    with (output/'per_image_boundary_auc.partial.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name in MODEL_NAMES:
            for fold in range(cfg.folds):
                model=load_checkpoint(str(root/name/f'best_model_fold_{fold}.pth'),name,cfg.device)
                for row in frame.itertuples():
                    for kind,value in conditions:
                        repeats=cfg.noise_repeats if kind=='gaussian_noise'else 1
                        for repeat in range(repeats):
                            image,x,target,valid=prepare_input(row,cfg,kind,value,repeat,cache[row.image_id])
                            with torch.inference_mode():
                                logits=model(tensor_image(x).to(cfg.device))
                                probability=logits[0,0].sigmoid().cpu().numpy()
                            v=valid>.5;y=target>.5;prediction=probability>=cfg.threshold
                            auc=float(roc_auc_score(y[v],probability[v]))if kind=='clean'else float('nan')
                            writer.writerow(dict(model=name,fold=fold,image_id=row.image_id,patient_id=row.patient_id,corruption=kind,
                                severity=0 if kind=='clean'else 3,repeat=repeat,boundary_f1_2px=boundary_f1(prediction,y,v),pixel_auroc=auc))
                            if fold==0 and repeat==0 and(kind,row.image_id)in selected_map:
                                cam=GradCAM(model,cfg.threshold)
                                try:_,heat=cam(tensor_image(x).to(cfg.device),torch.from_numpy(valid).unsqueeze(0).unsqueeze(0).to(cfg.device))
                                finally:cam.close()
                                panel_data[(kind,row.image_id,name)]=(probability.copy(),heat.copy())
                    stream.flush()
                del model;gc.collect()
                if cfg.device.type=='mps':torch.mps.empty_cache()
                print(f'Finished audit {name} fold {fold}',flush=True)
    (output/'per_image_boundary_auc.partial.csv').rename(output/'per_image_boundary_auc.csv')
    # The selected arrays enable exact visual reproduction without repeated inference.
    arrays={}
    for j,selection in enumerate(selected):
        row=next(r for r in frame.itertuples()if r.image_id==selection['image_id'])
        kind=selection['corruption'];value=dict(conditions)[kind]
        image,x,target,valid=prepare_input(row,cfg,kind,value,0,cache[row.image_id])
        arrays[f'case{j}_clean']=image;arrays[f'case{j}_degraded']=x;arrays[f'case{j}_target']=target;arrays[f'case{j}_valid']=valid
        for i,name in enumerate(MODEL_NAMES):
            probability,heat=panel_data[(kind,row.image_id,name)]
            arrays[f'case{j}_model{i}_probability']=probability;arrays[f'case{j}_model{i}_heat']=heat
    np.savez_compressed(output/'selected_case_arrays.npz',**arrays)
    (output/'audit_metadata.json').write_text(json.dumps({'purpose':'post-hoc segmentation false-negative and boundary audit, not clinical triage',
        'heldout_patients':len(frame),'folds':cfg.folds,'boundary_tolerance_canvas_pixels':2,'condition_levels':'clean and maximum-severity only',
        'noise_repetitions':cfg.noise_repeats,'illustrative_model_fold':0,'noise_panel_repeat':0,'clinical_actionable_outcomes_available':False,
        'pixel_auc':'per-image tooth-vs-background AUROC, then fold/patient macro; not case-level clinical AUROC',
        'selection_bias':'failure cases deliberately selected post hoc; not a representative safety sample','seed':cfg.seed,'device':str(cfg.device)},indent=2))
    print('SEGMENTATION FAILURE AUDIT COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run_audit(a.run,a.output)
