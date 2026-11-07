"""Execute the approved MPS study sequentially; preserve progress and failures."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import traceback
from config import Config
from train import train
from robustness_eval import evaluate
from explainability import explain
from generate_paper_artifacts import generate
from finalize_manuscript import finalize


def run(config_path: Path, compiler: Path, resume: bool = False) -> None:
    cfg=Config.load(config_path)
    if cfg.device.type!='mps':
        raise RuntimeError('This study must run on MPS')
    root=Path(cfg.output_dir)
    status_path=root.parent/(root.name+'_status.json')
    status_path.parent.mkdir(parents=True,exist_ok=True)
    def status(stage: str, **extra: object) -> None:
        record={'stage':stage,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra}
        status_path.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    try:
        status('training')
        if not (root/'TRAINING_COMPLETE').exists():
            train(cfg,resume=resume)
        status('heldout_robustness')
        evaluation=root/'evaluation_test'
        if not (evaluation/'EVALUATION_COMPLETE').exists():
            if evaluation.exists():
                evaluation.rename(evaluation.with_name('evaluation_test_interrupted_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
            evaluate(root)
        status('gradcam')
        explain(root,'test',0,5)
        status('tables_and_figures')
        generate(root,'test')
        status('manuscript_and_summary_compilation')
        finalize(root,compiler)
        status('complete',artifacts=str(root/'evaluation_test'/'paper_artifacts'))
        (root/'PIPELINE_COMPLETE').write_text(datetime.now(timezone.utc).isoformat()+'\n')
    except BaseException as error:
        status('failed',error_type=type(error).__name__,error=str(error))
        traceback.print_exc()
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Skip completed folds; restart any interrupted fold from its seeded initialization')
    a=p.parse_args();run(a.config,a.compiler,a.resume)
