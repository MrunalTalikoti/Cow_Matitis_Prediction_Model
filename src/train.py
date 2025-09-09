
from __future__ import annotations
import argparse, importlib
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .utils import read_yaml, save_json
from .data_loaders import load_mastitis_dataset
from .preprocess import split_features_target, build_preprocessor

def instantiate(type_path, params):
    mod, cls = type_path.rsplit('.',1); module = __import__(mod, fromlist=[cls]); return getattr(module, cls)(**(params or {}))

def metrics(probas, y):
    preds = (probas>=0.5).astype(int)
    return {"accuracy": float(accuracy_score(y,preds)),
            "f1": float(f1_score(y,preds)),
            "roc_auc": float(roc_auc_score(y, probas))}

def main(args):
    cfg = read_yaml(args.config)
    df = load_mastitis_dataset(args.csv)
    X, y = split_features_target(df, cfg["target_column"], cfg.get("drop_columns",[]))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.get("test_size",0.2), random_state=cfg.get("random_state",42), stratify=y)
    pre, num, cat = build_preprocessor(Xtr)
    best = {"roc_auc":-1, "name":None, "model":None}; results=[]
    for m in cfg["models"]:
        pipe = Pipeline([("pre", pre), ("clf", instantiate(m["type"], m.get("params",{})))])
        pipe.fit(Xtr,ytr)
        try: prob = pipe.predict_proba(Xte)[:,1]
        except Exception:
            s = pipe.decision_function(Xte); prob = (s-s.min())/(s.max()-s.min()+1e-9)
        met = metrics(prob, yte); met["model"]=m["name"]; results.append(met)
        if met["roc_auc"]>best["roc_auc"]: best={"roc_auc":met["roc_auc"],"name":m["name"],"model":pipe}
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_json(results, out/"leaderboard.json")
    if best["model"] is not None:
        joblib.dump(best["model"], out/"best_model.joblib")
        (out/"best_model_name.txt").write_text(best["name"])
    from .utils import save_json as sj
    sj({"numeric": pre.transformers_[0][2], "categorical": pre.transformers_[1][2], "target": cfg["target_column"]}, out/"schema.json")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv", default="data/clinical_mastitis_cows_version1.csv")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--out_dir", default="models")
    main(p.parse_args())
