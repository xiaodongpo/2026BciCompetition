"""Log order_bench results computed on TianXuan (pulled back as results/order20/<tag>_<variant>.json)."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from src.utils.exp_log import log_experiment

split = "order20_cv5_dev46"
args = sys.argv[1:]
if args and args[0].startswith("--split="):
    split, args = args[0].split("=", 1)[1], args[1:]
for path in args:
    d = json.load(open(path))
    cfg = d.pop("config", {})
    keys = ["infer_ms", "pooled_sens", "pooled_spec", "pooled_f1", "rec_auc_mean", "rec_auc_std", "order20_mean_auc", "order20_mean_bacc", "order20_min_auc", "order20_min_bacc", "order20_max_auc", "order20_score", "orig_auc", "orig_bacc", "worst_pattern", "order20_rec_auc_mean", "pooled_auc", "pooled_bacc"]
    res = {k: d[k] for k in keys if k in d}
    name = os.path.basename(path)[:-5]
    log_experiment(name, cfg.get("model", "lgb"), split, 0, cfg, res, compliant="yes-online-adapt", device="TianXuan CPU")
    print(f"{name}: mean AUC {res['order20_mean_auc']:.4f} bACC {res['order20_mean_bacc']:.4f} score {res['order20_score']:.4f} min AUC {res['order20_min_auc']:.4f}")
