"""
Builds the two files the dashboard reads:
  data/predictions.csv  - one row per hour of the held-out TEST period for every catchment
  data/metrics.json     - threshold, test metrics, confusion matrix and event counts per catchment

Model: Random Forest (same settings as the team notebooks). The decision threshold is tuned on the
VALIDATION set (max F2); the test period is never used for training or threshold selection.

Edit the paths in CONFIG, then run:  python prepare_data.py
If the team exports its own per-hour prediction files, replace data/predictions.csv with the same columns:
  catchment, timestamp, river_variable, river_value, rainfall_mm, risk_score, alert, actual_next24h
AUT data must NOT be uploaded to a public GitHub repository.
"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, confusion_matrix, precision_score, recall_score,
                             f1_score, fbeta_score, roc_auc_score, average_precision_score)

# ------------------------------------------------------------------ CONFIG
UP = "/mnt/user-data/uploads/"
OUT = "/mnt/user-data/outputs/"
CATCHMENTS = {
    # name: (csv, time column, period column, target column, river column, label for river column, unit)
    "Mahurangi": (UP + "mahurangi_classifier_ready.csv", "timestamp", "data_period", "flood_next_24h", "water_level_m", "Water level", "m"),
    "Otara":     (OUT + "otara_classifier_ready.csv",    "start_time", "data_period", "flood_next_24h", "water_level_m", "Water level", "m"),
    "Puhinui":   (UP + "puhinui_modelling_dataset.csv",  "timestamp",  "period",      "flood_risk_next_24h", "water_level_m", "Water level", "m"),
    "Swanson":   (OUT + "swanson_classifier_ready.csv",  "start_time", "data_period", "flood_next_24h", "water_level_m", "Water level", "m"),
    "Wairau":    (OUT + "wairau_classifier_ready.csv",   "start_time", "data_period", "flood_next_24h", "discharge_m3s", "Discharge", "m³/s"),
}
HIGH_WATER_OVERRIDE = {"Puhinui": 0.865}   # value used in the team's Puhinui notebook
RF = dict(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
          class_weight="balanced_subsample", random_state=42, n_jobs=-1)
# ------------------------------------------------------------------

def f2_threshold(y, p):
    P, R, T = precision_recall_curve(y, p); P, R = P[:-1], R[:-1]
    return float(T[np.argmax(5 * P * R / (4 * P + R + 1e-12))])

def runs(b):
    d = np.diff(np.r_[0, np.asarray(b).astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

frames, metrics = [], {}
for name, (path, tcol, pcol, ycol, rcol, rlabel, unit) in CATCHMENTS.items():
    df = pd.read_csv(path, parse_dates=[tcol]).sort_values(tcol).reset_index(drop=True)
    df[pcol] = df[pcol].astype(str).str.lower().replace({"training": "train"})
    feats = [c for c in df.columns if c not in (tcol, pcol, ycol) and "next" not in c and "future" not in c]
    tr, va, te = (df[df[pcol] == p] for p in ("train", "validation", "test"))
    rf = RandomForestClassifier(**RF).fit(tr[feats], tr[ycol])
    pv, pt = rf.predict_proba(va[feats])[:, 1], rf.predict_proba(te[feats])[:, 1]
    thr = f2_threshold(va[ycol], pv)
    yt, alert = te[ycol].astype(int).values, (pt >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, alert).ravel()
    ev, ep = runs(yt), runs(alert)
    metrics[name] = dict(
        model="Random Forest", river_variable=rlabel, unit=unit, risk_threshold=thr,
        high_water_level=float(HIGH_WATER_OVERRIDE.get(name, tr[rcol].quantile(0.995))),
        test_start=str(te[tcol].min()), test_end=str(te[tcol].max()), test_rows=int(len(te)),
        precision=float(precision_score(yt, alert)), recall=float(recall_score(yt, alert)),
        f1=float(f1_score(yt, alert)), f2=float(fbeta_score(yt, alert, beta=2)),
        roc_auc=float(roc_auc_score(yt, pt)), pr_auc=float(average_precision_score(yt, pt)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        events=len(ev), events_detected=int(sum(alert[a:b].any() for a, b in ev)),
        alert_episodes=len(ep), false_alert_episodes=int(sum(not yt[a:b].any() for a, b in ep)),
    )
    frames.append(pd.DataFrame({
        "catchment": name, "timestamp": te[tcol].values, "river_variable": rlabel,
        "river_value": te[rcol].values, "rainfall_mm": te["rainfall_mm"].values,
        "risk_score": pt, "alert": alert, "actual_next24h": yt}))
    print(name, f"thr={thr:.4f}", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics[name].items()
                                   if k in ("precision", "recall", "f2", "roc_auc", "events", "events_detected", "alert_episodes", "false_alert_episodes")})

pd.concat(frames).to_csv("data/predictions.csv", index=False)
json.dump(metrics, open("data/metrics.json", "w"), indent=2)
print("saved data/predictions.csv and data/metrics.json")
