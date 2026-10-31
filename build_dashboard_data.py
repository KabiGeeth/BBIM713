"""
build_dashboard_data.py
=======================
Builds the two files the Streamlit dashboard (app.py) reads, using the LSTM test predictions
saved by the team's LSTM notebooks:

    predictions.csv  - one row per test hour per catchment
                       (catchment, timestamp, river_variable, river_value, rainfall_mm,
                        risk_score, alert, actual_next24h)
    metrics.json     - per catchment: model, thresholds, test period, test metrics,
                       confusion matrix, event and alert-episode counts

Nothing is retrained here. Every number on the dashboard comes from:
  1. <catchment>_lstm_test_predictions.csv  (saved by each *_LSTM_English notebook), and
  2. the cleaned hourly data file that notebook used (for the river level and rainfall shown on the charts).

Run from the folder that contains the input files:
    python build_dashboard_data.py
The two output files are written next to this script; copy them next to app.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score)

HERE = Path(__file__).parent

# ----------------------------------------------------------------------------- CONFIG
# For each catchment:
#   pred      - LSTM test-prediction file saved by the LSTM notebook
#   base      - cleaned hourly file used for the river level and rainfall shown on the dashboard
#   threshold - LSTM warning threshold selected on validation (printed in the LSTM notebook)
#   high_water- training-period 99.5th-percentile water level used by that LSTM notebook (m)
#   inputs    - input variables the LSTM read (for documentation only; app.py does not use it)
CATCHMENTS = {
    "Mahurangi": dict(pred="mahurangi_lstm_test_predictions.csv", base="mahurangi_cleaned_hourly.csv",
                      threshold=0.6722, high_water=2.43024,
                      inputs=["rainfall_mm", "discharge_m3s", "water_level_m"],
                      # The Mahurangi LSTM notebook counts events on the FULL test timeline, including
                      # hours that had no prediction (no complete 48-hour input window). One short
                      # event (13-14 Jun 2024) falls entirely in such a gap, so it counts as missed.
                      full_timeline_events=True),
    "Otara":     dict(pred="otara_lstm_test_predictions.csv", base="otara_preprocessed.csv",
                      threshold=0.6928, high_water=1.211,
                      inputs=["rainfall_mm", "discharge_m3s", "water_level_m"]),
    "Puhinui":   dict(pred="puhinui_lstm_test_predictions.csv", base="puhinui_modelling_dataset.csv",
                      threshold=0.6888, high_water=0.865,
                      inputs=["rainfall_mm", "discharge_m3s", "water_level_m"]),
    "Swanson":   dict(pred="swanson_lstm_test_predictions.csv", base="swanson_lstm_hourly.csv",
                      threshold=0.6029, high_water=1.108,
                      inputs=["rainfall_mm", "discharge_m3s", "water_level_m"]),
    "Wairau":    dict(pred="wairau_lstm_test_predictions.csv", base="wairau_model_ready_data.csv",
                      threshold=0.6225, high_water=0.929,
                      inputs=["rainfall_mm", "discharge_m3s", "water_level_m"]),
}
# -----------------------------------------------------------------------------


def runs(timestamps, flags):
    """Return (start, end) row positions of each run of consecutive 1s.
    A run also ends where the hourly timeline has a gap, so hours either side of a gap
    are never joined into one event (same rule as the team's RF/XGB/NB notebooks)."""
    flags = np.asarray(flags).astype(int)
    continuous = np.r_[False, np.diff(timestamps.values) == np.timedelta64(1, "h")]
    starts = (flags == 1) & ((np.r_[0, flags[:-1]] == 0) | ~continuous)
    run_id = np.cumsum(starts)
    out = []
    for r in np.unique(run_id[flags == 1]):
        idx = np.where((run_id == r) & (flags == 1))[0]
        out.append((idx[0], idx[-1] + 1))
    return out


def full_test_target(base, high_water, horizon=24):
    """Rebuild the 24-hour target on the full hourly timeline, exactly as the Mahurangi LSTM notebook does:
    chronological 70/15/15 split by row, all 24 future water levels must be present, and the
    24-hour window must stay inside the same period. Returns the test-period hours with a target."""
    b = base.sort_values("timestamp").reset_index(drop=True)
    n = len(b)
    period = np.where(b.index < int(n * 0.70), "train",
                      np.where(b.index < int(n * 0.85), "validation", "test"))
    period = pd.Series(period)
    future = pd.concat([b["water_level_m"].shift(-h) for h in range(1, horizon + 1)], axis=1)
    future_max = future.max(axis=1, skipna=False)                       # NaN if any future hour is missing
    target = (future_max >= high_water).astype(float).where(future_max.notna())
    same_period = pd.concat([period.shift(-h) for h in range(1, horizon + 1)], axis=1).eq(period, axis=0).all(axis=1)
    target = target.where(same_period)                                  # drop windows crossing a boundary
    keep = period.eq("test") & target.notna()
    return pd.DataFrame({"timestamp": b.loc[keep, "timestamp"].values, "actual": target[keep].astype(int).values})


frames, metrics = [], {}
for name, cfg in CATCHMENTS.items():
    # 1. LSTM test predictions: one row per test hour that had a complete 48-hour input window
    pred = pd.read_csv(HERE / cfg["pred"], parse_dates=["timestamp"])
    pred = pred.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # Safety check: the saved warnings must come from ONE threshold, and that threshold must agree
    # with the value printed in the LSTM notebook (printed to 4 decimals, so allow for rounding).
    # The saved warnings themselves are used as-is; nothing is re-thresholded here.
    lowest_alert = pred.loc[pred["predicted_risk"] == 1, "risk_probability"].min()
    highest_no_alert = pred.loc[pred["predicted_risk"] == 0, "risk_probability"].max()
    if not highest_no_alert < lowest_alert:
        raise ValueError(f"{name}: warnings are not consistent with a single threshold.")
    if not (highest_no_alert - 0.00005 <= cfg["threshold"] <= lowest_alert + 0.00005):
        raise ValueError(f"{name}: notebook threshold {cfg['threshold']} does not match the saved warnings "
                         f"(they imply a threshold between {highest_no_alert:.5f} and {lowest_alert:.5f}).")

    # 2. River level and rainfall for the same hours (display only)
    base = pd.read_csv(HERE / cfg["base"], parse_dates=["timestamp"])
    base = base.drop_duplicates("timestamp")[["timestamp", "water_level_m", "rainfall_mm"]]
    d = pred.merge(base, on="timestamp", how="left", validate="one_to_one")
    missing_level = int(d["water_level_m"].isna().sum())

    # 3. Hourly test metrics
    y, p, a = d["actual_risk"].astype(int), d["risk_probability"], d["predicted_risk"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y, a, labels=[0, 1]).ravel()

    # 4. Event-level and alert-episode metrics (gap-aware), counted the same way as the LSTM notebooks
    if cfg.get("full_timeline_events"):
        # Events on the full test timeline; hours without a prediction count as "no warning"
        full = full_test_target(pd.read_csv(HERE / cfg["base"], parse_dates=["timestamp"]), cfg["high_water"])
        full = full.merge(d[["timestamp", "actual_risk", "predicted_risk"]], on="timestamp", how="left")
        agree = full.dropna(subset=["actual_risk"])
        if not (agree["actual"] == agree["actual_risk"]).all():
            raise ValueError(f"{name}: rebuilt target does not match the LSTM notebook's labels.")
        full_warn = full["predicted_risk"].fillna(0).astype(int)
        events = runs(full["timestamp"], full["actual"])
        events_detected = int(sum(full_warn.iloc[s:e].any() for s, e in events))
    else:
        events = runs(d["timestamp"], y)
        events_detected = int(sum(a.iloc[s:e].any() for s, e in events))
    alerts = runs(d["timestamp"], a)
    false_alerts = int(sum(not y.iloc[s:e].any() for s, e in alerts))

    metrics[name] = dict(
        model="LSTM",
        river_variable="Water level", unit="m",
        risk_threshold=float(cfg["threshold"]),
        high_water_level=float(cfg["high_water"]),
        test_start=str(d["timestamp"].min()), test_end=str(d["timestamp"].max()),
        test_rows=int(len(d)),
        accuracy=float(accuracy_score(y, a)),
        precision=float(precision_score(y, a, zero_division=0)),
        recall=float(recall_score(y, a, zero_division=0)),
        f1=float(f1_score(y, a, zero_division=0)),
        f2=float(fbeta_score(y, a, beta=2, zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        events=len(events), events_detected=events_detected,
        alert_episodes=len(alerts), false_alert_episodes=false_alerts,
        inputs=cfg["inputs"],
        source=f"{cfg['pred']} (LSTM notebook) + {cfg['base']} (river level and rainfall)",
    )

    frames.append(pd.DataFrame({
        "catchment": name,
        "timestamp": d["timestamp"],
        "river_variable": "Water level",
        "river_value": d["water_level_m"],
        "rainfall_mm": d["rainfall_mm"],
        "risk_score": p,
        "alert": a,
        "actual_next24h": y,
    }))

    m = metrics[name]
    print(f"{name:9s} rows={m['test_rows']:6d}  thr={m['risk_threshold']:.4f}  P={m['precision']:.3f}  "
          f"R={m['recall']:.3f}  F2={m['f2']:.3f}  ROC={m['roc_auc']:.3f}  "
          f"events={m['events_detected']}/{m['events']}  alerts={m['alert_episodes']} "
          f"(false {m['false_alert_episodes']})  hours without river value={missing_level}")

pd.concat(frames, ignore_index=True).to_csv(HERE / "predictions.csv", index=False)
(HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("Saved predictions.csv and metrics.json")
