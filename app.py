"""
Catchment High-Water Early Warning dashboard (24-hour lead time)
The Flood Managers - BBIM713 Capstone

Run:  streamlit run app.py
Reads: data/predictions.csv and data/metrics.json  (created by prepare_data.py)

Wording rule: the model predicts a percentile-defined HIGH-WATER RISK within the next 24 hours.
It does not forecast an exact water level and does not confirm a flood.
"""
import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Catchment High-Water Early Warning", page_icon="🌊", layout="wide")

DATA = Path(__file__).parent / "data"
GREEN, AMBER, RED = "#2e7d32", "#e07a1f", "#c1121f"
BLUE, GREY, RAIN = "#2a6f97", "#6b7280", "#7aa6c2"

DISCLAIMER = ("High-water-risk conditions are predicted within the next 24 hours. This is an early-warning "
              "classification based on historical sensor patterns and is not a confirmed flood declaration.")

# Test-set results from the team notebooks (all six models) - used only for the comparison table
MODEL_COMPARISON = {
    "F₂": {
        "Random Forest": [0.341, 0.345, 0.316, 0.314, 0.288],
        "XGBoost":       [None, 0.339, 0.241, 0.318, 0.309],
        "Naive Bayes":   [0.333, 0.277, 0.305, 0.338, 0.331],
        "KNN":           [0.318, 0.372, 0.321, 0.276, 0.315],
        "LSTM":          [0.362, 0.350, 0.340, 0.321, 0.314],
        "GRU":           [0.370, 0.338, 0.337, 0.301, None],
    },
    "ROC-AUC": {
        "Random Forest": [0.794, 0.773, 0.739, 0.830, 0.679],
        "XGBoost":       [None, 0.764, 0.721, 0.811, 0.714],
        "Naive Bayes":   [0.792, 0.763, 0.726, 0.803, 0.731],
        "KNN":           [0.775, 0.750, 0.731, 0.769, 0.684],
        "LSTM":          [0.814, 0.768, 0.725, 0.803, 0.733],
        "GRU":           [0.821, 0.768, 0.725, 0.799, None],
    },
}
CATCHMENT_ORDER = ["Mahurangi", "Otara", "Puhinui", "Swanson", "Wairau"]


# ----------------------------------------------------------------------------- data
def find_file(name):
    """Find a data file next to app.py, in a data/ folder, or anywhere below the app folder."""
    here = Path(__file__).parent
    for p in (here / "data" / name, here / name):
        if p.exists():
            return p
    hits = sorted(here.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


@st.cache_data
def load():
    pred_path, metrics_path = find_file("predictions.csv"), find_file("metrics.json")
    if pred_path is None or metrics_path is None:
        return None, None
    pred = pd.read_csv(pred_path, parse_dates=["timestamp"])
    pred = pred.sort_values(["catchment", "timestamp"]).reset_index(drop=True)
    return pred, json.loads(metrics_path.read_text())


pred_all, metrics_all = load()
if pred_all is None:
    here = Path(__file__).parent
    st.error("The dashboard data files were not found: **predictions.csv** and **metrics.json** must be in the same "
             "GitHub repository as app.py (in a folder called `data`, or next to app.py).")
    st.write("Files the app can see in the repository:")
    st.code("\n".join(sorted(str(p.relative_to(here)) for p in here.rglob("*")
                              if p.is_file() and ".git" not in p.parts)[:60]) or "(none)")
    st.stop()


def catchment_frame(name):
    d = pred_all[pred_all.catchment == name].reset_index(drop=True)
    d["rain_24h"] = d["rainfall_mm"].rolling(24, min_periods=1).sum()
    return d


def nearest_row(d, ts):
    """Row at or just before ts (falls back to the first row)."""
    i = d["timestamp"].searchsorted(ts, side="right") - 1
    return d.iloc[max(i, 0)]


def fmt_value(v, m):
    return f"{v:.2f} {m['unit']}" if m["unit"] == "m" else f"{v:.2f} {m['unit']}"


def status_card(alert, title_normal="NORMAL", title_high="HIGH-WATER RISK"):
    if alert:
        color, icon, title = RED, "▲", title_high
        text = "Early warning: high-water risk flagged for the next 24 hours."
    else:
        color, icon, title = GREEN, "✔", title_normal
        text = "No high-water risk flagged for the next 24 hours."
    st.markdown(
        f"""<div style="border-left:10px solid {color};padding:14px 18px;border-radius:8px;
        background:{color}1f;margin-bottom:6px">
        <div style="font-size:1.6rem;font-weight:700;color:{color}">{icon} {title}</div>
        <div style="font-size:1rem">{text}</div></div>""",
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🌊 High-Water Early Warning")
st.sidebar.caption("24-hour lead time · Auckland catchments")
page = st.sidebar.radio("Page", ["Risk monitor", "All catchments", "Model information", "About & limitations"],
                        label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Demonstration on the held-out test period: each time shown is replayed as if it were 'now', "
                   "using only information available up to that hour.")


# ============================================================================= PAGE 1
def page_monitor():
    st.title("Risk monitor")
    st.caption("Is there a high-water-risk warning for the selected catchment in the next 24 hours?")

    c1, c2, c3 = st.columns([1.1, 1, 1])
    catchment = c1.selectbox("Catchment", CATCHMENT_ORDER, key="catchment")
    d = catchment_frame(catchment)
    m = metrics_all[catchment]
    tmin, tmax = d.timestamp.min(), d.timestamp.max()

    # --- state: reset the selected time when the catchment changes
    default_ts = d.loc[d.risk_score.idxmax(), "timestamp"]     # highest-risk hour in the test period
    if st.session_state.get("_c") != catchment:
        st.session_state["_c"] = catchment
        st.session_state["d"], st.session_state["h"] = default_ts.date(), default_ts.hour

    def set_ts(ts):
        st.session_state["d"], st.session_state["h"] = ts.date(), ts.hour

    def jump_latest():
        set_ts(tmax)

    def jump_alert(direction):
        cur = pd.Timestamp(st.session_state["d"]) + pd.Timedelta(hours=st.session_state["h"])
        # alerts that start a new warning period (first hour of each run of alert hours)
        a = d.alert.values; starts = d.timestamp[(a == 1) & (np.r_[0, a[:-1]] == 0)]
        pick = starts[starts > cur].iloc[:1] if direction > 0 else starts[starts < cur].iloc[-1:]
        if len(pick):
            set_ts(pick.iloc[0])

    sel_date = c2.date_input("Date", min_value=tmin.date(), max_value=tmax.date(), key="d")
    sel_hour = c3.selectbox("Hour", list(range(24)), format_func=lambda h: f"{h:02d}:00", key="h")
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    b1.button("⏮ Previous warning", on_click=jump_alert, args=(-1,), width="stretch")
    b2.button("Next warning ⏭", on_click=jump_alert, args=(1,), width="stretch")
    b3.button("Latest observation", on_click=jump_latest, width="stretch")

    sel_ts = pd.Timestamp.combine(sel_date, time(sel_hour))
    row = nearest_row(d, sel_ts)
    ts = row.timestamp
    if abs((ts - sel_ts).total_seconds()) > 3600:
        st.info(f"No observation at {sel_ts:%d %b %Y %H:00}; showing the closest earlier hour, {ts:%d %b %Y %H:00}.")

    st.markdown(f"#### {catchment} · {ts:%A %d %B %Y, %H:00}")
    left, right = st.columns([1.15, 2])
    with left:
        status_card(bool(row.alert))
        k1, k2 = st.columns(2)
        k1.metric(f"Current {m['river_variable'].lower()}", fmt_value(row.river_value, m))
        k2.metric("High-water threshold", f'{m["high_water_level"]:.3f} {m["unit"]}',
                  help="99.5th percentile of the training-period record. A statistical proxy, not an official flood level.")
        k3, k4 = st.columns(2)
        k3.metric("Rainfall this hour", f"{row.rainfall_mm:.1f} mm")
        k4.metric("Rainfall last 24 h", f"{row.rain_24h:.1f} mm")
        k5, k6 = st.columns(2)
        k5.metric("Model risk score", f"{row.risk_score:.2f}",
                  help="Random Forest output between 0 and 1. It ranks risk; it is not a calibrated chance of an event.")
        k6.metric("Warning threshold", f"{m['risk_threshold']:.2f}",
                  help="A warning is raised when the risk score reaches this value. Tuned on validation data to favour catching events (F₂).")
        st.progress(float(min(row.risk_score / max(m["risk_threshold"] * 3, 1e-9), 1.0)),
                    text=f"Score {row.risk_score:.2f} vs warning threshold {m['risk_threshold']:.2f}")
        if row.alert:
            st.warning(DISCLAIMER)
        # retrospective check (possible because this is historical test data)
        outcome = "reached" if row.actual_next24h == 1 else "was not reached"
        verdict = ("✔ consistent with the warning" if row.alert == row.actual_next24h
                   else ("✖ false alarm" if row.alert else "✖ missed event"))
        st.caption(f"Retrospective check (historical data only): the high-water threshold {outcome} in the following 24 hours — {verdict}.")

    with right:
        win = st.radio("Time window", ["3 days", "7 days", "30 days"], index=1, horizontal=True)
        days = {"3 days": 3, "7 days": 7, "30 days": 30}[win]
        w = d[(d.timestamp >= ts - pd.Timedelta(days=days)) & (d.timestamp <= ts + pd.Timedelta(hours=24))]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                            row_heights=[0.52, 0.09, 0.09, 0.30])
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.river_value, mode="lines", name=m["river_variable"],
                                 line=dict(color=BLUE, width=1.8),
                                 hovertemplate=f"%{{x|%d %b %H:00}}<br>{m['river_variable']}: %{{y:.2f}} {m['unit']}<extra></extra>"), 1, 1)
        fig.add_hline(y=m["high_water_level"], line=dict(color=RED, dash="dash", width=1.4), row=1, col=1,
                      annotation_text="High-water threshold", annotation_position="top left",
                      annotation_font=dict(color=RED, size=11))
        for r, (col, sub, name, colr) in enumerate([("actual_next24h", w[w.actual_next24h == 1], "Actual high-risk period", GREY),
                                                    ("alert", w[w.alert == 1], "Model-predicted high-risk period", AMBER)], start=2):
            fig.add_trace(go.Bar(x=sub.timestamp, y=[1] * len(sub), width=3600000, marker_color=colr, name=name,
                                 hovertemplate=f"%{{x|%d %b %H:00}}<br>{name}<extra></extra>"), r, 1)
            fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, row=r, col=1)
        fig.add_trace(go.Bar(x=w.timestamp, y=w.rainfall_mm, width=3600000, marker_color=RAIN, name="Rainfall",
                             hovertemplate="%{x|%d %b %H:00}<br>Rainfall: %{y:.1f} mm<extra></extra>"), 4, 1)
        # selected prediction point and the 24 h it refers to
        fig.add_vrect(x0=ts, x1=ts + pd.Timedelta(hours=24), fillcolor=AMBER, opacity=0.10, line_width=0)
        fig.add_vline(x=ts, line=dict(color="#111", width=1.5, dash="dot"))
        fig.add_annotation(x=ts, y=1.0, yref="y domain", text="Selected time", showarrow=False, xanchor="right",
                           xshift=-4, yshift=-2, font=dict(size=11), row=1, col=1)
        fig.update_yaxes(title_text=f"{m['river_variable']} ({m['unit']})", row=1, col=1)
        fig.update_yaxes(title_text="Rain (mm/h)", row=4, col=1)
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, bargap=0, hovermode="x unified")
        for r, lab in ((2, "Actual"), (3, "Predicted")):
            fig.add_annotation(xref="paper", yref=f"y{r} domain", x=0, y=0.5, text=lab, showarrow=False, xanchor="right",
                               xshift=-6, font=dict(size=10, color=GREY if r == 2 else AMBER))
        st.plotly_chart(fig, width="stretch", theme="streamlit")
        st.caption("Top: river level with the high-water threshold. Middle strips: hours where high water occurred within the next 24 h "
                   "(**Actual**, grey) and hours where the model raised a warning (**Predicted**, orange). Bottom: rainfall. "
                   "The shaded block is the 24 hours the selected-time prediction refers to.")


# ============================================================================= PAGE 2
def page_overview():
    st.title("All catchments")
    st.caption("Latest status of each catchment at the selected time.")
    tmax = pred_all.timestamp.max()
    mode = st.radio("Show", ["Latest observation", "Choose a time"], horizontal=True)
    if mode == "Choose a time":
        c1, c2 = st.columns(2)
        dd = c1.date_input("Date", value=tmax.date(), min_value=pred_all.timestamp.min().date(), max_value=tmax.date(), key="od")
        hh = c2.selectbox("Hour", list(range(24)), index=0, format_func=lambda h: f"{h:02d}:00", key="oh")
        at = pd.Timestamp.combine(dd, time(hh))
    else:
        at = tmax
    rows, cols = [], st.columns(5)
    for col, name in zip(cols, CATCHMENT_ORDER):
        d = catchment_frame(name); m = metrics_all[name]; r = nearest_row(d, at)
        rows.append(dict(Catchment=name, Status="High-water risk" if r.alert else "Normal",
                         Level=f"{r.river_value:.2f} {m['unit']}", Variable=m["river_variable"],
                         **{"Risk score": f"{r.risk_score:.2f}", "Warning threshold": f"{m['risk_threshold']:.2f}",
                            "Rain 24 h (mm)": f"{r.rain_24h:.1f}", "Last update": f"{r.timestamp:%d %b %Y %H:00}"}))
        color = RED if r.alert else GREEN
        col.markdown(f"""<div style="border-top:8px solid {color};padding:10px 12px;border-radius:8px;background:{color}1a">
            <div style="font-weight:700;font-size:1.05rem">{name}</div>
            <div style="color:{color};font-weight:700">{'▲ HIGH-WATER RISK' if r.alert else '✔ NORMAL'}</div>
            <div style="font-size:.9rem">{m['river_variable']}: {r.river_value:.2f} {m['unit']}</div>
            <div style="font-size:.9rem">Risk score: {r.risk_score:.2f} (threshold {m['risk_threshold']:.2f})</div>
            <div style="font-size:.8rem;opacity:.75">{r.timestamp:%d %b %Y %H:00}</div></div>""", unsafe_allow_html=True)
    st.markdown("")
    st.dataframe(pd.DataFrame(rows).set_index("Catchment"), width="stretch")
    st.caption("Status refers to the next 24 hours. Rivers are measured differently: Mahurangi, Otara, Puhinui and Swanson use "
               "water level (m); Wairau uses discharge (m³/s). A catchment map is planned as future work.")


# ============================================================================= PAGE 3
def page_model():
    st.title("Model information")
    st.caption("Why the warning is trusted, and how often it is wrong. Held-out test period only.")
    catchment = st.selectbox("Catchment", CATCHMENT_ORDER, key="mcatchment")
    m = metrics_all[catchment]
    st.markdown(f"**Selected model:** {m['model']}  ·  **Warning threshold:** {m['risk_threshold']:.3f}  ·  "
                f"**Test period:** {m['test_start'][:10]} to {m['test_end'][:10]} ({m['test_rows']:,} hours)")
    cols = st.columns(5)
    for c, (lab, key, tip) in zip(cols, [("Precision", "precision", "Share of warning hours that were followed by high water."),
                                         ("Recall", "recall", "Share of high-water hours that were warned about."),
                                         ("F₂ score", "f2", "Balances precision and recall, counting recall twice as much."),
                                         ("PR-AUC", "pr_auc", "Area under the precision-recall curve."),
                                         ("ROC-AUC", "roc_auc", "Ability to rank risky hours above safe hours (0.5 = chance).")]):
        c.metric(lab, f"{m[key]:.3f}", help=tip)

    a, b = st.columns([1, 1])
    with a:
        z = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        share = z / z.sum(axis=1, keepdims=True)
        fig = go.Figure(go.Heatmap(z=share, x=["No warning", "Warning"], y=["No high water", "High water"],
                                   text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}", colorscale="Blues",
                                   zmin=0, zmax=1.3, showscale=False, hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed", title="Actual (next 24 h)"); fig.update_xaxes(title="Model")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Confusion matrix (hours)")
        st.plotly_chart(fig, width="stretch", theme="streamlit")
    with b:
        st.markdown("##### Events and alerts")
        e1, e2 = st.columns(2)
        e1.metric("High-water events detected", f"{m['events_detected']} of {m['events']}",
                  help="An event is a run of consecutive hours with high water ahead. Detected = at least one warning hour inside it.")
        e2.metric("False-alert episodes", f"{m['false_alert_episodes']} of {m['alert_episodes']}",
                  help="An alert episode is a run of consecutive warning hours. False = no high water followed it.")
        st.markdown(
            f"- **{m['tp']:,}** warning hours were followed by high water; **{m['fp']:,}** were false alarms.\n"
            f"- **{m['fn']:,}** high-water hours had no warning.\n"
            "- The threshold favours catching events, so false alarms are expected. Treat a warning as a prompt to check conditions.")
    st.markdown("##### Why this model")
    st.info("Random Forest is used for all five catchments so the warning is produced by one consistent, transparent pipeline. "
            "It ranks risk reasonably well (ROC-AUC 0.69–0.79 across catchments here). In the team notebooks the six models score "
            "very similarly (test F₂ 0.24–0.37) and no model is best in every catchment, and only 13–31 high-water events exist per "
            "test period, so no model is claimed to be best. The model choice is a working default until the team confirms a final selection rule.")
    st.markdown("##### All six models, test results from the team notebooks")
    tabs = st.tabs(list(MODEL_COMPARISON))
    for tab, metric in zip(tabs, MODEL_COMPARISON):
        df = pd.DataFrame(MODEL_COMPARISON[metric], index=CATCHMENT_ORDER).T
        z = df.astype(float).values
        text = [["n/a" if np.isnan(v) else f"{v:.3f}" for v in row] for row in z]
        fig = go.Figure(go.Heatmap(z=z, x=CATCHMENT_ORDER, y=list(df.index), text=text, texttemplate="%{text}",
                                   colorscale="Blues", showscale=False, hoverinfo="skip", xgap=2, ygap=2))
        fig.update_yaxes(autorange="reversed"); fig.update_xaxes(side="top")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
        tab.plotly_chart(fig, width="stretch", theme="streamlit")
    st.caption("n/a: XGBoost was not evaluated on the test set for Mahurangi, and the GRU was not run for Wairau. "
               "The table comes from the team notebooks. The metrics above come from this dashboard\u2019s own Random Forest run on the same test periods, "
               "so the two can differ slightly (for example Swanson, where the notebook and dashboard data versions differ).")


# ============================================================================= PAGE 4
def page_about():
    st.title("About & limitations")
    st.markdown("""
**What this dashboard shows.** For the selected catchment and time, whether the model flags **high-water risk within the next 24 hours**,
with the model risk score, the warning threshold, recent river and rainfall conditions, and a history-versus-prediction trend chart.

**How to read the status.**
- ✔ **Normal**: no high-water risk is flagged for the next 24 hours.
- ▲ **High-water risk**: an early warning. It means river and rainfall patterns resemble past hours that were followed by high water.

**What "high water" means.** It is defined statistically as the river reaching the **99.5th percentile of the training-period record**
(a water level in metres, or discharge in m³/s for Wairau). It is a proxy, not an official flood level or a confirmed flood.

**Limitations.**
- The model **classifies** risk; it does **not** forecast an exact future water level.
- The risk score is a ranking score, **not a calibrated probability**. Do not read 0.30 as "a 30% chance".
- Warnings are tuned to catch events, so **many warnings are false alarms** (see Model information).
- Results shown are a **replay of the held-out test period**, not live data. Only one lead time (24 h) is provided.
- Few high-water events exist in each test period, so accuracy figures are uncertain.

**Future development.** Live sensor integration, automated notifications, additional lead times (6 and 12 h), and a catchment map.

**Data.** River and rainfall telemetry supplied by AUT for Mahurangi, Otara, Puhinui, Swanson and Wairau, 2015–2025.
""")
    st.warning(DISCLAIMER)


{"Risk monitor": page_monitor, "All catchments": page_overview,
 "Model information": page_model, "About & limitations": page_about}[page]()
