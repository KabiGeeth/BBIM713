# Catchment High-Water Early Warning dashboard (24-hour lead time)
The Flood Managers · BBIM713 Capstone · Mahurangi, Otara, Puhinui, Swanson, Wairau

## Run it
    pip install -r requirements.txt
    streamlit run app.py

## Pages
1. **Risk monitor** – catchment + date/hour selector, Normal / High-water-risk status, current level or discharge, rainfall,
   model risk score vs warning threshold, trend chart (level + threshold, actual vs predicted high-risk strips, rainfall, selected 24 h).
2. **All catchments** – latest (or chosen-time) status of all five catchments side by side.
3. **Model information** – selected model, precision / recall / F2 / PR-AUC / ROC-AUC, confusion matrix, events detected,
   false-alert episodes, why the model was chosen, all six models' results from the team notebooks.
4. **About & limitations** – what the status means and what the model cannot do.

## Wording rules (from the client brief)
Use *Normal*, *High-water risk within the next 24 hours*, *Early warning*, *Model risk score*.
Never say "confirmed flood", "a flood will occur" or "exact forecast". The model classifies a percentile-defined high-water
condition; it does not predict an exact future water level, and the risk score is not a calibrated probability.

## Data files (the app reads only these two)
- `data/predictions.csv` – one row per hour of the held-out TEST period:
  `catchment, timestamp, river_variable, river_value, rainfall_mm, risk_score, alert, actual_next24h`
- `data/metrics.json` – per-catchment threshold, test metrics, confusion matrix, event counts.

`prepare_data.py` rebuilds both from the team's modelling CSVs (Random Forest, F2 threshold tuned on validation, test used once).
To use the team's own per-hour predictions instead, export a CSV with the same columns and replace `data/predictions.csv`
(then update `data/metrics.json`).

## Notes
- Puhinui high-water level is set to 0.865 m (value in the team's Puhinui notebook).
- Wairau uses discharge (m³/s); the other four use water level (m).
- Event definition: an event is a run of consecutive hours with high water ahead; an alert episode is a run of consecutive warning hours.
- Selected model = Random Forest for all catchments (working default). Change `model` in `metrics.json` / update the text on the Model page if the team agrees a different rule.

## IMPORTANT – data
`data/` contains AUT-supplied telemetry. Do NOT push it to a public GitHub repository; delete and return it to AUT at submission.
