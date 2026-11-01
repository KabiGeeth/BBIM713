# Catchment High-Water Early Warning dashboard (24-hour lead time)
The Flood Managers · BBIM713 Capstone · Mahurangi, Otara, Puhinui, Swanson, Wairau


## Project links
- **Team portfolio:** https://aucklandfloodprediction.wordpress.com/
- **Live dashboard:** https://the-flood-managers-dashboard.streamlit.app/

## Run it
    pip install -r requirements.txt
    streamlit run app.py

## Pages
1. **Home** – project purpose, workflow, team members, supervisors and data parameters.
2. **Upload your dataset** – upload a CSV or Excel file and train a session-only Random Forest using a chronological 70/15/15 split.
3. **All catchments** – view the latest or selected-time status of all five catchments side by side.
4. **Model information** – view the selected LSTM metrics, confusion matrix, event results and six-model comparison.
5. **Risk monitor** – select a catchment and historical test-period time to view its high-water-risk status, water level, rainfall, model risk score and trend chart.
6. **About & limitations** – explanation of the warning status, statistical high-water definition, limitations and future development.

## Wording rules (from the client brief)
Use *Normal*, *High-water risk within the next 24 hours*, *Early warning*, *Model risk score*.
Never say "confirmed flood", "a flood will occur" or "exact forecast". The model classifies a percentile-defined high-water
condition; it does not predict an exact future water level, and the risk score is not a calibrated probability.

## Data files (the app reads only these two)
- `predictions.csv` – one row per hour of the held-out TEST period for each catchment:
  `catchment, timestamp, river_variable, river_value, rainfall_mm, risk_score, alert, actual_next24h`
- `metrics.json` – per-catchment LSTM warning threshold, test metrics, confusion matrix, event and alert-episode counts.

Both files are created by `build_dashboard_data.py` from the test predictions saved by the LSTM notebooks
(`<catchment>_lstm_test_predictions.csv`), plus the cleaned hourly data each LSTM notebook used (for the
river level and rainfall shown on the charts). Nothing is retrained: the dashboard metrics match the LSTM notebooks.
To rebuild: put the input files next to the script and run `python build_dashboard_data.py`.

## Notes
- Puhinui high-water level is set to 0.865 m (value in the team's Puhinui notebook).
- All five catchments use river water level (m) to define the high-water target.
- Event definition: an event is a run of consecutive hours with high water ahead; an alert episode is a run of consecutive warning hours.
- Selected dashboard model: LSTM for all five catchments. LSTM was selected as a common deployment model because its performance was competitive across catchments and it provided a consistent sequence-based approach. It was not necessarily the highest-scoring model for every individual catchment.
- Model-comparison note: KNN used a separately prepared set of eligible observations, while LSTM and GRU used 48-hour sequences. The six-model results support broad benchmarking but are not a strict row-for-row comparison.

## Data access
`predictions.csv` and `metrics.json` contain AUT-supplied telemetry. AUT granted permission for these project data files to remain publicly accessible until the end of the project. After project completion, the team will remove or restrict the files in accordance with AUT's instructions.
