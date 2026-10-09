# Panel QC – database + dashboard kit

Files
- qc_db.py            SQLite layer (tables, view `v_panel_checked`, writer used by the camera script)
- train_model.py      YOUR script with database logging added (diff: train_model_db.patch)
- import_csv.py       one-off import of the old inspection_events.csv
- dashboard.py        Streamlit dashboard (filters: station, date, time-of-day, weekday)
- seed_demo_data.py   makes qc_demo.db with fake shifts to try the dashboard
- powerbi_queries.sql queries for Power BI
- start_dashboard.bat shares the dashboard on your network (port 8501)

Quick start (Windows, in this folder)
    python -m pip install -r requirements.txt
    python import_csv.py                      # optional: bring in old CSV history
    python train_model.py --camera-fps 60 --process-width 640 --hand-detect-every 2 --station LINE-1
    streamlit run dashboard.py                # second terminal, while or after counting

Try with fake data first
    python seed_demo_data.py
    set QC_DB=qc_demo.db
    streamlit run dashboard.py
