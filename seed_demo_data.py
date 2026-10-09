"""
seed_demo_data.py - fills a SEPARATE demo database with fake shifts so you can
try the dashboard before real data exists.

    python seed_demo_data.py            # creates qc_demo.db
    set QC_DB=qc_demo.db                # Windows  (Linux/Mac: export QC_DB=qc_demo.db)
    streamlit run dashboard.py
"""
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import qc_db

DB = Path(__file__).resolve().parent / "qc_demo.db"
if DB.exists():
    DB.unlink()

conn = qc_db.connect(DB)
random.seed(7)
today = datetime.now(qc_db.LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0)

for station, base_pace in [("STATION-1", 17), ("STATION-2", 22)]:
    for days_back in range(14, -1, -1):
        day = today - timedelta(days=days_back)
        if day.weekday() == 4:           # Friday off
            continue
        for start_h, end_h in [(8, 12), (13, 17)]:   # two work blocks
            t = day.replace(hour=start_h, minute=random.randint(0, 10))
            end = day.replace(hour=end_h)
            utc0, loc0 = qc_db.now_pair(t.astimezone(timezone.utc))
            sid = conn.execute(
                "INSERT INTO sessions(station_id, started_utc, started_local, ended_utc,"
                " source, note) VALUES (?,?,?,?,?,?)",
                (station, utc0, loc0, end.astimezone(timezone.utc).isoformat(), "demo", "demo")
            ).lastrowid
            last, n = None, 0
            while t < end:
                pace = base_pace * (0.75 if t.hour in (11, 16) else 1.0)
                gap = max(6, random.gauss(pace, 4))
                if random.random() < 0.03:
                    gap += random.randint(120, 600)      # short stoppage
                t += timedelta(seconds=gap)
                if t >= end:
                    break
                n += 1
                utc, loc = qc_db.now_pair(t.astimezone(timezone.utc))
                g = round((t - last).total_seconds(), 3) if last else None
                last = t
                conn.execute(
                    "INSERT OR IGNORE INTO events(session_id, station_id, event_type,"
                    " event_utc, event_local, running_count, transfer_time_s, gap_s)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (sid, station, "PANEL_CHECKED", utc, loc, n,
                     round(random.uniform(2, 6), 2), g))
conn.commit()
print("Demo rows:", conn.execute("SELECT COUNT(*) FROM v_panel_checked").fetchone()[0], "->", DB)
