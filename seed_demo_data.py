"""
seed_demo_data.py
=================
Creates qc_demo.db with ~1 month of capacity data for dashboard / Power BI demos.

Rules:
  - Average ~700 panels checked per hour (during working hours)
  - 5 stations: LINE-1 .. LINE-5
  - Working hours only: 08:00 – 17:00
  - Friday is off
  - Scattered realistic variation

Usage:
    python seed_demo_data.py
    python seed_demo_data.py --csv     # also write qc_demo_panels.csv
"""

import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import qc_db

DB = Path(__file__).resolve().parent / "qc_demo.db"
CSV_OUT = Path(__file__).resolve().parent / "qc_demo_panels.csv"

DAYS = 30
# Internal target is higher so that after hour/day factors the
# observed average lands near 700 panels/hour.
TARGET_PER_HOUR = 780
WORK_START = 8
WORK_END = 17
FRIDAY_OFF = True
STATIONS = [
    ("LINE-1", 1.00),
    ("LINE-2", 0.98),
    ("LINE-3", 1.02),
    ("LINE-4", 0.97),
    ("LINE-5", 1.01),
]

BASE_GAP = 3600.0 / TARGET_PER_HOUR


def main(export_csv: bool = False):
    if DB.exists():
        DB.unlink()

    conn = qc_db.connect(DB)
    random.seed(42)

    today = datetime.now(qc_db.LOCAL_TZ).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    for station, speed in STATIONS:
        for days_back in range(DAYS - 1, -1, -1):
            day = today - timedelta(days=days_back)

            if FRIDAY_OFF and day.weekday() == 4:
                continue

            dow_factor = {
                0: 0.97, 1: 1.02, 2: 1.03, 3: 1.01,
                4: 0.95, 5: 0.92, 6: 0.90,
            }.get(day.weekday(), 1.0)

            day_noise = random.uniform(0.97, 1.03)

            t = day.replace(
                hour=WORK_START,
                minute=random.randint(0, 3),
                second=random.randint(0, 20),
            )
            end = day.replace(hour=WORK_END, minute=0, second=0)

            utc0, loc0 = qc_db.now_pair(t.astimezone(timezone.utc))
            sid = conn.execute(
                "INSERT INTO sessions(station_id, started_utc, started_local,"
                " ended_utc, source, note) VALUES (?,?,?,?,?,?)",
                (
                    station, utc0, loc0,
                    end.astimezone(timezone.utc).isoformat(),
                    "demo", "1-month capacity demo ~700/h",
                ),
            ).lastrowid

            last = None
            n = 0

            while t < end:
                hour = t.hour
                # Mild hour profile (keeps average near target)
                if hour == 8:
                    hf = 0.90
                elif hour == 9:
                    hf = 0.97
                elif hour in (10, 11):
                    hf = 1.06
                elif hour == 12:
                    hf = 0.93
                elif hour in (13, 14):
                    hf = 1.04
                elif hour == 15:
                    hf = 1.00
                else:
                    hf = 0.94

                target_gap = BASE_GAP / (speed * dow_factor * day_noise * hf)
                gap = max(2.0, random.gauss(target_gap, target_gap * 0.09))

                # Very rare short stoppages
                r = random.random()
                if r < 0.001:
                    gap += random.randint(15, 50)
                elif r < 0.0015:
                    gap += random.randint(50, 150)

                t += timedelta(seconds=gap)
                if t >= end:
                    break

                n += 1
                utc, loc = qc_db.now_pair(t.astimezone(timezone.utc))
                g = round((t - last).total_seconds(), 3) if last else None
                last = t
                transfer = round(max(1.2, random.gauss(3.8, 1.0)), 2)

                conn.execute(
                    "INSERT OR IGNORE INTO events("
                    " session_id, station_id, event_type,"
                    " event_utc, event_local, running_count,"
                    " transfer_time_s, gap_s)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (sid, station, "PANEL_CHECKED", utc, loc, n, transfer, g),
                )

    conn.commit()

    row_count = conn.execute(
        "SELECT COUNT(*) FROM v_panel_checked"
    ).fetchone()[0]
    unique_days = conn.execute(
        "SELECT COUNT(DISTINCT event_date) FROM v_panel_checked"
    ).fetchone()[0]

    hours_per_day = WORK_END - WORK_START
    n_stations = len(STATIONS)
    total_hours = unique_days * hours_per_day * n_stations
    avg = row_count / total_hours if total_hours else 0

    print("=" * 64)
    print("Fabricated capacity data written to:")
    print(f"  {DB}")
    print()
    print(f"  Stations                 : {n_stations}  ({', '.join(s for s, _ in STATIONS)})")
    print(f"  Working days (no Friday) : {unique_days}")
    print(f"  Hours per day            : {hours_per_day}  ({WORK_START}:00-{WORK_END}:00)")
    print(f"  Total production hours   : {total_hours}")
    print(f"  Total PANEL_CHECKED      : {row_count:,}")
    print(f"  Average panels / hour    : {avg:.0f}")
    print()
    print("Hourly sample (LINE-1, last 2 days):")
    rows = conn.execute(
        """
        SELECT event_date, hour_label, SUM(panels) AS cnt
        FROM v_panel_checked
        WHERE station_id = 'LINE-1'
          AND event_date >= date('now', '-2 days')
        GROUP BY event_date, hour_label
        ORDER BY event_date, hour_label
        """
    ).fetchall()
    for r in rows:
        print(f"  {r[0]}  {r[1]}  ->  {r[2]} panels")
    print("=" * 64)

    if export_csv:
        print()
        print(f"Writing CSV: {CSV_OUT}")
        cur = conn.execute(
            """
            SELECT event_id, session_id, station_id, event_local,
                   event_date, event_hour, hour_label, event_month,
                   weekday, running_count, transfer_time_s, gap_s, panels
            FROM v_panel_checked
            ORDER BY event_local
            """
        )
        cols = [d[0] for d in cur.description]
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            while True:
                batch = cur.fetchmany(50_000)
                if not batch:
                    break
                w.writerows(batch)
        print(f"  CSV rows written: {row_count:,}")
        print(f"  File: {CSV_OUT}")

    conn.close()
    print()
    print("Next steps:")
    print("  Streamlit :  set QC_DB=qc_demo.db   then   streamlit run dashboard.py")
    print("  Power BI  :  ODBC -> qc_demo.db   OR   Get Data -> Text/CSV -> qc_demo_panels.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true",
                        help="Also export qc_demo_panels.csv")
    args = parser.parse_args()
    main(export_csv=args.csv)
