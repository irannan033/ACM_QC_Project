"""
import_csv.py - load your existing inspection_events.csv into the database.

    python import_csv.py                       # uses defaults
    python import_csv.py --csv inspection_events.csv --db qc_events.db --station STATION-1

Safe to run more than once (duplicates are ignored).

Your CSV has two row layouts (older rows have no 'event' column):
    old : inspection_count, wall_time_utc, video_time_seconds
    new : inspection_count, event, wall_time_utc, video_time_seconds
Both are handled. Old rows and SOURCE_TO_CHECKED rows become PANEL_CHECKED.
A new session is started whenever video time or count goes backwards
(= the program was restarted or the count was reset).
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import qc_db

HERE = Path(__file__).resolve().parent


def parse_row(row):
    row = [c.strip() for c in row]
    if len(row) == 3:
        count, wall, video = row
        event = "SOURCE_TO_CHECKED"
    elif len(row) == 4:
        count, event, wall, video = row
    else:
        return None
    when = datetime.fromisoformat(wall)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    event_type = {
        "SOURCE_TO_CHECKED": qc_db.EVENT_PANEL_CHECKED,
        "COUNT_RESET": qc_db.EVENT_COUNT_RESET,
    }.get(event, event)
    return int(count), event_type, when.astimezone(timezone.utc), float(video)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(HERE / "inspection_events.csv"))
    p.add_argument("--db", default=str(HERE / "qc_events.db"))
    p.add_argument("--station", default="STATION-1")
    args = p.parse_args()

    conn = qc_db.connect(args.db)
    inserted = skipped = 0
    session_id = None
    prev_video = prev_count = None
    last_checked = None

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for raw in reader:
            if not raw or not any(raw):
                continue
            parsed = parse_row(raw)
            if parsed is None:
                skipped += 1
                continue
            count, event_type, when, video = parsed
            utc, local = qc_db.now_pair(when)

            restarted = (
                session_id is None
                or (prev_video is not None and video < prev_video)
                or (event_type == qc_db.EVENT_PANEL_CHECKED
                    and prev_count is not None and count <= prev_count
                    and video < (prev_video or 0))
            )
            if restarted:
                row = conn.execute(
                    "SELECT session_id FROM sessions WHERE station_id=? AND"
                    " started_utc=? AND source='csv-import'",
                    (args.station, utc),
                ).fetchone()
                if row:                      # already imported before
                    session_id = row[0]
                else:
                    cur = conn.execute(
                        "INSERT INTO sessions(station_id, started_utc, started_local,"
                        " source, note) VALUES (?,?,?,?,?)",
                        (args.station, utc, local, "csv-import", "imported from CSV"),
                    )
                    session_id = cur.lastrowid
                last_checked = None
            prev_video, prev_count = video, count

            gap = None
            if event_type == qc_db.EVENT_PANEL_CHECKED:
                if last_checked is not None:
                    gap = round((when - last_checked).total_seconds(), 3)
                last_checked = when

            cur = conn.execute(
                "INSERT OR IGNORE INTO events(session_id, station_id, event_type,"
                " event_utc, event_local, video_time_s, running_count, gap_s)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (session_id, args.station, event_type, utc, local, video, count, gap),
            )
            inserted += cur.rowcount
            skipped += 1 - cur.rowcount

    # close imported sessions at their last event
    conn.execute(
        "UPDATE sessions SET ended_utc = (SELECT MAX(event_utc) FROM events e"
        " WHERE e.session_id = sessions.session_id), last_seen_utc = (SELECT MAX(event_utc)"
        " FROM events e WHERE e.session_id = sessions.session_id)"
        " WHERE source='csv-import' AND ended_utc IS NULL"
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM v_panel_checked").fetchone()[0]
    print(f"Inserted {inserted} rows, skipped {skipped}. "
          f"Panels checked in database now: {total}")
    conn.close()


if __name__ == "__main__":
    main()
