"""
qc_db.py - SQLite storage layer for the Panel QC counter
=========================================================

One file (qc_events.db) holds everything. Two tables:

  sessions : one row each time the camera program is started
  events   : one row per workflow event (panel checked, bundle accepted, ...)

The view  v_panel_checked  is what the dashboard / Power BI reads. It adds
date, hour, weekday columns so filtering by date / time / hour is trivial.

Times are stored twice:
  event_utc   - ISO-8601 UTC (unambiguous, good for audit)
  event_local - 'YYYY-MM-DD HH:MM:SS.mmm' in local factory time (Asia/Dhaka),
                used for all filtering and charts.
"""

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Dhaka")
except Exception:  # Windows without the 'tzdata' package
    LOCAL_TZ = timezone(timedelta(hours=6))  # Bangladesh has no DST

EVENT_PANEL_CHECKED = "PANEL_CHECKED"
EVENT_BUNDLE_ACCEPTED = "BUNDLE_ACCEPTED"
EVENT_RETURNED = "RETURNED_TO_BUNDLE"
EVENT_TIMEOUT = "TIMEOUT"
EVENT_COUNT_RESET = "COUNT_RESET"

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS sessions (
    session_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id    TEXT NOT NULL,
    started_utc   TEXT NOT NULL,
    started_local TEXT NOT NULL,
    last_seen_utc TEXT,
    ended_utc     TEXT,
    source        TEXT,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES sessions(session_id),
    station_id      TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    event_utc       TEXT NOT NULL,
    event_local     TEXT NOT NULL,
    video_time_s    REAL,
    running_count   INTEGER,
    transfer_time_s REAL,   -- bundle picked -> placed in checked area
    gap_s           REAL    -- seconds since previous PANEL_CHECKED
);

-- prevents duplicates when the same CSV is imported twice
CREATE UNIQUE INDEX IF NOT EXISTS ux_events_dedupe
    ON events(station_id, event_type, event_utc);
CREATE INDEX IF NOT EXISTS ix_events_type_local
    ON events(event_type, event_local);

DROP VIEW IF EXISTS v_panel_checked;
CREATE VIEW v_panel_checked AS
SELECT
    event_id,
    session_id,
    station_id,
    event_utc,
    event_local,
    date(event_local)                              AS event_date,
    CAST(strftime('%H', event_local) AS INTEGER)   AS event_hour,
    strftime('%H:00', event_local)                 AS hour_label,
    strftime('%Y-%m', event_local)                 AS event_month,
    CASE strftime('%w', event_local)
        WHEN '0' THEN 'Sun' WHEN '1' THEN 'Mon' WHEN '2' THEN 'Tue'
        WHEN '3' THEN 'Wed' WHEN '4' THEN 'Thu' WHEN '5' THEN 'Fri'
        ELSE 'Sat' END                             AS weekday,
    running_count,
    transfer_time_s,
    gap_s,
    1                                              AS panels
FROM events
WHERE event_type = 'PANEL_CHECKED';
"""


def now_pair(when_utc=None):
    """Return (utc_iso, local_string) for 'now' or a given aware UTC datetime."""
    when_utc = when_utc or datetime.now(timezone.utc)
    local = when_utc.astimezone(LOCAL_TZ)
    return (
        when_utc.isoformat(),
        local.strftime("%Y-%m-%d %H:%M:%S.") + f"{local.microsecond // 1000:03d}",
    )


def connect(path):
    conn = sqlite3.connect(str(path), timeout=15, check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


class QCDatabase:
    """Used by the camera program. Never raises into the video loop."""

    def __init__(self, path, station_id="STATION-1", source=""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.station_id = station_id
        self.source = source
        self.conn = connect(self.path)
        self.session_id = None
        self._last_checked_utc = None
        self._last_heartbeat = 0.0

    # -- sessions ---------------------------------------------------------
    def start_session(self, note=""):
        utc, local = now_pair()
        cur = self.conn.execute(
            "INSERT INTO sessions(station_id, started_utc, started_local,"
            " last_seen_utc, source, note) VALUES (?,?,?,?,?,?)",
            (self.station_id, utc, local, utc, self.source, note),
        )
        self.conn.commit()
        self.session_id = cur.lastrowid
        return self.session_id

    def heartbeat(self, every_seconds=30):
        """Marks the program as alive. Lets the dashboard measure uptime."""
        if self.session_id is None:
            return
        if time.monotonic() - self._last_heartbeat < every_seconds:
            return
        self._last_heartbeat = time.monotonic()
        try:
            self.conn.execute(
                "UPDATE sessions SET last_seen_utc=? WHERE session_id=?",
                (datetime.now(timezone.utc).isoformat(), self.session_id),
            )
            self.conn.commit()
        except sqlite3.Error as error:
            print(f"[db] heartbeat failed: {error}")

    def end_session(self):
        if self.session_id is None:
            return
        try:
            utc = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                "UPDATE sessions SET ended_utc=?, last_seen_utc=? WHERE session_id=?",
                (utc, utc, self.session_id),
            )
            self.conn.commit()
        except sqlite3.Error as error:
            print(f"[db] end_session failed: {error}")

    # -- events -----------------------------------------------------------
    def log_event(self, event_type, running_count=None, video_time_s=None,
                  transfer_time_s=None):
        now = datetime.now(timezone.utc)
        gap = None
        if event_type == EVENT_PANEL_CHECKED:
            if self._last_checked_utc is not None:
                gap = round((now - self._last_checked_utc).total_seconds(), 3)
            self._last_checked_utc = now
        utc, local = now_pair(now)
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO events(session_id, station_id, event_type,"
                " event_utc, event_local, video_time_s, running_count,"
                " transfer_time_s, gap_s) VALUES (?,?,?,?,?,?,?,?,?)",
                (self.session_id, self.station_id, event_type, utc, local,
                 video_time_s, running_count, transfer_time_s, gap),
            )
            self.conn.commit()
            return True
        except sqlite3.Error as error:
            print(f"[db] could not write {event_type}: {error}")
            return False

    def close(self):
        self.end_session()
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
