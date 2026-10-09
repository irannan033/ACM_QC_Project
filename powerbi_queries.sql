-- Paste into Power BI "Get Data > ODBC > Advanced options > SQL statement"
-- (SQLite ODBC driver required, see README). Import mode, one query per table.

-- Fact table: one row per checked panel
SELECT event_id, session_id, station_id, event_local, event_date, event_hour,
       hour_label, event_month, weekday, running_count, transfer_time_s, gap_s, panels
FROM v_panel_checked;

-- Sessions (for uptime / utilisation)
SELECT session_id, station_id, started_local, started_utc, ended_utc, last_seen_utc
FROM sessions;
