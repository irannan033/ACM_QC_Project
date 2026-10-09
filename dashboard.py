"""
dashboard.py - Panel QC capacity dashboard (Streamlit)

Run:        streamlit run dashboard.py
Share (LAN): streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
Other DB:   set QC_DB=C:\\path\\to\\qc_events.db   (Linux/Mac: export QC_DB=...)
"""

import os
import sqlite3
from datetime import time, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = os.environ.get("QC_DB", str(Path(__file__).resolve().parent / "qc_events.db"))

st.set_page_config(page_title="Panel QC Dashboard", page_icon="✅", layout="wide")


# ---------------------------------------------------------------- data ----
@st.cache_data(ttl=15, show_spinner=False)
def load_data(db_path):
    con = sqlite3.connect(db_path, timeout=15)
    try:
        con.execute("PRAGMA query_only = ON")
        events = pd.read_sql_query(
            "SELECT * FROM v_panel_checked ORDER BY event_local", con)
        sessions = pd.read_sql_query("SELECT * FROM sessions", con)
    finally:
        con.close()
    events["event_local"] = pd.to_datetime(events["event_local"])
    for col in ("started_local",):
        sessions[col] = pd.to_datetime(sessions[col])
    for col in ("started_utc", "last_seen_utc", "ended_utc"):
        sessions[col] = pd.to_datetime(sessions[col], utc=True, errors="coerce")
    return events, sessions


if not Path(DB_PATH).exists():
    st.error(f"Database not found: {DB_PATH}\n\nRun the camera program once, or "
             "`python import_csv.py` to import your old CSV.")
    st.stop()

events, sessions = load_data(DB_PATH)
st.title("Panel QC Inspection Dashboard")

if events.empty:
    st.info("The database exists but has no checked panels yet.")
    st.stop()

# ------------------------------------------------------------- filters ----
with st.sidebar:
    st.header("Filters")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    stations = sorted(events["station_id"].unique())
    chosen_stations = st.multiselect("Station", stations, default=stations)

    dmin, dmax = events["event_local"].dt.date.min(), events["event_local"].dt.date.max()
    picked = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    d_start, d_end = (picked if isinstance(picked, (tuple, list)) and len(picked) == 2
                      else (picked[0] if isinstance(picked, (tuple, list)) else picked,) * 2)

    t_start, t_end = st.slider(
        "Time of day (applied to every day)",
        value=(time(0, 0), time(23, 59)), step=timedelta(minutes=15), format="HH:mm")

    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    chosen_days = st.multiselect("Weekday", weekdays, default=weekdays)

    bucket_label = st.radio("Chart resolution", ["5 min", "15 min", "Hour", "Day"], index=2)
    bucket = {"5 min": "5min", "15 min": "15min", "Hour": "60min", "Day": "D"}[bucket_label]

    break_s = st.number_input(
        "Ignore gaps longer than (seconds) in cycle-time stats", 30, 7200, 300, step=30,
        help="A gap longer than this is treated as a break/stoppage, not as working pace.")

minutes = events["event_local"].dt.hour * 60 + events["event_local"].dt.minute
mask = (
    events["station_id"].isin(chosen_stations)
    & (events["event_local"].dt.date >= d_start)
    & (events["event_local"].dt.date <= d_end)
    & (minutes >= t_start.hour * 60 + t_start.minute)
    & (minutes <= t_end.hour * 60 + t_end.minute)
    & events["weekday"].isin(chosen_days)
)
df = events[mask].copy()

st.caption(f"Showing **{d_start} → {d_end}**, **{t_start:%H:%M}–{t_end:%H:%M}**, "
           f"{', '.join(chosen_stations) or 'no station'}  •  database: `{DB_PATH}`")

if df.empty:
    st.warning("No panels match the current filters.")
    st.stop()

# ----------------------------------------------------------------- KPIs ----
total = len(df)
active_hours = df.groupby(["event_date", "event_hour", "station_id"]).ngroups
per_active_hour = total / active_hours if active_hours else 0
good_gaps = df["gap_s"].dropna()
good_gaps = good_gaps[good_gaps <= break_s]
pace = 3600 / good_gaps.mean() if len(good_gaps) else float("nan")
best_hour = df.groupby("event_hour").size()
by_day = df.groupby("event_date").size()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Panels checked", f"{total:,}")
c2.metric("Avg per active hour", f"{per_active_hour:.1f}",
          help="Total ÷ number of (day, hour, station) slots that had at least one panel.")
c3.metric("Pace while working", "–" if pd.isna(pace) else f"{pace:.0f} /h",
          help="3600 ÷ average gap between consecutive panels (gaps above the break limit ignored).")
c4.metric("Median time between panels",
          "–" if good_gaps.empty else f"{good_gaps.median():.0f} s")
c5.metric("Busiest hour of day", f"{best_hour.idxmax():02d}:00", f"{best_hour.max()} panels",
          delta_color="off")

c6, c7, c8, c9 = st.columns(4)
c6.metric("Days with data", f"{by_day.size}")
c7.metric("Avg per day", f"{by_day.mean():.0f}")
c8.metric("Best day", pd.to_datetime(by_day.idxmax()).strftime("%a %d %b"), f"{by_day.max()} panels", delta_color="off")
tt = df["transfer_time_s"].dropna()
c9.metric("Avg pick → checked time", "–" if tt.empty else f"{tt.mean():.1f} s")

tab_time, tab_hour, tab_heat, tab_cycle, tab_data, tab_up = st.tabs(
    ["Throughput over time", "Hour-of-day profile", "Heatmap", "Cycle time",
     "Event table", "Station uptime"])

# --------------------------------------------------------------- charts ----
with tab_time:
    ts = (df.set_index("event_local").groupby("station_id")
            .resample(bucket).size().rename("panels").reset_index())
    fig = px.bar(ts, x="event_local", y="panels", color="station_id",
                 labels={"event_local": "Time", "panels": "Panels checked", "station_id": "Station"})
    fig.update_layout(barmode="stack", height=420, legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    cum = df.sort_values("event_local").assign(cumulative=1)
    cum["cumulative"] = cum.groupby(["station_id", "event_date"]).cumcount() + 1
    fig2 = px.line(cum, x="event_local", y="cumulative", color="station_id",
                   title="Running total within each day (steeper = faster)")
    fig2.update_layout(height=320, legend_title_text="")
    st.plotly_chart(fig2, width="stretch")

with tab_hour:
    hours = (df.groupby(["event_hour"]).agg(panels=("panels", "sum"),
                                            days=("event_date", "nunique")).reset_index())
    hours["avg_per_day"] = hours["panels"] / hours["days"]
    left, right = st.columns(2)
    left.plotly_chart(px.bar(hours, x="event_hour", y="panels", title="Total panels by hour of day",
                             labels={"event_hour": "Hour", "panels": "Panels"}),
                      width="stretch")
    right.plotly_chart(px.bar(hours, x="event_hour", y="avg_per_day",
                              title="Average panels per day, by hour of day",
                              labels={"event_hour": "Hour", "avg_per_day": "Avg panels"}),
                       width="stretch")
    wd = df.groupby("weekday").size().reindex(weekdays).fillna(0).reset_index(name="panels")
    st.plotly_chart(px.bar(wd, x="weekday", y="panels", title="By weekday"), width="stretch")

with tab_heat:
    pivot = df.pivot_table(index="event_date", columns="event_hour", values="panels",
                           aggfunc="sum", fill_value=0)
    pivot.index = pivot.index.astype(str)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", text_auto=True,
                    labels={"x": "Hour of day", "y": "Date", "color": "Panels"})
    fig.update_layout(height=max(320, 28 * len(pivot) + 120))
    st.plotly_chart(fig, width="stretch")

with tab_cycle:
    if good_gaps.empty:
        st.info("No gap data yet (needs at least two consecutive panels in a session).")
    else:
        fig = px.histogram(good_gaps, nbins=40, labels={"value": "Seconds between consecutive panels"},
                           title=f"Time between panels (gaps ≤ {break_s}s)")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, width="stretch")
        p = good_gaps.quantile([.1, .5, .9])
        st.markdown(
            f"**Capacity reading:** fast 10% of cycles ≤ **{p[.1]:.0f}s**, typical **{p[.5]:.0f}s**, "
            f"slow 10% ≥ **{p[.9]:.0f}s** → typical capacity ≈ **{3600 / p[.5]:.0f} panels/hour** "
            f"per station when working continuously.")
        st.caption(f"{len(df['gap_s'].dropna()) - len(good_gaps)} gaps longer than {break_s}s "
                   "were excluded as breaks/stoppages.")

with tab_data:
    show = df[["event_local", "station_id", "session_id", "running_count",
               "transfer_time_s", "gap_s"]].sort_values("event_local", ascending=False)
    st.dataframe(show, width="stretch", hide_index=True)
    st.download_button("⬇ Download filtered rows (CSV)", show.to_csv(index=False).encode("utf-8"),
                       "panels_checked_filtered.csv", "text/csv")

with tab_up:
    s = sessions.copy()
    s = s[s["station_id"].isin(chosen_stations)]
    s["end"] = s["ended_utc"].fillna(s["last_seen_utc"])
    s["run_hours"] = (s["end"] - s["started_utc"]).dt.total_seconds() / 3600
    s = s[(s["started_local"].dt.date >= d_start) & (s["started_local"].dt.date <= d_end)]
    if s.empty:
        st.info("No sessions in this date range.")
    else:
        st.write("Each row is one run of the camera program. Use *run hours* vs. panels to see "
                 "how much of the time the station was actually productive.")
        view = s[["session_id", "station_id", "started_local", "run_hours", "source"]].copy()
        view["panels"] = view["session_id"].map(df.groupby("session_id").size()).fillna(0).astype(int)
        view["panels_per_run_hour"] = (view["panels"] / view["run_hours"]).round(1)
        view["run_hours"] = view["run_hours"].round(2)
        st.dataframe(view.sort_values("started_local", ascending=False),
                     width="stretch", hide_index=True)
