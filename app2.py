import os
from pathlib import Path
import random
from typing import Dict, List, Tuple
import hashlib
import time
import base64
from io import BytesIO

import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Attempt to import OR-Tools; provide graceful fallback if unavailable
ORTOOLS_AVAILABLE = True
try:
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover
    ORTOOLS_AVAILABLE = False


# ==========================
# Constants and Configuration
# ==========================

APP_TITLE = "Trackमित्र — Controller Prototype"
PRIMARY_BLUE = "#0A3D62"  # Deep Blue
ACCENT_SAFFRON = "#FF6B00"  # Saffron
LIGHT_BG = "#F7F9FB"

TYPE_OPTIONS = ["Express", "Local", "Freight"]

# Energy rate (units per minute) per train type
ENERGY_RATE: Dict[str, int] = {
    "Express": 5,
    "Local": 3,
    "Freight": 8,
}

# Delay weights for objective (higher => more important / penalized)
# These will be made adjustable by user
DEFAULT_DELAY_WEIGHTS: Dict[str, int] = {
    "Express": 3,
    "Local": 2,
    "Freight": 1,
}

# Plot colors per type (Express deep blue, Local saffron, Freight grey)
COLOR_MAP: Dict[str, str] = {
    "Express": PRIMARY_BLUE,
    "Local": ACCENT_SAFFRON,
    "Freight": "#7F8C8D",
}

# Train type icons
TRAIN_ICONS: Dict[str, str] = {
    "Express": "🚄",
    "Local": "🚋", 
    "Freight": "🚛",
}

# Optimization strategies
OPTIMIZATION_STRATEGIES = ["CP-SAT", "Greedy", "Hybrid"]


# ==========================
# Session State Initialization
# ==========================

def init_state() -> None:
    if "trains" not in st.session_state:
        # Seed demo trains
        st.session_state.trains = [
            {"id": "E101", "name": "Rajdhani", "type": "Express", "duration": 90, "planned_offset": 30},
            {"id": "L205", "name": "Mumbai Local", "type": "Local", "duration": 50, "planned_offset": 20},
            {"id": "F301", "name": "Coal Freight", "type": "Freight", "duration": 120, "planned_offset": 40},
            {"id": "E102", "name": "Shatabdi", "type": "Express", "duration": 80, "planned_offset": 60},
            {"id": "L206", "name": "Chennai Suburban", "type": "Local", "duration": 45, "planned_offset": 55},
        ]
    if "delays" not in st.session_state:
        st.session_state.delays = {t["id"]: 0 for t in st.session_state.trains}
    if "sim_horizon_mins" not in st.session_state:
        st.session_state.sim_horizon_mins = 600  # 10 hours
    if "headway_mins" not in st.session_state:
        st.session_state.headway_mins = 5
    if "lambda_energy" not in st.session_state:
        st.session_state.lambda_energy = 0.3
    if "greedy_schedule" not in st.session_state:
        st.session_state.greedy_schedule = None
    if "optimized_schedule" not in st.session_state:
        st.session_state.optimized_schedule = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "random_seed" not in st.session_state:
        st.session_state.random_seed = 42
    if "max_random_delay" not in st.session_state:
        st.session_state.max_random_delay = 20
    if "delay_weights" not in st.session_state:
        st.session_state.delay_weights = DEFAULT_DELAY_WEIGHTS.copy()
    if "optimization_strategy" not in st.session_state:
        st.session_state.optimization_strategy = "CP-SAT"
    if "conflicts_resolved" not in st.session_state:
        st.session_state.conflicts_resolved = 0
    if "punctuality_percent" not in st.session_state:
        st.session_state.punctuality_percent = 0.0


# ==========================
# UI Helpers and Layout
# ==========================

def inject_theme_css() -> None:
    st.markdown(
        f"""
        <style>
        .app-header {{
            background: {PRIMARY_BLUE};
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 12px;
        }}
        .app-title {{
            font-size: 1.4rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .accent-badge {{
            background: {ACCENT_SAFFRON};
            color: white;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.75rem;
            margin-left: 6px;
        }}
        .kpi-card {{
            background: white;
            border: 1px solid #E6ECF1;
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            transition: all 0.3s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .kpi-title {{
            color: #607D8B;
            font-size: 0.85rem;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .kpi-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: {PRIMARY_BLUE};
        }}
        .kpi-improvement {{
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 4px;
        }}
        .kpi-improvement.positive {{
            color: #2E7D32;
        }}
        .kpi-improvement.negative {{
            color: #D32F2F;
        }}
        .recommendation-card {{
            background: #F8F9FA;
            border-left: 4px solid {ACCENT_SAFFRON};
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }}
        .energy-savings {{
            color: #2E7D32;
            font-weight: 600;
        }}
        .extra-holding {{
            color: #D32F2F;
            font-weight: 600;
        }}
        .stApp {{ background-color: {LIGHT_BG}; }}
        .plotly-graph-div {{ border-radius: 8px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    inject_theme_css()
    logo_path = Path("assets/logo.png")
    with st.container():
        cols = st.columns([1, 10])
        with cols[0]:
            if logo_path.exists():
                st.image(str(logo_path), use_column_width=True)
            else:
                # Placeholder badge when logo is missing
                st.markdown(
                    f"""
                    <div style="background:{ACCENT_SAFFRON};color:white;padding:10px;border-radius:8px;text-align:center;font-weight:700;">IR</div>
                    """,
                    unsafe_allow_html=True,
                )
        with cols[1]:
            st.markdown(
                f"""
                <div class="app-header">
                  <div class="app-title">{APP_TITLE} <span class="accent-badge">Prototype</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==========================
# Data and Scheduling Utilities
# ==========================

def planned_schedule(trains: List[Dict]) -> pd.DataFrame:
    rows = []
    for t in trains:
        start = int(t["planned_offset"])
        end = start + int(t["duration"])
        rows.append({
            "id": t["id"],
            "name": t["name"],
            "type": t["type"],
            "planned_start": start,
            "planned_end": end,
        })
    df = pd.DataFrame(rows)
    return df


def list_schedule_to_df(trains: List[Dict], schedule: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    """Convert schedule dict {id: (start, end)} to DataFrame with metadata."""
    meta = {t["id"]: t for t in trains}
    rows = []
    for tid, (s, e) in schedule.items():
        t = meta[tid]
        rows.append({
            "id": tid,
            "name": t["name"],
            "type": t["type"],
            "start": int(s),
            "end": int(e),
            "planned_start": int(t["planned_offset"]),
            "planned_end": int(t["planned_offset"]) + int(t["duration"]),
            "duration": int(t["duration"]),
        })
    return pd.DataFrame(rows)


def planned_to_schedule_df(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Convert planned schedule into schedule-like DataFrame for plotting and KPIs."""
    if plan_df is None or plan_df.empty:
        return pd.DataFrame()
    df = plan_df.copy()
    df["start"] = df["planned_start"]
    df["end"] = df["planned_end"]
    df["duration"] = df["planned_end"] - df["planned_start"]
    return df[["id", "name", "type", "start", "end", "planned_start", "planned_end", "duration"]]


def inject_random_delays(trains: List[Dict], max_delay: int, seed: int) -> Dict[str, int]:
    rng = random.Random(seed)
    delays = {}
    for t in trains:
        delays[t["id"]] = rng.randint(0, max(0, int(max_delay)))
    return delays


def greedy_schedule(trains: List[Dict], delays: Dict[str, int], headway: int) -> Dict[str, Tuple[int, int]]:
    """Priority-based list scheduling with readiness times and headway.

    - readiness = planned_offset + delay
    - choose highest priority available at current time; if none, jump to next readiness
    - single-track non-overlap with headway
    """
    # Use adjustable delay weights as priority map
    priority_map = st.session_state.delay_weights.copy()
    remaining = {t["id"]: {
        "id": t["id"],
        "name": t["name"],
        "type": t["type"],
        "duration": int(t["duration"]),
        "planned": int(t["planned_offset"]),
        "ready": int(t["planned_offset"]) + int(delays.get(t["id"], 0)),
        "priority": priority_map.get(t["type"], 1),
    } for t in trains}

    # Start time is at earliest readiness among trains
    time_now = min(r["ready"] for r in remaining.values()) if remaining else 0
    schedule: Dict[str, Tuple[int, int]] = {}

    while remaining:
        available = [r for r in remaining.values() if r["ready"] <= time_now]
        if not available:
            # Jump time to next readiness
            time_now = min(r["ready"] for r in remaining.values())
            available = [r for r in remaining.values() if r["ready"] <= time_now]

        # Choose by highest priority then earliest planned
        available.sort(key=lambda x: (-x["priority"], x["planned"]))
        chosen = available[0]

        start_time = max(time_now, chosen["ready"])
        end_time = start_time + chosen["duration"]
        schedule[chosen["id"]] = (start_time, end_time)

        # Update clock: include headway
        time_now = end_time + int(headway)

        # Remove scheduled
        remaining.pop(chosen["id"])

    return schedule


def solve_cpsat(
    trains: List[Dict],
    delays: Dict[str, int],
    headway: int,
    energy_lambda: float,
) -> Tuple[Dict[str, Tuple[int, int]], str]:
    """Solve energy-aware scheduling with CP-SAT.

    Decision vars:
    - start_t (int)
    - delay_t (>= 0) where delay w.r.t. planned_offset

    Constraints:
    - start_t >= planned_offset + injected_delay (readiness)
    - Non-overlap with headway via pairwise precedence booleans

    Objective:
    - Minimize sum_i ( w_delay[type] * delay_i + energy_lambda * energy_rate[type] * delay_i )
      Note: Energy of runtime is constant; only idling due to delay is penalized here.
    """
    if not ORTOOLS_AVAILABLE:
        return {}, "OR-Tools not installed"

    model = cp_model.CpModel()

    # Build lookup maps
    train_ids = [t["id"] for t in trains]
    idx_by_id = {tid: i for i, tid in enumerate(train_ids)}
    duration = {t["id"]: int(t["duration"]) for t in trains}
    planned = {t["id"]: int(t["planned_offset"]) for t in trains}
    readiness = {tid: planned[tid] + int(delays.get(tid, 0)) for tid in train_ids}

    # Horizon: generous upper bound
    horizon = max(planned.values()) + sum(duration.values()) + len(trains) * (headway + 5)
    if horizon < 1:
        horizon = 1

    # Variables
    start_vars: Dict[str, cp_model.IntVar] = {}
    delay_vars: Dict[str, cp_model.IntVar] = {}

    for tid in train_ids:
        start_vars[tid] = model.NewIntVar(readiness[tid], horizon, f"start_{tid}")
        delay_vars[tid] = model.NewIntVar(0, horizon, f"delay_{tid}")
        # delay >= start - planned
        # temp = start - planned -> temp + planned == start
        temp = model.NewIntVar(-horizon, horizon, f"temp_{tid}")
        model.Add(temp + planned[tid] == start_vars[tid])
        model.Add(delay_vars[tid] >= temp)
        model.Add(delay_vars[tid] >= 0)

    # Non-overlap constraints with headway via pairwise precedence
    big_m = horizon + max(duration.values()) + headway + 10
    for i in range(len(train_ids)):
        for j in range(i + 1, len(train_ids)):
            ti = train_ids[i]
            tj = train_ids[j]
            before_ij = model.NewBoolVar(f"{ti}_before_{tj}")
            # ti ends before tj starts
            model.Add(start_vars[ti] + duration[ti] + headway <= start_vars[tj] + big_m * (1 - before_ij))
            # tj ends before ti starts
            model.Add(start_vars[tj] + duration[tj] + headway <= start_vars[ti] + big_m * before_ij)

    # Objective: linear weighted sum of delays with energy component
    # Scale by 100 to keep integer coefficients
    objective_terms = []
    for t in trains:
        tid = t["id"]
        typ = t["type"]
        w_delay = int(st.session_state.delay_weights.get(typ, 1))
        e_rate = int(ENERGY_RATE.get(typ, 1))
        lam = int(round(energy_lambda * 100))
        coef = (100 * w_delay + lam * e_rate)
        objective_terms.append(coef * delay_vars[tid])

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8
    status_code = solver.Solve(model)

    status_map = {
        cp_model.OPTIMAL: "Optimal",
        cp_model.FEASIBLE: "Feasible",
        cp_model.INFEASIBLE: "Infeasible",
        cp_model.MODEL_INVALID: "Model Invalid",
        cp_model.UNKNOWN: "Unknown",
    }
    status = status_map.get(status_code, "Unknown")

    schedule: Dict[str, Tuple[int, int]] = {}
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for tid in train_ids:
            s = solver.Value(start_vars[tid])
            e = s + duration[tid]
            schedule[tid] = (int(s), int(e))

    return schedule, status


# ==========================
# KPIs and Analytics
# ==========================

def compute_energy_per_train(row: pd.Series) -> int:
    typ = row.get("type", "Local")
    rate = ENERGY_RATE.get(typ, 3)
    delay = int(row.get("start", 0) - row.get("planned_start", 0))
    runtime = int(row.get("duration", 0))
    return rate * max(0, delay + runtime)


def compute_kpis(
    planned_df: pd.DataFrame,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    horizon: int,
) -> Dict[str, float]:
    def avg_delay(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 0.0
        return float(np.mean((df["start"] - df["planned_start"]).clip(lower=0)))

    def throughput(df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        return int(np.sum(df["end"] <= horizon))

    def total_energy(df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        return int(df.apply(compute_energy_per_train, axis=1).sum())
    
    def count_conflicts(df: pd.DataFrame, headway: int) -> int:
        """Count potential conflicts in schedule"""
        if df is None or df.empty or len(df) < 2:
            return 0
        conflicts = 0
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                train_i = df.iloc[i]
                train_j = df.iloc[j]
                # Check if trains overlap considering headway
                if (train_i["start"] < train_j["end"] + headway and 
                    train_j["start"] < train_i["end"] + headway):
                    conflicts += 1
        return conflicts
    
    def punctuality_percent(df: pd.DataFrame) -> float:
        """Calculate percentage of trains within 10 minutes of planned schedule"""
        if df is None or df.empty:
            return 0.0
        delays = (df["start"] - df["planned_start"]).clip(lower=0)
        on_time = np.sum(delays <= 10)
        return float(on_time / len(df) * 100)

    headway = st.session_state.headway_mins
    conflicts_before = count_conflicts(before_df, headway)
    conflicts_after = count_conflicts(after_df, headway)
    
    kpis = {
        "avg_delay_before": avg_delay(before_df),
        "avg_delay_after": avg_delay(after_df),
        "throughput_before": throughput(before_df),
        "throughput_after": throughput(after_df),
        "energy_before": total_energy(before_df),
        "energy_after": total_energy(after_df),
        "conflicts_before": conflicts_before,
        "conflicts_after": conflicts_after,
        "conflicts_resolved": max(0, conflicts_before - conflicts_after),
        "punctuality_before": punctuality_percent(before_df),
        "punctuality_after": punctuality_percent(after_df),
    }
    
    # Update session state
    st.session_state.conflicts_resolved = kpis["conflicts_resolved"]
    st.session_state.punctuality_percent = kpis["punctuality_after"]
    
    return kpis


def plot_timeline(df: pd.DataFrame, title: str) -> None:
    if df is None or df.empty:
        st.info("No data to plot yet. Add trains and simulate.")
        return

    # Build timeline in datetime space
    base = pd.Timestamp("2025-01-01 00:00:00")
    df_plot = df.copy()
    df_plot["Start"] = base + pd.to_timedelta(df_plot["start"], unit="m")
    df_plot["Finish"] = base + pd.to_timedelta(df_plot["end"], unit="m")
    df_plot["Train"] = (df_plot["type"].map(TRAIN_ICONS) + " " + 
                       df_plot["name"].astype(str) + " (" + df_plot["id"].astype(str) + ")")

    fig = px.timeline(
        df_plot,
        x_start="Start",
        x_end="Finish",
        y="Train",
        color="type",
        color_discrete_map=COLOR_MAP,
        title=title,
        hover_data={"start": True, "end": True, "type": True, "planned_start": True},
    )
    fig.update_yaxes(autorange="reversed")  # earliest at top
    fig.update_layout(
        showlegend=True,
        height=420,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    # Create a more unique key by including title, data length, and current time
    unique_key = hashlib.md5(f"{title}_{len(df)}_{id(df)}_{time.time()}".encode()).hexdigest()[:8]
    st.plotly_chart(fig, use_container_width=True, key=f"timeline_{unique_key}")


def generate_recommendations(
    trains: List[Dict],
    greedy_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    delays: Dict[str, int],
    headway: int,
) -> List[str]:
    tips: List[str] = []
    if greedy_df is None or greedy_df.empty or opt_df is None or opt_df.empty:
        return tips

    meta = {t["id"]: t for t in trains}
    gd = greedy_df.set_index("id")
    od = opt_df.set_index("id")

    # Compare each train's change
    for tid in od.index:
        tmeta = meta[tid]
        typ = tmeta["type"]
        g_start = int(gd.loc[tid, "start"]) if tid in gd.index else None
        o_start = int(od.loc[tid, "start"]) if tid in od.index else None
        delay_injected = int(delays.get(tid, 0))
        planned = int(tmeta["planned_offset"])
        g_delay = (g_start - planned) if g_start is not None else 0
        o_delay = (o_start - planned) if o_start is not None else 0

        if g_start is None or o_start is None:
            continue

        if o_start < g_start:
            tips.append(
                f"{tmeta['name']} ({tid}) was advanced by {g_start - o_start} min to cut weighted delay for {typ}."
            )
        elif o_start > g_start:
            tips.append(
                f"{tmeta['name']} ({tid}) was held {o_start - g_start} min to maintain safety headway and reduce conflicts."
            )

        if o_delay > delay_injected:
            tips.append(
                f"Due to congestion after disruptions (+{delay_injected} min), {tid} incurred extra holding. Consider resequencing or increasing buffer."
            )

        # Energy rationale
        if o_delay < g_delay:
            tips.append(
                f"By reducing idle time for {tid}, energy consumption decreased by {(g_delay - o_delay) * ENERGY_RATE.get(typ, 3)} units."
            )

    if not tips:
        tips.append("Schedules are similar. Consider adjusting headway or disruption severity to see differences.")
    return tips


# ==========================
# Sidebar Forms
# ==========================

def add_train_form() -> None:
    st.subheader("Add Train")
    with st.form("add_train_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            train_id = st.text_input("Train ID", max_chars=10)
            train_name = st.text_input("Name")
        with c2:
            train_type = st.selectbox("Type", TYPE_OPTIONS)
            duration = st.number_input("Duration (min)", min_value=5, max_value=600, value=60, step=5)
            planned_offset = st.number_input("Planned Start Offset (min)", min_value=0, max_value=1440, value=30, step=5)
        submitted = st.form_submit_button("Add Train")
        if submitted:
            if not train_id or not train_name:
                st.warning("Please provide both Train ID and Name")
            elif any(t["id"] == train_id for t in st.session_state.trains):
                st.error("Train ID already exists. Use a unique ID.")
            else:
                st.session_state.trains.append({
                    "id": train_id.strip(),
                    "name": train_name.strip(),
                    "type": train_type,
                    "duration": int(duration),
                    "planned_offset": int(planned_offset),
                })
                # Initialize delay for new train
                st.session_state.delays[train_id.strip()] = 0
                st.success("Train added.")


def manage_trains() -> None:
    st.subheader("Manage Trains")
    if not st.session_state.trains:
        st.info("No trains yet.")
        return
    df = pd.DataFrame(st.session_state.trains)
    st.dataframe(df, use_container_width=True, hide_index=True)

    to_delete = st.multiselect(
        "Select trains to delete",
        options=[t["id"] for t in st.session_state.trains],
    )
    if st.button("Delete Selected", type="secondary") and to_delete:
        st.session_state.trains = [t for t in st.session_state.trains if t["id"] not in to_delete]
        for tid in to_delete:
            st.session_state.delays.pop(tid, None)
        st.success(f"Deleted {len(to_delete)} trains.")

    # Editable energy & weight overview
    st.markdown("**Type Parameters**")
    df_params = pd.DataFrame({
        "Type": TYPE_OPTIONS,
        "Delay Weight": [st.session_state.delay_weights[t] for t in TYPE_OPTIONS],
        "Energy Rate (u/min)": [ENERGY_RATE[t] for t in TYPE_OPTIONS],
    })
    st.dataframe(df_params, hide_index=True, use_container_width=True)


def energy_breakdown_sidebar(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    st.subheader("Energy Breakdown per Train")
    if before_df is None or before_df.empty:
        st.info("Run simulation to see energy breakdown.")
        return
    bdf = before_df.copy()
    bdf["Energy Before"] = bdf.apply(compute_energy_per_train, axis=1)
    if after_df is not None and not after_df.empty:
        adf = after_df.set_index("id")
        # Align columns to compute energy after
        merged = bdf.set_index("id").copy()
        merged.loc[:, "start"] = adf["start"]
        merged.loc[:, "end"] = adf["end"]
        merged = merged.reset_index()
        bdf["Energy After"] = merged.apply(compute_energy_per_train, axis=1)
    else:
        bdf["Energy After"] = 0

    display_cols = ["id", "name", "type", "duration", "planned_start", "start", "Energy Before", "Energy After"]
    bdf = bdf[display_cols]
    st.dataframe(bdf, use_container_width=True, hide_index=True)


# ==========================
# Controls and Main Flow
# ==========================

def controls() -> None:
    st.sidebar.markdown(f"**Settings**")
    st.session_state.headway_mins = st.sidebar.slider("Safety Headway (min)", 0, 30, st.session_state.headway_mins, 1)
    st.session_state.lambda_energy = st.sidebar.slider("Energy Weight (λ)", 0.0, 2.0, float(st.session_state.lambda_energy), 0.1)
    st.session_state.sim_horizon_mins = st.sidebar.slider("Simulation Horizon (min)", 60, 1440, int(st.session_state.sim_horizon_mins), 30)
    st.session_state.max_random_delay = st.sidebar.slider("Max Random Delay (min)", 0, 120, int(st.session_state.max_random_delay), 5)
    st.session_state.random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999_999, value=int(st.session_state.random_seed), step=1)

    # Optimization Strategy Selection
    st.sidebar.markdown("**Optimization Strategy**")
    st.session_state.optimization_strategy = st.sidebar.selectbox(
        "Choose Algorithm", 
        OPTIMIZATION_STRATEGIES, 
        index=OPTIMIZATION_STRATEGIES.index(st.session_state.optimization_strategy)
    )

    # Train Priority Weights
    st.sidebar.markdown("**Train Priority Weights**")
    st.session_state.delay_weights["Express"] = st.sidebar.slider(
        f"{TRAIN_ICONS['Express']} Express Priority", 1, 10, st.session_state.delay_weights["Express"], 1
    )
    st.session_state.delay_weights["Local"] = st.sidebar.slider(
        f"{TRAIN_ICONS['Local']} Local Priority", 1, 10, st.session_state.delay_weights["Local"], 1
    )
    st.session_state.delay_weights["Freight"] = st.sidebar.slider(
        f"{TRAIN_ICONS['Freight']} Freight Priority", 1, 10, st.session_state.delay_weights["Freight"], 1
    )

    if st.sidebar.button("Inject Random Delays", type="primary"):
        st.session_state.delays = inject_random_delays(
            st.session_state.trains, st.session_state.max_random_delay, st.session_state.random_seed
        )
    with st.sidebar:
        add_train_form()
        manage_trains()


def run_simulation_and_optimization() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    trains = st.session_state.trains
    headway = int(st.session_state.headway_mins)
    lam = float(st.session_state.lambda_energy)
    strategy = st.session_state.optimization_strategy

    plan_df = planned_schedule(trains)

    # Greedy baseline with disruptions
    g_sched = greedy_schedule(trains, st.session_state.delays, headway)
    g_df = list_schedule_to_df(trains, g_sched)
    st.session_state.greedy_schedule = g_df

    # Choose optimization strategy
    if strategy == "Greedy":
        o_df = g_df
        solver_status = "Greedy Algorithm"
    elif strategy == "CP-SAT" and ORTOOLS_AVAILABLE and trains:
        opt_sched, status = solve_cpsat(trains, st.session_state.delays, headway, lam)
        o_df = list_schedule_to_df(trains, opt_sched) if opt_sched else pd.DataFrame()
        solver_status = status
    elif strategy == "Hybrid" and ORTOOLS_AVAILABLE and trains:
        # Hybrid: Use CP-SAT if available, fallback to greedy
        with st.spinner("Running hybrid optimization..."):
            opt_sched, status = solve_cpsat(trains, st.session_state.delays, headway, lam)
            if opt_sched:
                o_df = list_schedule_to_df(trains, opt_sched)
                solver_status = f"Hybrid (CP-SAT): {status}"
            else:
                o_df = g_df
                solver_status = "Hybrid (Greedy fallback)"
    else:
        solver_status = "OR-Tools not installed"
        o_df = pd.DataFrame()

    st.session_state.optimized_schedule = o_df
    return plan_df, g_df, o_df, solver_status


def kpi_card(title: str, value: str, icon: str = "", improvement: str = "", status: str = "neutral") -> None:
    """Enhanced KPI card with icons and status indicators"""
    improvement_html = ""
    if improvement:
        status_class = "positive" if status == "positive" else "negative"
        improvement_html = f'<div class="kpi-improvement {status_class}">{improvement}</div>'
    
    icon_html = f'<span style="font-size: 1.2rem;">{icon}</span>' if icon else ""
    
    st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">{icon_html} {title}</div>
            <div class="kpi-value">{value}</div>
            {improvement_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def plot_energy_breakdown(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    """Create energy breakdown chart"""
    if before_df is None or before_df.empty:
        st.info("No data available for energy breakdown")
        return
    
    # Calculate energy for each train
    before_df_copy = before_df.copy()
    before_df_copy["Energy Before"] = before_df_copy.apply(compute_energy_per_train, axis=1)
    
    if after_df is not None and not after_df.empty:
        after_df_copy = after_df.copy()
        # Align columns to compute energy after
        merged = before_df_copy.set_index("id").copy()
        merged.loc[:, "start"] = after_df_copy.set_index("id")["start"]
        merged.loc[:, "end"] = after_df_copy.set_index("id")["end"]
        merged = merged.reset_index()
        merged["Energy After"] = merged.apply(compute_energy_per_train, axis=1)
    else:
        merged = before_df_copy.copy()
        merged["Energy After"] = 0
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Add energy before bars
    fig.add_trace(go.Bar(
        name="Energy Before",
        x=merged["name"] + " (" + merged["id"] + ")",
        y=merged["Energy Before"],
        marker_color="#FF6B00",  # Saffron
        text=merged["Energy Before"],
        textposition="inside",
    ))
    
    # Add energy after bars
    fig.add_trace(go.Bar(
        name="Energy After",
        x=merged["name"] + " (" + merged["id"] + ")",
        y=merged["Energy After"],
        marker_color="#0A3D62",  # Deep blue
        text=merged["Energy After"],
        textposition="inside",
    ))
    
    fig.update_layout(
        title="Energy Consumption Breakdown by Train",
        xaxis_title="Train",
        yaxis_title="Energy Units",
        barmode="group",
        height=400,
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_collapsible_recommendations(recommendations: List[str], trains: List[Dict]) -> None:
    """Display recommendations in collapsible cards grouped by train"""
    if not recommendations:
        st.info("No recommendations available.")
        return
    
    # Group recommendations by train ID
    train_recommendations = {}
    general_recommendations = []
    
    for rec in recommendations:
        # Extract train ID from recommendation text
        train_found = False
        for train in trains:
            if train["id"] in rec or train["name"] in rec:
                if train["id"] not in train_recommendations:
                    train_recommendations[train["id"]] = []
                train_recommendations[train["id"]].append(rec)
                train_found = True
                break
        if not train_found:
            general_recommendations.append(rec)
    
    # Display train-specific recommendations
    for train_id, recs in train_recommendations.items():
        train = next(t for t in trains if t["id"] == train_id)
        icon = TRAIN_ICONS.get(train["type"], "🚆")
        
        with st.expander(f"{icon} {train['name']} ({train_id}) - {len(recs)} recommendation(s)", expanded=False):
            for rec in recs:
                # Highlight energy savings and extra holding
                highlighted_rec = rec
                if "energy consumption decreased" in rec.lower():
                    highlighted_rec = rec.replace("energy consumption decreased", 
                                                '<span class="energy-savings">energy consumption decreased</span>')
                elif "extra holding" in rec.lower():
                    highlighted_rec = rec.replace("extra holding", 
                                                '<span class="extra-holding">extra holding</span>')
                
                st.markdown(f"• {highlighted_rec}", unsafe_allow_html=True)
    
    # Display general recommendations
    if general_recommendations:
        with st.expander("General Recommendations", expanded=False):
            for rec in general_recommendations:
                st.markdown(f"• {rec}")


def generate_pdf_report(kpis: Dict[str, float], recommendations: List[str], trains: List[Dict], solver_status: str) -> bytes:
    """Generate PDF report with KPIs, plots, and recommendations"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0A3D62')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#0A3D62')
        )
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph("Trackमित्र — AI Powered Train Controller", title_style))
        story.append(Spacer(1, 20))
        
        # KPIs Section
        story.append(Paragraph("Key Performance Indicators", heading_style))
        
        kpi_data = [
            ["Metric", "Before", "After", "Change"],
            ["Average Delay (min)", f"{kpis['avg_delay_before']:.1f}", f"{kpis['avg_delay_after']:.1f}", 
             f"{kpis['avg_delay_after'] - kpis['avg_delay_before']:+.1f}"],
            ["Throughput (trains)", str(kpis['throughput_before']), str(kpis['throughput_after']), 
             f"{kpis['throughput_after'] - kpis['throughput_before']:+d}"],
            ["Energy Consumption", str(kpis['energy_before']), str(kpis['energy_after']), 
             f"{kpis['energy_after'] - kpis['energy_before']:+d}"],
            ["Conflicts Resolved", "-", str(kpis['conflicts_resolved']), f"{kpis['conflicts_resolved']:d}"],
            ["Punctuality %", f"{kpis['punctuality_before']:.1f}%", f"{kpis['punctuality_after']:.1f}%", 
             f"{kpis['punctuality_after'] - kpis['punctuality_before']:+.1f}%"]
        ]
        
        kpi_table = Table(kpi_data)
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0A3D62')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 20))
        
        # Solver Status
        story.append(Paragraph(f"Optimization Status: {solver_status}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Recommendations Section
        story.append(Paragraph("AI Recommendations", heading_style))
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
            story.append(Spacer(1, 6))
        
        if not recommendations:
            story.append(Paragraph("No specific recommendations available.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Train Summary
        story.append(Paragraph("Train Summary", heading_style))
        train_data = [["ID", "Name", "Type", "Duration (min)", "Planned Start (min)"]]
        for train in trains:
            train_data.append([
                train["id"], 
                train["name"], 
                train["type"], 
                str(train["duration"]), 
                str(train["planned_offset"])
            ])
        
        train_table = Table(train_data)
        train_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF6B00')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(train_table)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        st.error("PDF generation requires reportlab. Install with: pip install reportlab")
        return b""
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return b""


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🚆")
    init_state()
    header()

    controls()

    st.divider()

    # Run the baseline and optimization each render
    plan_df, g_df, o_df, solver_status = run_simulation_and_optimization()

    # Enhanced KPIs with icons and status indicators
    kpis = compute_kpis(plan_df, g_df, o_df, st.session_state.sim_horizon_mins)
    
    # Calculate improvements
    delay_change = kpis['avg_delay_after'] - kpis['avg_delay_before']
    throughput_change = kpis['throughput_after'] - kpis['throughput_before']
    energy_change = kpis['energy_after'] - kpis['energy_before']
    punctuality_change = kpis['punctuality_after'] - kpis['punctuality_before']
    
    # Display enhanced KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delay_icon = "⏱️"
        delay_status = "positive" if delay_change < 0 else "negative" if delay_change > 0 else "neutral"
        delay_improvement = f"{delay_change:+.1f} min" if delay_change != 0 else ""
        kpi_card("Average Delay", f"{kpis['avg_delay_after']:.1f} min", 
                delay_icon, delay_improvement, delay_status)
    
    with c2:
        throughput_icon = "🚆"
        throughput_status = "positive" if throughput_change > 0 else "negative" if throughput_change < 0 else "neutral"
        throughput_improvement = f"{throughput_change:+d} trains" if throughput_change != 0 else ""
        kpi_card("Throughput", f"{kpis['throughput_after']} trains", 
                throughput_icon, throughput_improvement, throughput_status)
    
    with c3:
        energy_icon = "🔋"
        energy_percent = ((energy_change / kpis['energy_before']) * 100) if kpis['energy_before'] > 0 else 0
        energy_status = "positive" if energy_change < 0 else "negative" if energy_change > 0 else "neutral"
        energy_improvement = f"{energy_percent:+.1f}%" if energy_change != 0 else ""
        kpi_card("Energy Savings", f"{kpis['energy_after']} units", 
                energy_icon, energy_improvement, energy_status)
    
    with c4:
        punctuality_icon = "🎯"
        punctuality_status = "positive" if punctuality_change > 0 else "negative" if punctuality_change < 0 else "neutral"
        punctuality_improvement = f"{punctuality_change:+.1f}%" if punctuality_change != 0 else ""
        kpi_card("Punctuality", f"{kpis['punctuality_after']:.1f}%", 
                punctuality_icon, punctuality_improvement, punctuality_status)

    # Additional KPIs in second row
    c5, c6 = st.columns(2)
    with c5:
        kpi_card("Conflicts Resolved", f"{kpis['conflicts_resolved']} conflicts", "⚡", "", "positive")
    with c6:
        kpi_card("Optimization Status", solver_status, "🔧", "", "neutral")

    # Export PDF Report Button
    st.markdown("---")
    col_export, col_spacer = st.columns([1, 4])
    with col_export:
        if st.button("📄 Download PDF Report", type="secondary"):
            recommendations = generate_recommendations(st.session_state.trains, g_df, o_df, st.session_state.delays, st.session_state.headway_mins)
            pdf_data = generate_pdf_report(kpis, recommendations, st.session_state.trains, solver_status)
            if pdf_data:
                st.download_button(
                    label="Download Report",
                    data=pdf_data,
                    file_name=f"trackmitra_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

    # Layout: timelines with tabs
    planned_like_df = planned_to_schedule_df(plan_df)
    tab1, tab2, tab3 = st.tabs(["📊 Planned vs Optimized", "🔄 Greedy vs Optimized", "🔋 Energy Breakdown"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            plot_timeline(planned_like_df, "Planned (Reference)")
        with col2:
            plot_timeline(o_df, f"Optimized ({st.session_state.optimization_strategy})")
    
    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            plot_timeline(g_df, "Baseline (Greedy)")
        with col4:
            plot_timeline(o_df, f"Optimized ({st.session_state.optimization_strategy})")
    
    with tab3:
        plot_energy_breakdown(g_df, o_df)

    # Enhanced Recommendations with collapsible sections
    st.subheader("🤖 AI-generated Recommendations")
    recos = generate_recommendations(st.session_state.trains, g_df, o_df, st.session_state.delays, st.session_state.headway_mins)
    st.session_state.recommendations = recos
    display_collapsible_recommendations(recos, st.session_state.trains)

    # Sidebar energy table
    with st.sidebar:
        energy_breakdown_sidebar(g_df, o_df)

    st.divider()
    st.markdown(
        "🚂 **Trackमित्र** — AI-Powered Train Traffic Controller | Built with Streamlit & OR-Tools",
    )


if __name__ == "__main__":
    main()
