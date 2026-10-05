"""
HRTN — Strategic Hospital Operations Dashboard
================================================
Interactive what-if simulation for hospital main executives (CEO/COO/CFO).

Performance notes (why this version is fast):
  * The event log is trimmed to the last N days of activity on load (N chosen
    by the user in the sidebar), and the trimmed result is cached
    (st.cache_data) so it is only computed once per uploaded file + N combination.
  * All per-patient "waiting proxy" calculations use a single vectorised
    groupby().shift() pass instead of a Python loop with per-row DataFrame
    filtering (the old approach was O(n_starts * n_events) and took several
    seconds even on a 1-week slice; the new approach is O(n log n) and runs
    in milliseconds).
  * Routing / referral extraction uses vectorised string ops (str.extract +
    str.endswith masks) instead of iterrows().
  * The expensive analytical step (build_metrics) is wrapped in
    st.cache_data, so moving a capacity or demand slider — which only
    rescales already-computed numbers — never re-parses the log.

streamlit run hrtn_app.py
"""

from pathlib import Path
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="HRTN — Strategic Hospital Operations Dashboard",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------
# BRANDING
# ---------------------------------------------------------------------
col_logo_left, col_title, col_logo_right = st.columns([1, 6, 1])

with col_logo_left:
    logo_left = "../figs/Logo_UFMG.png"
    if Path(logo_left).exists():
        st.image(logo_left, width=180)
    else:
        st.markdown(
            "<div style='text-align:center;color:gray;font-size:12px;'>[ Logo UFMG ]</div>",
            unsafe_allow_html=True,
        )

with col_title:
    st.markdown(
        "<h1 style='text-align:center;font-size:27px;margin-bottom:0;'>"
        "HRTN — Painel de Operações Estratégicas Hospitalares"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#666;font-size:13px;margin-top:5px;'>"
        "Análise de cenários sobre demanda, capacidade, disputa por recursos e efeitos em cadeia "
        "em toda a unidade hospitalar — janela de análise configurável na barra lateral."
        "</p>"
        "<p style='text-align: center; color: #333; font-size: 14px; margin-top: 10px;'>"
        "<strong>Equipe</strong><br>"
        "Prof. João Flávio F. Almeida &lt;joao.flavio@dep.ufmg.br&gt; (DEP-UFMG)<br>"
        "Prof. Noel Torres Júnior &lt;noelface@gmail.com&gt; (DEP-UFMG)<br>"
        "Rogério Moraes Brito &lt;rogermorr4@gmail.com&gt; (DEP-UFMG)<br>"
        "Samuel Mol Lima &lt;samuellima2187@gmail.com&gt; (DEP-UFMG)<br>"
        "Sabrina Andrade &lt;sabrina.andrade@hrtn.fundep.ufmg.br&gt; (Hospital Risoleta Neves)"
        "</p>",
        unsafe_allow_html=True,
    )

with col_logo_right:
    logo_right = "../figs/logo_risoleta.png"
    if Path(logo_right).exists():
        st.image(logo_right, width=180)
    else:
        st.markdown(
            "<div style='text-align:center;color:gray;font-size:12px;'>[ Logo HRTN ]</div>",
            unsafe_allow_html=True,
        )

st.divider()


# ---------------------------------------------------------------------
# DEPARTMENT / RESOURCE DEFINITION
# ---------------------------------------------------------------------
DEPARTMENTS = ["PS", "MAT", "CC", "CLIC", "CLIM", "CTI", "SRPA", "AMB"]

DEPT_FULL_NAME = {
    "PS": "PS — Pronto Socorro (Emergency Department)",
    "MAT": "MAT — Maternidade (Maternity)",
    "CC": "CC — Centro Cirúrgico (Surgery Center / ORs)",
    "CLIC": "CLIC — Clínica Cirúrgica (Surgical Ward)",
    "CLIM": "CLIM — Clínica Médica (Medical Ward)",
    "CTI": "CTI — Centro de Terapia Intensiva (ICU)",
    "SRPA": "SRPA — Sala de Recuperação Pós-Anestésica (Recovery Room)",
    "AMB": "AMB — Ambulatório (Outpatient Clinic)",
}

# Portuguese-only display name (no English translation) used for the
# "Sigla: Nome" formatting in the Capacity Matrix table.
DEPT_PT_NAME = {
    "PS": "Pronto Socorro",
    "MAT": "Maternidade",
    "CC": "Centro Cirúrgico",
    "CLIC": "Clínica Cirúrgica",
    "CLIM": "Clínica Médica",
    "CTI": "Centro de Terapia Intensiva",
    "SRPA": "Sala de Recuperação Pós-Anestésica",
    "AMB": "Ambulatório",
}

RESOURCE_MAP = {
    "PS": "psBeds",
    "MAT": "matBeds",
    "CC": "CC_Rooms",
    "CLIC": "clicBeds",
    "CLIM": "climBeds",
    "CTI": "ctiBeds",
    "SRPA": "srpaBeds",
    "AMB": "ambBeds",
}

# Initial capacities are the hospital configuration represented by the
# supplied event log. They can be changed interactively.
DEFAULT_CAPACITY = {
    "PS": 110, # 51+59=110
    "MAT": 30, # 41
    "CC": 5,
    "CLIC": 95,
    "CLIM": 164,
    "CTI": 31,
    "SRPA": 13,
    "AMB": 5,  # Espaços de consulta
}

# Routing text present in the event log's *_Encaminha_para_* activities.
ROUTE_TEXT_TO_DEPT = {
    "Pronto Socorro": "PS",
    "Maternidade": "MAT",
    "Centro Cirurgico": "CC",
    "Centro Cirúrgico": "CC",
    "a Clinica Cirurgica": "CLIC",
    "a Clínica Cirúrgica": "CLIC",
    "a Clinica Medica": "CLIM",
    "a Clínica Médica": "CLIM",
    "o CTI": "CTI",
    "o Ambulatorio": "AMB",
    "o Ambulatório": "AMB",
}

# Service activity used to measure resource occupation per department.
# The model was simplified: CC (surgery) and SRPA (recovery) are now
# ordinary ProcessBlocks, logged exactly like every other department
# (a single "start"/"complete" pair per activity name) — no more
# container-based CC_Seize_OR / CC_Recover_Release_SRPA naming.
SERVICE_ACTIVITY = {
    "PS": "PS",
    "MAT": "MAT",
    "CLIM": "CLIM",
    "CLIC": "CLIC",
    "CTI": "CTI",
    "AMB": "AMB",
    "CC": "CC",      # OR occupation
    "SRPA": "SRPA",  # recovery-room occupation
}


# ---------------------------------------------------------------------
# DATA LOADING  (cached — runs once per uploaded file + n_days)
# ---------------------------------------------------------------------
def _reduce_to_last_n_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Keep only the last n_days of simulated activity.

    Complete patient journeys are preserved: any case with at least one
    event inside the window keeps *all* of its events, so length-of-stay
    and routing calculations stay consistent.
    """
    if df.empty or n_days <= 0:
        return df

    window_minutes = n_days * 1440
    max_ts = df["timestamp"].max()
    min_ts = max_ts - window_minutes

    if df["timestamp"].min() >= min_ts:
        return df

    case_ids_in_window = df.loc[df["timestamp"] >= min_ts, "case_id"].unique()
    return df[df["case_id"].isin(case_ids_in_window)].copy()


@st.cache_data(show_spinner="Carregando event log...")
def load_log(file_bytes: bytes, n_days: int = 7) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))

    required = {"case_id", "activity", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(sorted(missing)))

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["case_id", "activity", "timestamp"])
    df["case_id"] = df["case_id"].astype(str)
    df["activity"] = df["activity"].astype(str)

    if "lifecycle" not in df.columns:
        df["lifecycle"] = np.nan
    if "resource" not in df.columns:
        df["resource"] = np.nan

    df = df.sort_values(["case_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    df = _reduce_to_last_n_days(df, n_days)

    # Vectorised origin tag (no per-row python function calls).
    df["origin"] = np.select(
        [df["case_id"].str.startswith("PS_"), df["case_id"].str.startswith("MAT_")],
        ["PS", "MAT"],
        default="OTHER",
    )

    # Vectorised "previous event in the same case" wait proxy — replaces the
    # old O(n_starts * n_events) row-by-row scan.
    df["prev_timestamp"] = df.groupby("case_id")["timestamp"].shift(1)
    df["wait_from_prev"] = (df["timestamp"] - df["prev_timestamp"]).clip(lower=0).fillna(0.0)

    return df


# ---------------------------------------------------------------------
# ANALYTICS  (vectorised)
# ---------------------------------------------------------------------
def _pair_events(df: pd.DataFrame, start_activity: str, end_activity: str | None = None,
                  end_lifecycle: str = "complete") -> pd.DataFrame:
    """Pair start/complete events for an activity (or two related activities)
    by case_id and occurrence order. Fully vectorised via merge — no loops.
    """
    if end_activity is None:
        sub = df[df["activity"] == start_activity]
        starts = sub[sub["lifecycle"].eq("start")][["case_id", "timestamp", "origin"]]
        ends = sub[sub["lifecycle"].eq(end_lifecycle)][["case_id", "timestamp"]]
    else:
        starts = df[df["activity"] == start_activity][["case_id", "timestamp", "origin"]]
        ends = df[df["activity"] == end_activity][["case_id", "timestamp"]]

    if starts.empty or ends.empty:
        return pd.DataFrame(columns=["case_id", "start_time", "end_time", "duration_min", "origin"])

    starts = starts.copy()
    ends = ends.copy()
    starts["occ"] = starts.groupby("case_id").cumcount()
    ends["occ"] = ends.groupby("case_id").cumcount()

    pairs = starts.merge(ends, on=["case_id", "occ"], how="inner", suffixes=("_start", "_end"))
    pairs["duration_min"] = (pairs["timestamp_end"] - pairs["timestamp_start"]).clip(lower=0)
    pairs = pairs.rename(columns={"timestamp_start": "start_time", "timestamp_end": "end_time"})
    return pairs[["case_id", "start_time", "end_time", "duration_min", "origin"]]


def _extract_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised extraction of every patient-flow edge in the log:
      * Chegada (Arrival)  → PS / MAT   — the hospital's two entry points.
      * department → department          — internal referrals
        (e.g. "PS_Encaminha_para a Clinica Medica" — note the space, not
        an underscore, between "para" and the destination article).
      * department → Saída (Discharge)   — every terminal outcome (alta
        hospitalar, óbito, transferência externa, desistência/evasão) is
        folded into a single "Saída" node representing the patient leaving
        the hospital.
    """
    # --- 1. Arrivals: Chegada -> PS / MAT --------------------------------
    arrivals = df.loc[df["activity"].eq("Arrival") & df["origin"].isin(["PS", "MAT"]), ["case_id", "origin"]]
    arrival_flows = (
        arrivals.groupby("origin", as_index=False)
        .agg(flow=("case_id", "nunique"))
        .rename(columns={"origin": "target"})
    )
    if not arrival_flows.empty:
        arrival_flows["source"] = "Chegada"
        arrival_flows["origin"] = arrival_flows["target"]
        arrival_flows = arrival_flows[["source", "target", "origin", "flow"]]

    # --- 2. Referral / outcome events -------------------------------------
    mask = df["activity"].str.contains("_Encaminha_", regex=False, na=False)
    sub = df.loc[mask, ["case_id", "activity", "origin"]].copy()

    if sub.empty:
        internal_exit_flows = pd.DataFrame(columns=["source", "target", "origin", "flow"])
    else:
        extracted = sub["activity"].str.extract(r"^(PS|MAT|CC|CLIC|CLIM|CTI|AMB)_Encaminha_(.+)$")
        sub["source"] = extracted[0]
        sub["rest"] = extracted[1].str.strip()
        sub = sub.dropna(subset=["source", "rest"])
        sub = sub[sub["source"].isin(DEPARTMENTS)]

        # 2a. Internal referrals: "...Encaminha_para <destination>"
        is_internal = sub["rest"].str.startswith("para ")
        internal = sub[is_internal].copy()
        internal["dest_text"] = internal["rest"].str[len("para "):].str.strip()
        internal["target"] = pd.Series(pd.NA, index=internal.index, dtype="object")
        for key, dept in ROUTE_TEXT_TO_DEPT.items():
            m = internal["target"].isna() & (
                internal["dest_text"].eq(key) | internal["dest_text"].str.endswith(key)
            )
            internal.loc[m, "target"] = dept
        internal = internal.dropna(subset=["target"])

        # 2b. Terminal outcomes -> Saída (discharge, death, external
        # transfer, or the patient leaving against medical advice).
        exit_labels = {"Alta Hospitalar", "Obito", "Desistencia/Evasao", "Transferencia Externa"}
        exits = sub[sub["rest"].isin(exit_labels)].copy()
        exits["target"] = "Saída"

        internal_exit_flows = pd.concat(
            [internal[["source", "target", "origin", "case_id"]], exits[["source", "target", "origin", "case_id"]]],
            ignore_index=True,
        )
        if internal_exit_flows.empty:
            internal_exit_flows = pd.DataFrame(columns=["source", "target", "origin", "flow"])
        else:
            internal_exit_flows = (
                internal_exit_flows.groupby(["source", "target", "origin"], as_index=False)
                .agg(flow=("case_id", "nunique"))
            )

    flows = pd.concat([arrival_flows, internal_exit_flows], ignore_index=True)
    if flows.empty:
        return pd.DataFrame(columns=["source", "target", "origin", "flow"])
    return flows


@st.cache_data(show_spinner="Calculando métricas do hospital...")
def build_metrics(df: pd.DataFrame, n_days: int = 7) -> dict:
    window_minutes = n_days * 1440
    max_ts = float(df["timestamp"].max()) if not df.empty else 0.0
    window_start = max(max_ts - window_minutes, float(df["timestamp"].min()) if not df.empty else 0.0)
    sim_days = max(max_ts / 1440.0, 1.0) if max_ts <= window_minutes else float(n_days)

    # Baseline patient counts by origin.
    arrivals = df[df["activity"] == "Arrival"]
    origin_cases = (
        arrivals[arrivals["origin"].isin(["PS", "MAT"])]
        .groupby("origin")["case_id"]
        .nunique()
        .to_dict()
    )

    # Service occupancy per department. CC and SRPA are now plain
    # ProcessBlocks (start/complete pair on their own activity name),
    # exactly like every other department — no special-casing needed.
    pairs_by_dept = {}
    for dept, activity in SERVICE_ACTIVITY.items():
        pairs_by_dept[dept] = _pair_events(df, activity)

    # Baseline workload in resource-minutes, split by patient origin.
    rows = []
    for dept in DEPARTMENTS:
        p = pairs_by_dept[dept]
        for origin in ["PS", "MAT"]:
            q = p[p["origin"] == origin] if not p.empty else p
            rows.append(
                {
                    "department": dept,
                    "origin": origin,
                    "workload_min": float(q["duration_min"].sum()) if not q.empty else 0.0,
                    "patients": int(q["case_id"].nunique()) if not q.empty else 0,
                }
            )
    workload = pd.DataFrame(rows)

    # Department-level daily workload (for the per-process "deep dive" charts).
    daily_rows = []
    for dept in DEPARTMENTS:
        p = pairs_by_dept[dept]
        if p.empty:
            continue
        # Only count occupations that *started* inside the analysis window,
        # so a long stay that began before the window doesn't spuriously
        # inflate "day 1" (avoids a distorted daily-trend chart).
        d = p[p["start_time"] >= window_start].copy()
        if d.empty:
            continue
        d["day_idx"] = np.clip(
            np.floor((d["start_time"] - window_start) / 1440.0).astype(int), 0, max(n_days - 1, 0)
        )
        agg = (
            d.groupby("day_idx", as_index=False)
            .agg(workload_min=("duration_min", "sum"), patients=("case_id", "nunique"))
        )
        agg["department"] = dept
        daily_rows.append(agg)
    daily_workload = (
        pd.concat(daily_rows, ignore_index=True)
        if daily_rows
        else pd.DataFrame(columns=["day_idx", "workload_min", "patients", "department"])
    )

    flows = _extract_flows(df)

    # Waiting proxy per department (vectorised — built directly from the
    # wait_from_prev column computed once in load_log).
    queue_rows = []
    for dept, activity in SERVICE_ACTIVITY.items():
        starts = df.loc[
            (df["activity"] == activity) & (df["lifecycle"] == "start"),
            ["case_id", "origin", "wait_from_prev"],
        ]
        if starts.empty:
            continue
        queue_rows.append(starts.assign(department=dept).rename(columns={"wait_from_prev": "wait_proxy"}))

    waits_df = (
        pd.concat(queue_rows, ignore_index=True)
        if queue_rows
        else pd.DataFrame(columns=["case_id", "origin", "wait_proxy", "department"])
    )

    throughput = int(df.loc[df["activity"].eq("Discharge"), "case_id"].nunique())

    return {
        "simulation_days": sim_days,
        "window_start": window_start,
        "n_days": n_days,
        "arrivals": origin_cases,
        "workload": workload,
        "daily_workload": daily_workload,
        "flows": flows,
        "waits": waits_df,
        "throughput": throughput,
        "pairs": pairs_by_dept,
    }


def capacity_metrics(base: dict, capacity: dict, ps_multiplier: float, mat_multiplier: float) -> pd.DataFrame:
    """Counterfactual capacity / workload calculation.

    Rescales the observed event-log workload according to the selected
    PS/MAT origin multipliers and divides by the user-selected capacity.
    A nonlinear stress index represents queue pressure: below 70% it tracks
    the observed baseline; above 70% pressure grows progressively faster.
    This is a decision-support indicator, not a re-run DES simulation.

    Estimated time in process = service time + estimated queue time.
    The event log itself records almost no explicit queueing (patients are
    assigned a bed the instant one is requested), so a wait proxy built
    only from log gaps is close to zero everywhere and badly understates
    real pressure once a department is near or above capacity. Instead,
    queue time here is estimated from queueing theory as an ADDITIVE extra
    on top of the real average length of stay (service time):

        service_min = average length of stay actually observed in the dept.
        queue_min   = service_min * max(stress - 1, 0)
        total_min   = service_min + queue_min

    Below 70% utilization, stress <= 1 so queue_min = 0 and total_min
    equals the real observed service time (no phantom queue is invented).
    Above 70% utilization, stress > 1 and queue_min grows progressively
    faster as the department approaches/exceeds capacity. This keeps the
    "total time" honestly decomposable into a real, observed part (service)
    and an estimated part (queue), instead of silently multiplying the
    service time by a factor that can also shrink it.
    """
    scale = {"PS": ps_multiplier, "MAT": mat_multiplier}
    days = base["simulation_days"]
    wl = base["workload"]

    rows = []
    for dept in DEPARTMENTS:
        dept_wl = wl[wl["department"] == dept]
        workload_min = 0.0
        for origin in ["PS", "MAT"]:
            x = dept_wl.loc[dept_wl["origin"] == origin, "workload_min"].sum()
            workload_min += float(x) * scale[origin]

        cap_min = max(float(capacity[dept]) * days * 1440.0, 1.0)
        utilization = workload_min / cap_min * 100.0

        stress = utilization / 70.0 if utilization <= 70 else 1.0 + ((utilization - 70.0) / 30.0) ** 2

        baseline_dept = dept_wl["workload_min"].sum()
        baseline_cap = max(DEFAULT_CAPACITY[dept] * days * 1440.0, 1.0)
        baseline_util = baseline_dept / baseline_cap * 100.0

        pairs = base["pairs"].get(dept)
        avg_service_min = (
            float(pairs["duration_min"].mean()) if pairs is not None and not pairs.empty else 0.0
        )
        # Queue time is an ADDITIVE extra on top of the real service time —
        # zero whenever utilization is at/below the 70% comfort threshold,
        # growing only once the department is under real pressure.
        avg_queue_min = avg_service_min * max(stress - 1.0, 0.0)
        total_time_min = avg_service_min + avg_queue_min

        rows.append(
            {
                "department": dept,
                "capacity": int(capacity[dept]),
                "workload_hours": workload_min / 60.0,
                "utilization": utilization,
                "baseline_utilization": baseline_util,
                "avg_wait_proxy_min": total_time_min,  # kept for backward-compat (KPIs/sorting)
                "avg_service_min": avg_service_min,
                "avg_queue_min": avg_queue_min,
                "stress": stress,
            }
        )

    return pd.DataFrame(rows)


def counterfactual_flows(base: dict, ps_multiplier: float, mat_multiplier: float) -> pd.DataFrame:
    flows = base["flows"].copy()
    if flows.empty:
        return flows
    flows["multiplier"] = np.where(flows["origin"].eq("PS"), ps_multiplier, mat_multiplier)
    flows["flow_scenario"] = flows["flow"] * flows["multiplier"]
    return flows


def scaled_daily_workload(base: dict, dept: str, ps_multiplier: float, mat_multiplier: float) -> pd.DataFrame:
    """Daily workload for one department, rescaled by origin multipliers so
    the per-department deep-dive reflects the current what-if scenario
    (propagating the 'domino effect' of PS/MAT sliders into every tab).
    """
    wl = base["workload"]
    dept_wl = wl[wl["department"] == dept]
    ps_share = dept_wl.loc[dept_wl["origin"] == "PS", "workload_min"].sum()
    mat_share = dept_wl.loc[dept_wl["origin"] == "MAT", "workload_min"].sum()
    total = ps_share + mat_share
    # Blend factor: weight PS/MAT multipliers by each origin's historical
    # share of this department's workload (other-origin patients unscaled).
    if total > 0:
        blended = (ps_share * ps_multiplier + mat_share * mat_multiplier) / total
    else:
        blended = 1.0

    d = base["daily_workload"]
    d = d[d["department"] == dept].copy()
    if d.empty:
        return d
    d["workload_min_scenario"] = d["workload_min"] * blended
    d["patients_scenario"] = (d["patients"] * blended).round().astype(int)
    return d


# ---------------------------------------------------------------------
# SIDEBAR — INPUTS
# ---------------------------------------------------------------------
st.sidebar.header("Cenário estratégico")

# Controle dinâmico da janela de análise
n_days = st.sidebar.slider(
    "Janela de análise (dias)",
    min_value=1,
    max_value=365,
    value=7,
    step=1,
    help="Quantidade de dias finais do event log a serem considerados na análise. "
         "Jornadas completas de pacientes que tocam a janela são preservadas.",
)

uploaded = st.sidebar.file_uploader(
    "Upload do event log hospitalar",
    type="csv",
    help="CSV com pelo menos case_id, activity e timestamp.",
)

default_log = Path("hrtn_event_log.csv")

if uploaded is not None:
    source_bytes = uploaded.getvalue()
    source_label = f"arquivo enviado: {uploaded.name}"
elif default_log.exists():
    source_bytes = default_log.read_bytes()
    source_label = "hrtn_event_log.csv"
else:
    st.error("Nenhum event log encontrado. Envie um CSV contendo `case_id`, `activity` e `timestamp`.")
    st.stop()

try:
    df = load_log(source_bytes, n_days=n_days)
    base = build_metrics(df, n_days=n_days)
except Exception as exc:
    st.error(f"Não foi possível processar o event log: {exc}")
    st.stop()

st.sidebar.caption(
    f"Fonte: **{source_label}** · "
    f"{df['case_id'].nunique():,} pacientes · "
    f"últimos {base['simulation_days']:.1f} dias de atividade"
)

st.sidebar.divider()
st.sidebar.subheader("1. Origens — demanda")

# ---------------------------------------------------------------------
# Sincronização PS/MAT entre o slider da barra lateral e o slider-atalho
# exibido em cima de cada gráfico de sensibilidade (seção 2, mais abaixo).
# Os dois controles de cada origem compartilham o mesmo percentual: mover
# qualquer um dos dois atualiza o outro no próximo rerun, via callbacks
# que copiam o valor entre as chaves de session_state antes do script
# rodar novamente.
# ---------------------------------------------------------------------
def _sync_ps_from_sidebar():
    st.session_state["ps_pct_top"] = st.session_state["ps_pct_sidebar"]


def _sync_ps_from_top():
    st.session_state["ps_pct_sidebar"] = st.session_state["ps_pct_top"]


def _sync_mat_from_sidebar():
    st.session_state["mat_pct_top"] = st.session_state["mat_pct_sidebar"]


def _sync_mat_from_top():
    st.session_state["mat_pct_sidebar"] = st.session_state["mat_pct_top"]


ps_pct = st.sidebar.slider(
    "PS — variação da chegada (%)", min_value=-50, max_value=100,
    value=st.session_state.get("ps_pct_sidebar", 0), step=5,
    format="%d%%", key="ps_pct_sidebar", on_change=_sync_ps_from_sidebar,
    help="Varia proporcionalmente o workload observado dos pacientes originados no PS. "
         "Também disponível como atalho em cima do gráfico de sensibilidade do PS.",
)
mat_pct = st.sidebar.slider(
    "MAT — variação da chegada (%)", min_value=-50, max_value=100,
    value=st.session_state.get("mat_pct_sidebar", 0), step=5,
    format="%d%%", key="mat_pct_sidebar", on_change=_sync_mat_from_sidebar,
    help="Varia proporcionalmente o workload observado dos pacientes originados na MAT. "
         "Também disponível como atalho em cima do gráfico de sensibilidade da MAT.",
)

ps_multiplier = 1.0 + ps_pct / 100.0
mat_multiplier = 1.0 + mat_pct / 100.0

st.sidebar.divider()
st.sidebar.subheader("2. Capacidade instalada")

capacity = {}
for dept in DEPARTMENTS:
    resource = RESOURCE_MAP[dept]
    default = DEFAULT_CAPACITY[dept]
    if dept == "CC":
        capacity[dept] = st.sidebar.slider(
            "CC — Salas cirúrgicas", min_value=1, max_value=15, value=default, step=1,
            help=f"Recurso do event log: {resource}.",
        )
    elif dept == "AMB":
        capacity[dept] = st.sidebar.slider(
            "AMB — Salas de consulta", min_value=1, max_value=15, value=default, step=1,
            help=f"Recurso do event log: {resource}.",
        )
    else:
        capacity[dept] = st.sidebar.slider(
            f"{dept} — Leitos", min_value=1, max_value=max(default * 2, 10), value=default, step=1,
            help=f"Recurso do event log: {resource}.",
        )

st.sidebar.divider()
st.sidebar.info(
    "O cenário simulado é calculado a partir dos tempos observados no "
    "log de eventos da simulação (janela configurável). Não são executadas novas simulações DES."
)


# ---------------------------------------------------------------------
# SCENARIO CALCULATION
# ---------------------------------------------------------------------
metrics = capacity_metrics(base, capacity, ps_multiplier, mat_multiplier)
flows_scenario = counterfactual_flows(base, ps_multiplier, mat_multiplier)

demand_multiplier_weighted = (
    base["arrivals"].get("PS", 0) * ps_multiplier + base["arrivals"].get("MAT", 0) * mat_multiplier
) / max(sum(base["arrivals"].values()), 1)

max_util = metrics["utilization"].max()
throughput_indicator = base["throughput"] * demand_multiplier_weighted
if max_util > 100:
    throughput_indicator = throughput_indicator * (100.0 / max_util)
throughput_indicator = max(0.0, throughput_indicator)

bottleneck_util = metrics.loc[metrics["utilization"].idxmax()]
bottleneck_wait = metrics.loc[metrics["avg_wait_proxy_min"].idxmax()]


# ---------------------------------------------------------------------
# TOP KPIs
# ---------------------------------------------------------------------
st.header("Painel Executivo (estratégico)")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Fluxo no período",
    f"{throughput_indicator:,.0f} pacientes",
    delta=f"{(demand_multiplier_weighted - 1) * 100:+.1f}% demanda",
)
k2.metric(
    "Maior utilização",
    f"{bottleneck_util['department']} · {bottleneck_util['utilization']:.1f}%",
    delta=f"{bottleneck_util['utilization'] - bottleneck_util['baseline_utilization']:+.1f} p.p.",
    delta_color="inverse",
)
k3.metric(
    "Maior tempo estimado no processo",
    f"{bottleneck_wait['department']} · {bottleneck_wait['avg_wait_proxy_min'] / 60.0:.1f} h",
)
k4.metric(
    "Casos de entrada",
    f"PS: {base['arrivals'].get('PS', 0):,} · MAT: {base['arrivals'].get('MAT', 0):,}",
    delta=f"PS {ps_pct:+d}% · MAT {mat_pct:+d}%",
)

if max_util > 100:
    st.error(
        f"⚠️ **Gargalo crítico:** {bottleneck_util['department']} atinge "
        f"{bottleneck_util['utilization']:.1f}% da capacidade teórica. A demanda excede a capacidade instalada."
    )
elif max_util > 85:
    st.warning(
        f"⚠️ **Zona de alta ocupação:** {bottleneck_util['department']} atinge "
        f"{bottleneck_util['utilization']:.1f}%. A folga operacional é reduzida."
    )
else:
    st.success(
        f"✅ **Capacidade global controlada:** maior utilização = "
        f"{bottleneck_util['department']} ({bottleneck_util['utilization']:.1f}%)."
    )

st.divider()


# ---------------------------------------------------------------------
# 1. RESOURCE UTILIZATION (enterprise-wide)
# ---------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    plot_df = metrics.sort_values("utilization", ascending=True)
    fig_util = go.Figure()
    fig_util.add_trace(
        go.Bar(
            y=plot_df["department"], x=plot_df["utilization"], orientation="h",
            text=[f"{x:.1f}%" for x in plot_df["utilization"]], textposition="outside",
            name="Cenário",
        )
    )
    fig_util.add_vline(x=70, line_dash="dot", annotation_text="70%")
    fig_util.add_vline(x=85, line_dash="dash", annotation_text="85%")
    fig_util.add_vline(x=100, line_dash="solid", annotation_text="100%")
    fig_util.update_layout(
        title="Utilização por departamento", xaxis_title="Utilização (%)", yaxis_title="Departamento",
        margin=dict(l=20, r=40, t=55, b=30),
    )
    st.plotly_chart(fig_util, use_container_width=True)

with c2:
    wait_df = metrics.copy()
    wait_df["service_h"] = wait_df["avg_service_min"] / 60.0
    wait_df["queue_h"] = wait_df["avg_queue_min"] / 60.0
    wait_df["total_h"] = wait_df["service_h"] + wait_df["queue_h"]
    wait_df = wait_df.sort_values("total_h", ascending=True)

    fig_wait = go.Figure()
    fig_wait.add_trace(
        go.Bar(
            y=wait_df["department"], x=wait_df["service_h"], orientation="h",
            name="Tempo estimado no atendimento",
            marker_color="#1f77b4",
            text=[f"{x:.1f} h" for x in wait_df["service_h"]], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="white"),
            hovertemplate="Atendimento: %{x:.1f} h<extra></extra>",
        )
    )
    fig_wait.add_trace(
        go.Bar(
            y=wait_df["department"], x=wait_df["queue_h"], orientation="h",
            name="Tempo estimado em fila",
            marker_color="#B0B0B0",
            text=[f"{x:.1f} h" if x > 0.05 else "" for x in wait_df["queue_h"]], textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Fila estimada: %{x:.1f} h<extra></extra>",
        )
    )
    # Explicit total-hours label at the end of each stacked bar.
    fig_wait.add_trace(
        go.Scatter(
            y=wait_df["department"], x=wait_df["total_h"],
            mode="text",
            text=[f"{x:.1f} h total" for x in wait_df["total_h"]],
            textposition="middle right",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_wait.update_layout(
        barmode="stack",
        title="Tempo estimado no processo — atendimento + fila (horas)",
        xaxis_title="Horas", yaxis_title="Departamento",
        margin=dict(l=20, r=90, t=55, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_wait.update_xaxes(range=[0, wait_df["total_h"].max() * 1.2])
    st.plotly_chart(fig_wait, use_container_width=True)

st.caption(
    "A utilização é calculada diretamente do tempo observado no log de eventos e da capacidade "
    "selecionada. O tempo estimado no processo soma duas partes: o tempo de atendimento (permanência "
    "média realmente observada no departamento, em azul) e uma fila estimada (em cinza), que só existe "
    "quando o departamento passa de ~70% de ocupação — abaixo disso, a barra cinza é zero e o total "
    "equivale ao tempo de atendimento real. Não representa uma nova execução de simulação (DES)."
)

st.divider()


# ---------------------------------------------------------------------
# Visualização do fluxo (HTML)
# ---------------------------------------------------------------------
st.subheader("Visualização do Fluxo Geral do Processo")
st.caption("Visualização interativa do fluxo da simulação.")

try:
    import streamlit.components.v1 as components
    with open("hrtn.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
except FileNotFoundError:
    st.info("Arquivo 'hrtn.html' não encontrado. Coloque o arquivo HTML na mesma pasta para exibi-lo aqui.")

st.divider()

# ---------------------------------------------------------------------
# 2. WHAT-IF: ORIGIN DEMAND SENSITIVITY
# ---------------------------------------------------------------------
st.subheader("📈 Sensibilidade das duas portas de entrada")
st.caption(
    "PS e MAT são prédios/unidades distintos e não compartilham capacidade — por isso cada porta de "
    "entrada tem seu próprio gráfico, mostrando o impacto da variação de demanda **apenas sobre a "
    "utilização do seu próprio departamento** (PS → utilização do PS; MAT → utilização da MAT)."
)


def _nearest(sweep_values, value):
    """Ponto da varredura mais próximo do valor atual do slider, para sempre
    destacar um marcador mesmo quando o slider não bate exatamente com um
    dos valores fixos do eixo x."""
    return min(sweep_values, key=lambda x: abs(x - value))


sweep = [-50, -25, 0, 25, 50, 75, 100]


def _origin_sensitivity_detail(origin: str) -> pd.DataFrame:
    baseline_arrivals = base["arrivals"].get(origin, 0)
    sim_days = base["simulation_days"] if base["simulation_days"] > 0 else 1.0
    per_day_baseline = baseline_arrivals / sim_days

    rows = []
    for pct in sweep:
        psm = (1 + pct / 100) if origin == "PS" else ps_multiplier
        matm = (1 + pct / 100) if origin == "MAT" else mat_multiplier
        mm = capacity_metrics(base, capacity, psm, matm)
        row = mm.loc[mm["department"] == origin].iloc[0]
        rows.append(
            {
                "variação": pct,
                "utilização": float(row["utilization"]),
                "avg_total_h": float(row["avg_wait_proxy_min"]) / 60.0,
                "pacientes_dia": per_day_baseline * (1 + pct / 100.0),
            }
        )
    return pd.DataFrame(rows)


def _build_sensitivity_figure(df: pd.DataFrame, current_pct: float, dept_label: str) -> go.Figure:
    marker_pct = _nearest(sweep, current_pct)
    x_vals = [f"{p:+d}%" for p in df["variação"]]

    normal_x, normal_y, normal_text = [], [], []
    hi_x, hi_y, hi_text = [], [], []

    for p, x_str, u, th, pd_ in zip(
        df["variação"], x_vals, df["utilização"], df["avg_total_h"], df["pacientes_dia"]
    ):
        if p == marker_pct:
            hi_x.append(x_str)
            hi_y.append(u)
            hi_text.append(
                f"<b>{u:.0f}%</b><br>{pd_:.1f} pacientes/dia<br>{th:.1f} h/paciente no período"
            )
        else:
            normal_x.append(x_str)
            normal_y.append(u)
            normal_text.append(f"{u:.0f}%<br>{pd_:.1f} pac./dia<br>{th:.1f} h/paciente")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=df["utilização"], mode="lines",
            line=dict(width=3, color="#1f77b4"), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=normal_x, y=normal_y, mode="markers+text",
            text=normal_text, textposition="bottom center",
            marker=dict(size=10, color="#1f77b4", line=dict(width=1, color="white")),
            textfont=dict(size=10, color="#333333"), showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hi_x, y=hi_y, mode="markers+text",
            text=hi_text, textposition="top center",
            marker=dict(size=18, color="#ff7f0e", line=dict(width=2, color="white")),
            textfont=dict(size=12, color="#d9381e", family="Arial Black"), showlegend=False,
        )
    )
    fig.add_hline(y=85, line_dash="dash", annotation_text="Zona de atenção")
    fig.add_hline(y=100, line_dash="solid", line_color="crimson", annotation_text="Capacidade")
    fig.update_layout(
        title=f"{dept_label} — utilização vs. variação da demanda",
        xaxis_title="Variação da demanda (%)", yaxis_title="Utilização (%)",
        margin=dict(t=55, b=30),
    )
    y_top = max(150, float(df["utilização"].max()) * 1.15)
    fig.update_yaxes(range=[0, y_top])
    return fig, marker_pct


sens_ps = _origin_sensitivity_detail("PS")
sens_mat = _origin_sensitivity_detail("MAT")

sens_col1, sens_col2 = st.columns(2)

with sens_col1:
    st.slider(
        "🎚️ PS — variação da chegada (%)", min_value=-50, max_value=100,
        value=st.session_state.get("ps_pct_top", ps_pct), step=5,
        format="%d%%", key="ps_pct_top", on_change=_sync_ps_from_top,
        help="Atalho para o mesmo parâmetro do slider 'PS — variação da chegada' na barra "
             "lateral: mover um dos dois atualiza o outro.",
    )
    fig_sens_ps, ps_marker = _build_sensitivity_figure(sens_ps, ps_pct, "PS")
    st.plotly_chart(fig_sens_ps, use_container_width=True)
    st.caption(
        f"🔍 O ponto laranja destaca o valor atual do slider PS (~{ps_marker:+d}%, mais próximo de "
        f"{ps_pct:+d}%), com o número absoluto de pacientes/dia e o tempo médio (h) por paciente no PS "
        f"nesse cenário."
    )

with sens_col2:
    st.slider(
        "🎚️ MAT — variação da chegada (%)", min_value=-50, max_value=100,
        value=st.session_state.get("mat_pct_top", mat_pct), step=5,
        format="%d%%", key="mat_pct_top", on_change=_sync_mat_from_top,
        help="Atalho para o mesmo parâmetro do slider 'MAT — variação da chegada' na barra "
             "lateral: mover um dos dois atualiza o outro.",
    )
    fig_sens_mat, mat_marker = _build_sensitivity_figure(sens_mat, mat_pct, "MAT")
    st.plotly_chart(fig_sens_mat, use_container_width=True)
    st.caption(
        f"🔍 O ponto laranja destaca o valor atual do slider MAT (~{mat_marker:+d}%, mais próximo de "
        f"{mat_pct:+d}%), com o número absoluto de pacientes/dia e o tempo médio (h) por paciente na "
        f"MAT nesse cenário."
    )

st.caption(
    "Cada curva mantém a capacidade e a outra origem no valor atualmente selecionado. Os pontos azuis "
    "mostram utilização, pacientes/dia e tempo médio (h) por paciente em cada variação simulada; o "
    "ponto laranja é sempre a posição atual do slider correspondente. O tempo médio por paciente é o "
    "mesmo indicador de 'tempo estimado no processo' (atendimento + fila) usado no restante do "
    "dashboard: fica praticamente estável enquanto a utilização está abaixo de ~70% e cresce quando a "
    "demanda simulada empurra o departamento para a zona de saturação."
)

st.divider()


# ---------------------------------------------------------------------
# 3. DOMINO EFFECT / SANKEY
# ---------------------------------------------------------------------
st.subheader("🔀 Fluxos entre departamentos do HRTN")

if flows_scenario.empty:
    st.info("O event log não contém transições de encaminhamento suficientes para construir o Sankey.")
else:
    sankey = flows_scenario.groupby(["source", "target"], as_index=False)["flow_scenario"].sum()

    # "Chegada" (Arrival) feeds PS/MAT; every department can also flow into
    # "Saída" (Discharge) when a patient leaves the hospital. No explicit
    # color is forced on any node — Plotly's automatic per-node palette
    # (the same one used before) keeps each department its own distinct
    # color, and simply extends that same palette to the two new nodes.
    labels = ["Chegada"] + DEPARTMENTS + ["Saída"]
    index = {name: i for i, name in enumerate(labels)}

    sankey = sankey[sankey["source"].isin(index) & sankey["target"].isin(index)]
    sources = sankey["source"].map(index).tolist()
    targets = sankey["target"].map(index).tolist()
    values = sankey["flow_scenario"].clip(lower=0.0).tolist()

    fig_sankey = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=labels, pad=20, thickness=18),
            link=dict(source=sources, target=targets, value=values),
        )
    )
    fig_sankey.update_layout(
        title="Fluxo de pacientes: Chegada → departamentos → Saída",
        font_size=12, height=580, margin=dict(l=10, r=10, t=60, b=20),
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

st.caption(
    "O Gráfico inclui a Chegada (Arrival), que se divide entre PS e MAT — as duas portas de entrada do "
    "hospital — e a Saída (Discharge), que reúne todos os desfechos que tiram o paciente do hospital "
    "(alta hospitalar, óbito, transferência externa ou evasão). Observe que a maior parte dos pacientes "
    "que chegam pelo PS sai diretamente do hospital após o atendimento, sem passar por outro "
    "departamento. Alterar PS/MAT modifica proporcionalmente os fluxos associados à origem; alterar "
    "capacidade modifica a pressão sobre os recursos, não inventa novas rotas."
)

st.divider()


# ---------------------------------------------------------------------
# 4. PER-DEPARTMENT DEEP DIVE — at least two charts per process
# ---------------------------------------------------------------------
st.subheader("🏥 Análise de processos hospitalares do HRTN")
st.caption(
    f"Cada aba traz, no mínimo, dois gráficos dedicados ao processo: tendência diária de utilização "
    f"(escalada pelo cenário PS/MAT) e a distribuição do tempo de permanência observado no log "
    f"(janela de {n_days} dia{'s' if n_days != 1 else ''})."
)

dept_tabs = st.tabs(DEPARTMENTS)

for dept, tab in zip(DEPARTMENTS, dept_tabs):
    with tab:
        row = metrics[metrics["department"] == dept].iloc[0]
        pairs = base["pairs"][dept]
        daily = scaled_daily_workload(base, dept, ps_multiplier, mat_multiplier)

        st.markdown(f"#### {DEPT_FULL_NAME[dept]}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Utilização no cenário", f"{row['utilization']:.1f}%")
        m2.metric(
            "Permanência média",
            f"{pairs['duration_min'].mean() / 60:.1f} h" if not pairs.empty else "—",
        )
        m3.metric("Pacientes atendidos (janela)", f"{int(pairs['case_id'].nunique()):,}" if not pairs.empty else "0")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if daily.empty:
                st.info("Sem dados diários suficientes para este processo na janela analisada.")
            else:
                cap_min_day = max(float(capacity[dept]) * 1440.0, 1.0)
                daily = daily.sort_values("day_idx")
                daily["utilization_pct"] = daily["workload_min_scenario"] / cap_min_day * 100.0
                fig_daily = go.Figure()
                fig_daily.add_trace(
                    go.Scatter(
                        x=[f"Dia {int(d) + 1}" for d in daily["day_idx"]],
                        y=daily["utilization_pct"],
                        mode="lines+markers",
                        name="Utilização diária",
                        line=dict(width=3),
                    )
                )
                fig_daily.add_hline(y=85, line_dash="dash", annotation_text="85%")
                fig_daily.add_hline(y=100, line_dash="solid", annotation_text="100%")
                fig_daily.update_layout(
                    title=f"{dept} — utilização diária (cenário atual)",
                    xaxis_title=f"Dia (últimos {n_days})",
                    yaxis_title="Utilização (%)",
                    margin=dict(l=20, r=20, t=55, b=30), height=350,
                )
                st.plotly_chart(fig_daily, use_container_width=True)

        with chart_col2:
            if pairs.empty:
                st.info("Sem eventos de ocupação registrados para este processo na janela analisada.")
            else:
                fig_dur = px.histogram(
                    pairs, x=pairs["duration_min"] / 60.0, nbins=25,
                    labels={"x": "Duração (horas)"},
                    title=f"{dept} — distribuição do tempo de permanência",
                )
                fig_dur.update_layout(
                    xaxis_title="Duração (horas)", yaxis_title="Nº de pacientes",
                    margin=dict(l=20, r=20, t=55, b=30), height=350, showlegend=False,
                )
                median_h = pairs["duration_min"].median() / 60.0
                fig_dur.add_vline(x=median_h, line_dash="dash", annotation_text=f"mediana {median_h:.1f}h")
                st.plotly_chart(fig_dur, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            if daily.empty:
                st.empty()
            else:
                fig_vol = go.Figure(
                    go.Bar(
                        x=[f"Dia {int(d) + 1}" for d in daily["day_idx"]],
                        y=daily["patients_scenario"],
                        name="Pacientes/dia",
                    )
                )
                fig_vol.update_layout(
                    title=f"{dept} — volume de pacientes por dia (cenário)",
                    xaxis_title=f"Dia (últimos {n_days})",
                    yaxis_title="Pacientes",
                    margin=dict(l=20, r=20, t=55, b=30), height=320,
                )
                st.plotly_chart(fig_vol, use_container_width=True)

        with chart_col4:
            # Real, observed admission-time profile — replaces the previous
            # "espera observada" box plot (raw log gaps, near-zero and not
            # meaningful as a queue proxy). Hour-of-day admission volume is
            # a genuine hospital bed-management chart: it drives shift and
            # staffing decisions (when does this department actually get busy).
            window_start = base["window_start"]
            p_admit = pairs[pairs["start_time"] >= window_start] if not pairs.empty else pairs
            if p_admit.empty:
                p_admit = pairs
            if p_admit.empty:
                st.empty()
            else:
                hour_of_day = ((p_admit["start_time"] % 1440) / 60.0).astype(int)
                counts = hour_of_day.value_counts().reindex(range(24), fill_value=0).sort_index()
                fig_hour = go.Figure(
                    go.Bar(
                        x=[f"{h:02d}h" for h in counts.index], y=counts.values,
                        name="Admissões", marker_color="#2ca02c",
                    )
                )
                fig_hour.update_layout(
                    title=f"{dept} — perfil de admissões por hora do dia",
                    xaxis_title="Hora do dia", yaxis_title="Nº de admissões (janela)",
                    margin=dict(l=20, r=20, t=55, b=30), height=320, showlegend=False,
                )
                st.plotly_chart(fig_hour, use_container_width=True)

st.divider()


# ---------------------------------------------------------------------
# 5. RESOURCE / WORKLOAD TABLE
# ---------------------------------------------------------------------
st.subheader("📋 Matriz de Capacidade por Departamento no HRTN")

table = metrics.copy()
table["service_h"] = table["avg_service_min"] / 60.0
table["queue_h"] = table["avg_queue_min"] / 60.0
table["total_h"] = table["service_h"] + table["queue_h"]
table = table[
    ["department", "capacity", "workload_hours", "utilization", "service_h", "queue_h", "total_h"]
].copy()
table["department"] = table["department"].map(lambda d: f"{d}: {DEPT_PT_NAME[d]}")
table.columns = [
    "Departamento", "Capacidade (leitos)", "Capacidade (h)", "Utilização (%)",
    "Tempo de atendimento (h)", "Tempo estimado em fila (h)", "Tempo total no processo (h)",
]
table["Utilização (%)"] = table["Utilização (%)"].round(1)
table["Capacidade (h)"] = table["Capacidade (h)"].round(1)
table["Tempo de atendimento (h)"] = table["Tempo de atendimento (h)"].round(1)
table["Tempo estimado em fila (h)"] = table["Tempo estimado em fila (h)"].round(1)
table["Tempo total no processo (h)"] = table["Tempo total no processo (h)"].round(1)

st.dataframe(table, use_container_width=True, hide_index=True)
st.caption(
    "Tempo de atendimento (h): permanência média realmente observada no log para o departamento. "
    "Tempo estimado em fila (h): estimativa baseada em teoria de filas, existente apenas acima de "
    "~70% de ocupação — abaixo disso é sempre zero, pois o log praticamente não registra fila "
    "explícita (o leito é ocupado no instante em que é solicitado). Tempo total = soma das duas colunas."
)

st.divider()


# ---------------------------------------------------------------------
# 6. EXECUTIVE INTERPRETATION
# ---------------------------------------------------------------------
st.subheader("🎯 Apoio à Decisão Gerencial")

util_top = metrics.sort_values("utilization", ascending=False).head(3)
for _, row in util_top.iterrows():
    dept = row["department"]
    util = row["utilization"]
    if util > 100:
        st.error(f"**{dept}: capacidade excedida ({util:.1f}%).** Prioridade: ampliar capacidade, reduzir demanda ou redistribuir o fluxo.")
    elif util > 85:
        st.warning(f"**{dept}: alta utilização ({util:.1f}%).** Pequenas oscilações de demanda podem gerar pressão de fila.")
    else:
        st.info(f"**{dept}: utilização de {util:.1f}%.** Há margem relativa no cenário atual.")

st.caption(
    f"Nota metodológica: esta aplicação é uma camada de decisão sobre o event log fornecido "
    f"(últimos {n_days} dia{'s' if n_days != 1 else ''}, por desempenho). Ela não substitui uma nova simulação DES. "
    "As projeções de workload e fluxo são estimativas proporcionais aos padrões observados."
)



# Como conduzir a demonstração (sem expor o desenvolvimento)

# Apresente como um "simulador de cenários", não como um "app que analisamos dados". Ordem sugerida:

# Abra no cenário-base (sliders no default) e mostre os KPIs executivos — isso ancora a conversa em números reais e observados.
# Mexa primeiro na capacidade (seção 2 da barra lateral) — é o slider mais intuitivo para diretor: "e se eu tivesse 5 leitos a mais na CTI?"
# Depois mexa na demanda (PS/MAT %) — mostre o efeito em cascata (Sankey + gráficos de sensibilidade). É aqui que "clica" a ideia de efeito dominó entre departamentos.
# Termine na Matriz de Capacidade + Apoio à Decisão Gerencial — é o "resumo executivo" que fecha a reunião com uma recomendação objetiva.

# Evite abrir as abas de "deep dive por departamento" logo de cara — guarde-as para responder perguntas específicas que surgirem ("e a CLIM, como está?").

# Perguntas estratégicas que o dashboard responde

# Sobre capacidade instalada e investimento (sliders de capacidade)

# Se aumentarmos X leitos na CTI/CLIM/CC, a utilização cai para um nível seguro (<85%)?
# Qual departamento tem a menor folga hoje e deveria ser prioridade de expansão?
# Vale mais investir em leitos de enfermaria (CLIC/CLIM) ou em salas cirúrgicas (CC)? O impacto na utilização se compara?
# Existe capacidade ociosa em algum departamento que poderia ser realocada (ex.: AMB)?

# Sobre demanda e portas de entrada (sliders PS/MAT)

# Se o volume do Pronto Socorro crescer 20% (novo convênio, sazonalidade, epidemia), qual departamento quebra primeiro?
# Até que variação de demanda do PS/MAT a operação atual aguenta sem fila relevante?
# Um aumento de partos na Maternidade tem efeito cascata em quais outros setores?

# Sobre fluxo entre departamentos (Sankey / efeito dominó)

# Qual é o caminho mais comum do paciente dentro do hospital — e ele bate com o desenho assistencial que imaginávamos?
# Que proporção de pacientes do PS recebe alta direta vs. é internada em outro setor?
# Se um departamento giratório (ex. CC) travar, quais setores a jusante (SRPA, CTI) sentem o impacto primeiro?

# Sobre desempenho operacional e risco de fila

# Quais departamentos estão hoje em zona de atenção (>85%) ou já estourados (>100%)?
# Onde o tempo total no processo (atendimento + fila estimada) mais penaliza o paciente?
# Um pico de demanda pontual (ex. feriado, fim de semana) levaria algum setor à ruptura?

# Sobre priorização executiva (fechamento da reunião)

# Dado o cenário atual, qual é a ação nº 1 recomendada: ampliar capacidade, reduzir demanda ou redistribuir fluxo?
# Se o orçamento permitir apenas uma intervenção neste trimestre, qual departamento traz o maior retorno em redução de risco?
