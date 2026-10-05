import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# streamlit run cti_app.py

st.set_page_config(
    page_title="CTI: Dashboard de Sensibilidade",
    page_icon="🏥",
    layout="wide",
)

# Criação de 3 colunas para alinhar a logo esquerda, o título no centro e a logo direita
col_logo_esq, col_titulo, col_logo_dir = st.columns([1, 6, 1])

with col_logo_esq:
    # Substitua pelo caminho da sua logo esquerda (ex: URL, arquivo local .png/.jpg)
    # Dica: use o parâmetro width para ajustar o tamanho médio desejado (ex: 120 a 180 pixels)
    caminho_logo_esq = "../figs/Logo_UFMG.png"  # Ajuste para o seu arquivo ou URL
    if Path(caminho_logo_esq).exists():
        st.image(caminho_logo_esq, width=180)
    else:
        # Placeholder visual caso o arquivo ainda não esteja na pasta
        st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>[ Logo Esquerda ]</div>", unsafe_allow_html=True)

with col_titulo:
    st.markdown("<h1 style='text-align: center; font-size: 26px; margin-bottom: 0px;'>Utilização do CTI do Risoleta Neves — Análise de Sensibilidade</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 13px; margin-top: 5px;'>"
        "Painel interativo, calibrado com dados reais dos logs de simulação, para analisar: "
        "A utilização do CTI para diferentes números de leitos (31, 40 e 50),  "
        "reconstruída com detalhamento por fonte de chegada e análises de sensibilidade. "
        "Janela de análise configurável na barra lateral."
        "</p>"
        "<p style='text-align: center; color: #333; font-size: 14px; margin-top: 10px;'>"
        "<strong>Equipe</strong><br>"
        "Prof. João Flávio F. Almeida &lt;joao.flavio@dep.ufmg.br&gt; (DEP-UFMG)<br>"
        "Prof. Noel Torres Júnior &lt;noelface@gmail.com&gt; (DEP-UFMG)<br>"
        "Rogério Moraes Brito &lt;rogermorr4@gmail.com&gt; (DEP-UFMG)<br>"
        "Samuel Mol Lima &lt;samuellima2187@gmail.com&gt; (DEP-UFMG)<br>"
        "Sabrina Andrade &lt;sabrina.andrade@hrtn.fundep.ufmg.br&gt; (Hospital Risoleta Neves)"
        "</p>",

        unsafe_allow_html=True
    )

with col_logo_dir:
    # Substitua pelo caminho da sua logo direita
    caminho_logo_dir = "../figs/logo_risoleta.png"  # Ajuste para o seu arquivo ou URL
    if Path(caminho_logo_dir).exists():
        st.image(caminho_logo_dir, width=180)
    else:
        st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>[ Logo Direita ]</div>", unsafe_allow_html=True)

st.divider()


SOURCE_MAP = {
    "CTI (PS)": "Pronto Socorro",
    "CTI (CC)": "Centro Cirúrgico",
    "CTI (CLIM)": "Clínica Médica",
    "CTI (CLIC)": "Clínica Cirúrgica",
    "CTI (MAT)": "Maternidade",
}

# Cenários padrão: leitos -> (log de eventos, relatório consolidado)
DEFAULT_SCENARIOS = {
    31: {"event_log": "cti_event_log-31.csv", "report": "cti_report_31.txt"},
    # 35: {"event_log": "cti_event_log-35.csv", "report": "cti_report_35.txt"},
    40: {"event_log": "cti_event_log-40.csv", "report": "cti_report_40.txt"},
    50: {"event_log": "cti_event_log-50.csv", "report": "cti_report_50.txt"},
}


# ---------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------
def _reduce_to_last_n_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Mantém apenas os últimos n_days de atividade simulada.

    Jornadas completas são preservadas: qualquer case_id que tenha pelo menos
    um evento dentro da janela mantém *todos* os seus eventos, para que
    tempo de permanência e roteamento permaneçam consistentes.
    """
    if df.empty or n_days <= 0:
        return df

    if "timestamp" not in df.columns:
        return df

    window_minutes = n_days * 1440
    max_ts = df["timestamp"].max()
    min_ts = max_ts - window_minutes

    if df["timestamp"].min() >= min_ts:
        return df

    # Preserva jornadas completas dos pacientes que tocam a janela
    if "case_id" in df.columns:
        case_ids_in_window = df.loc[df["timestamp"] >= min_ts, "case_id"].unique()
        return df[df["case_id"].isin(case_ids_in_window)].copy()

    return df[df["timestamp"] >= min_ts].copy()


@st.cache_data
def load_event_log(path_or_buffer, n_days: int = 365):
    """Carrega o CSV e aplica o recorte dos últimos n_days."""
    df = pd.read_csv(path_or_buffer)
    # Garante tipos básicos
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str)
    if "activity" in df.columns:
        df["activity"] = df["activity"].astype(str)
    df = _reduce_to_last_n_days(df, n_days)
    return df


@st.cache_data
def load_report_text(path):
    return Path(path).read_text(encoding="utf-8")


def parse_report_metrics(text: str) -> dict:
    """Extrai as principais métricas do relatório consolidado (texto)."""

    def find_float(pattern, default=np.nan):
        m = re.search(pattern, text)
        return float(m.group(1).replace(",", "")) if m else default

    def find_int(pattern, default=np.nan):
        m = re.search(pattern, text)
        return int(m.group(1).replace(",", "")) if m else default

    return {
        "stability_index": find_float(r"STABILITY INDEX:\s*([\d.]+)"),
        "avg_time_system_h": find_float(r"Average time in the system:\s*([\d.]+)"),
        "entities_processed": find_int(r"Total number of entities processed:\s*([\d,]+)"),
        "throughput": find_float(r"Throughput:\s*([\d.]+)"),
        "avg_wip": find_float(r"Average WIP:\s*([\d.]+)"),
        "max_wip": find_float(r"Maximum WIP:\s*([\d.]+)"),
        "resource_utilization": find_float(r"Utilization rate:\s*([\d.]+)"),
        "total_revenue": find_float(r"Total Revenue:\s*\$([\d,]+\.\d+)"),
        "total_costs": find_float(r"Total Costs:\s*\$([\d,]+\.\d+)"),
        "net_profit": find_float(r"NET PROFIT:\s*\$([\d,-]+\.\d+)"),
        "profit_margin": find_float(r"Profit Margin:\s*([\d.]+)%"),
    }


def reconstruct_bed_occupancy(df: pd.DataFrame, capacity: int, window_minutes: float) -> tuple:
    """Reconstrói ocupação de leitos CTI a partir do log de eventos start/complete.

    window_minutes: duração efetiva da janela de análise (em minutos de simulação).
    Usado para censurar estadias ainda abertas no final da janela.
    """
    cti_start = df[
        (df["resource"] == "ctiBeds") & (df["lifecycle"].eq("start"))
    ][["case_id", "activity", "timestamp"]].copy()

    cti_end = df[
        (df["resource"] == "ctiBeds") & (df["lifecycle"].eq("complete"))
    ][["case_id", "activity", "timestamp"]].copy()

    pairs = cti_start.merge(cti_end, on=["case_id", "activity"], suffixes=("_start", "_end"))
    pairs["duration_min"] = (pairs["timestamp_end"] - pairs["timestamp_start"]).clip(lower=0)

    # Censurar estadias incompletas (paciente ainda no leito no fim da janela).
    # Usa o timestamp máximo observado na janela (mais robusto que um valor fixo).
    if not df.empty and "timestamp" in df.columns:
        end_of_window = float(df["timestamp"].max())
    else:
        end_of_window = window_minutes

    incomplete = cti_start.merge(cti_end, on=["case_id", "activity"], how="left", suffixes=("_start", "_end"))
    incomplete = incomplete[incomplete["timestamp_end"].isna()].copy()
    if not incomplete.empty:
        incomplete["duration_min"] = (end_of_window - incomplete["timestamp_start"]).clip(lower=0)
        pairs = pd.concat(
            [pairs, incomplete[["case_id", "activity", "timestamp_start", "duration_min"]]],
            ignore_index=True,
        )

    return pairs, cti_start


def compute_scenario_metrics(df: pd.DataFrame, capacity: int, n_days: int) -> dict:
    """Calcula métricas do cenário usando a janela de n_days dias."""
    window_minutes = n_days * 1440.0

    # Se o log (já recortado) for mais curto que n_days, usa a duração real observada
    if not df.empty and "timestamp" in df.columns:
        observed_span = float(df["timestamp"].max() - df["timestamp"].min())
        # Evita divisão por zero e usa no máximo a janela pedida
        effective_minutes = max(min(observed_span, window_minutes), 1.0)
        # Quando preservamos jornadas completas, o span pode ser um pouco maior
        # que n_days; limitamos ao pedido do usuário para taxas diárias consistentes.
        sim_days = n_days
        sim_minutes = window_minutes
    else:
        sim_days = float(n_days)
        sim_minutes = window_minutes

    pairs, cti_start = reconstruct_bed_occupancy(df, capacity, sim_minutes)

    total_bed_minutes = pairs["duration_min"].sum()
    baseline_util = total_bed_minutes / (capacity * sim_minutes)

    # Contribuição de cada fonte em NÚMERO MÉDIO DE LEITOS ocupados (não normalizada
    # pela capacidade). Esta é a grandeza correta para projeções de sensibilidade:
    # ela não "satura" artificialmente perto de 100% quando o cenário já está quase
    # cheio na linha de base — o que gerava resultados contraintuitivos (ex.: parecer
    # que o cenário de 31 leitos, já quase saturado, tinha MENOS margem de piora que
    # os cenários de 35/40 leitos, que têm mais folga).
    # Nota: usamos .rename() (não .reindex() direto) e então SOMAMOS por nome já traduzido,
    # para não perder silenciosamente nenhuma atividade cujo nome bruto não esteja em
    # SOURCE_MAP — caso contrário, source_beds.sum() ficaria menor que o total real de
    # leitos-minuto (total_bed_minutes), fazendo a soma das barras por fonte não bater com
    # o total/"Geral" em outros gráficos do dashboard.
    source_beds_raw = pairs.groupby("activity")["duration_min"].sum() / sim_minutes
    source_beds = (
        source_beds_raw.rename(index=SOURCE_MAP)
        .groupby(level=0)
        .sum()
        .reindex(list(SOURCE_MAP.values()))
        .fillna(0)
    )
    unmapped_mask = ~source_beds_raw.index.isin(SOURCE_MAP.keys())
    outras_beds = float(source_beds_raw[unmapped_mask].sum()) if unmapped_mask.any() else 0.0
    if outras_beds > 1e-9:
        source_beds["Outras origens"] = outras_beds
    total_beds_used = source_beds.sum()

    # Contribuição de cada fonte como fração da capacidade (só para a tabela-resumo
    # estática da linha de base — descreve o cenário atual, não uma projeção).
    source_contrib = source_beds / capacity

    source_arrivals = {}
    for activity, source in SOURCE_MAP.items():
        origem_label = {
            "Pronto Socorro": "Origem_do Pronto Socorro",
            "Centro Cirúrgico": "Origem_do Centro Cirurgico",
            "Clínica Médica": "Origem_da Clinica Medica",
            "Clínica Cirúrgica": "Origem_da Clinica Cirurgica",
            "Maternidade": "Origem_da Maternidade",
        }[source]
        source_arrivals[source] = int((df["activity"] == origem_label).sum())

    source_arrival_rate = {s: n / sim_days for s, n in source_arrivals.items()}
    total_cti_arrival_rate = len(cti_start) / sim_days

    mean_los = (
        pairs.groupby("activity")["duration_min"].mean().rename(index=SOURCE_MAP).reindex(list(SOURCE_MAP.values()))
    )

    return {
        "capacity": capacity,
        "n_days": n_days,
        "sim_days": sim_days,
        "pairs": pairs,
        "cti_start": cti_start,
        "baseline_util": baseline_util,
        "source_contrib": source_contrib,
        "source_beds": source_beds,
        "total_beds_used": total_beds_used,
        "source_arrival_rate": source_arrival_rate,
        "total_cti_arrival_rate": total_cti_arrival_rate,
        "mean_los": mean_los,
    }


@st.cache_data
def build_all_scenarios(scenario_paths: dict, n_days: int = 365, cache_version: str = "v5") -> dict:
    """scenario_paths: {capacity: {'event_log': path, 'report': path or None}}

    n_days controla o recorte temporal aplicado a *todos* os logs simultaneamente.
    """
    scenarios = {}
    for capacity, paths in scenario_paths.items():
        df = load_event_log(paths["event_log"], n_days=n_days)
        metrics = compute_scenario_metrics(df, capacity, n_days=n_days)
        report_metrics = {}
        if paths.get("report"):
            try:
                report_metrics = parse_report_metrics(load_report_text(paths["report"]))
            except FileNotFoundError:
                report_metrics = {}
        metrics["report"] = report_metrics
        scenarios[capacity] = metrics
    return scenarios


# ---------------------------------------------------------------------
# Sidebar: janela de análise + fontes de dados
# ---------------------------------------------------------------------
st.sidebar.header("Janela de análise")

n_days = st.sidebar.slider(
    "Janela de análise (dias)",
    min_value=1,
    max_value=365,
    value=365,
    step=1,
    help="Quantidade de dias finais do event log a serem considerados na análise. "
         "O recorte é aplicado a todos os cenários (31/35/40 leitos) simultaneamente. "
         "Jornadas completas de pacientes que tocam a janela são preservadas. "
         "Valores muito altos (ex.: > 90 dias) podem tornar a aplicação mais lenta.",
)

st.sidebar.caption(
    f"Análise restrita aos **últimos {n_days} dia{'s' if n_days != 1 else ''}** "
    "de cada log de eventos."
)

st.sidebar.divider()
st.sidebar.header("Cenários (leitos de CTI)")
st.sidebar.caption(
    "Por padrão, o app carrega os logs de eventos e relatórios para 31, 40 e 50 leitos "
    "presentes na mesma pasta. Você pode substituir qualquer cenário por um arquivo próprio."
)

scenario_paths = {cap: dict(v) for cap, v in DEFAULT_SCENARIOS.items()}
missing_defaults = []

with st.sidebar.expander("Substituir logs de eventos (opcional)"):
    for capacity in DEFAULT_SCENARIOS:
        uploaded = st.file_uploader(
            f"Log de eventos — {capacity} leitos",
            type=["csv"],
            key=f"upload_{capacity}",
        )
        if uploaded is not None:
            scenario_paths[capacity]["event_log"] = uploaded
            scenario_paths[capacity]["report"] = None  # relatório enviado não disponível

# Verifica quais arquivos padrão existem; se faltar algum, avisa e tenta seguir com os demais.
resolved_paths = {}
for capacity, paths in scenario_paths.items():
    ev = paths["event_log"]
    if isinstance(ev, str) and not Path(ev).exists():
        missing_defaults.append((capacity, ev))
        continue
    resolved_paths[capacity] = paths

if missing_defaults:
    faltando = ", ".join(f"{cap} leitos ({p})" for cap, p in missing_defaults)
    st.warning(
        f"Não encontrei o(s) arquivo(s) padrão para: {faltando}. "
        "Envie o(s) log(s) correspondente(s) na barra lateral ou coloque-os ao lado de streamlit_app.py."
    )

if not resolved_paths:
    st.error("Nenhum log de eventos disponível. Envie ao menos um arquivo CSV na barra lateral.")
    st.stop()

scenarios = build_all_scenarios(resolved_paths, n_days=n_days, cache_version="v5")
available_capacities = sorted(scenarios.keys())

# ---------------------------------------------------------------------
# 0. Visão geral: utilização vs número de leitos
# ---------------------------------------------------------------------
st.header("Utilização do CTI vs. Número de Leitos")

overview_rows = []
for cap in available_capacities:
    s = scenarios[cap]
    r = s["report"]
    overview_rows.append({
        "Leitos": cap,
        "Utilização (log de eventos)": s["baseline_util"] * 100,
        "Utilização (relatório)": r.get("resource_utilization", np.nan) * 100 if r.get("resource_utilization") == r.get("resource_utilization") else np.nan,
        "Índice de estabilidade": r.get("stability_index", np.nan),
        "WIP médio": r.get("avg_wip", np.nan),
        "WIP máximo": r.get("max_wip", np.nan),
        "Pacientes atendidos": r.get("entities_processed", np.nan),
        "Lucro líquido ($)": r.get("net_profit", np.nan),
        "Margem de lucro (%)": r.get("profit_margin", np.nan),
    })
overview_df = pd.DataFrame(overview_rows).sort_values("Leitos")

# "Sobrecarga (%)" traduz o STABILITY INDEX do relatório (capacidade/demanda: índice > 1 =
# folga, < 1 = sobrecarga) para a mesma escala da utilização (demanda/capacidade em %),
# invertendo a razão (100 / índice). Assim os dois indicadores ficam no mesmo eixo e podem
# ser comparados diretamente no mesmo gráfico, em vez de dois gráficos quase iguais e com
# números ligeiramente diferentes — o que passava a impressão (incorreta) de inconsistência.
tem_indice_estabilidade = overview_df["Índice de estabilidade"].notna().any()
if tem_indice_estabilidade:
    overview_df["Sobrecarga (índice de estabilidade, %)"] = 100 / overview_df["Índice de estabilidade"]

max_y = max(
    float(overview_df["Utilização (log de eventos)"].max()),
    float(overview_df["Sobrecarga (índice de estabilidade, %)"].max()) if tem_indice_estabilidade else 0,
    100,
) * 1.2

fig_overview = go.Figure()

# Zonas de referência (comuns às duas métricas, já que ambas expressam demanda/capacidade em %)
fig_overview.add_hrect(y0=0, y1=85, fillcolor="#2ca02c", opacity=0.06, line_width=0)
fig_overview.add_hrect(y0=85, y1=100, fillcolor="#f1c232", opacity=0.08, line_width=0)
fig_overview.add_hrect(y0=100, y1=max_y, fillcolor="#d62728", opacity=0.06, line_width=0)
fig_overview.add_hline(
    y=100, line_dash="dash", line_color="crimson",
    annotation_text="Limite crítico: acima disso, a fila de espera por leito cresce continuamente",
    annotation_position="top left",
)

fig_overview.add_trace(go.Bar(
    x=overview_df["Leitos"].astype(str),
    y=overview_df["Utilização (log de eventos)"],
    name="Utilização observada (log de eventos)",
    marker_color="#1f77b4",
    text=[f"{v:.1f}%" for v in overview_df["Utilização (log de eventos)"]],
    textposition="outside",
    customdata=overview_df["Leitos"],
    hovertemplate=(
        "<b>Utilização observada (log de eventos)</b><br>"
        "Leitos: %{customdata:.0f}<br>"
        "Utilização: %{y:.1f}%"
        "<extra></extra>"
    ),
))

if tem_indice_estabilidade:
    fig_overview.add_trace(go.Scatter(
        x=overview_df["Leitos"].astype(str),
        y=overview_df["Sobrecarga (índice de estabilidade, %)"],
        name="Sobrecarga estimada (índice de estabilidade)",
        mode="lines+markers+text",
        line=dict(width=3, color="#ff7f0e", dash="dot"),
        marker=dict(size=11, symbol="diamond"),
        text=[
            f"{v:.0f}%" if pd.notna(v) else ""
            for v in overview_df["Sobrecarga (índice de estabilidade, %)"]
        ],
        textposition="top center",
        customdata=overview_df["Leitos"],
        hovertemplate=(
            "<b>Sobrecarga estimada (índice de estabilidade)</b><br>"
            "Leitos: %{customdata:.0f}<br>"
            "Sobrecarga: %{y:.0f}%"
            "<extra></extra>"
        ),
    ))

fig_overview.update_layout(
    title=f"Utilização e grau de sobrecarga do CTI por número de leitos (últimos {n_days} dias)",
    xaxis_title="Número de leitos de CTI",
    yaxis_title="Demanda em relação à capacidade (%)",
    # Eixo X categórico: os cenários (31, 40, 50 leitos, etc.) são pontos discretos, não uma
    # escala contínua — sem isso o Plotly interpola posições intermediárias (31.5, 32...) e
    # o hover passa a mostrar um número de leitos "entre" cenários em vez do valor real.
    xaxis=dict(
        type="category",
        categoryorder="array",
        categoryarray=[str(c) for c in available_capacities],
    ),
    yaxis_range=[0, max_y],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hoverlabel=dict(align="left"),
)
st.plotly_chart(fig_overview, use_container_width=True)

display_cols = ["Leitos", "Utilização (log de eventos)"]
format_dict = {"Utilização (log de eventos)": "{:.1f}%"}
if tem_indice_estabilidade:
    display_cols.append("Sobrecarga (índice de estabilidade, %)")
    format_dict["Sobrecarga (índice de estabilidade, %)"] = "{:.0f}%"

st.dataframe(
    overview_df[display_cols].style.format(format_dict, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

if tem_indice_estabilidade:
    st.caption(
        "Este gráfico combina **duas fontes de dado independentes** sobre o mesmo fenômeno "
        "(quão perto o CTI está de sua capacidade máxima), por isso é normal que os valores "
        "sejam próximos mas não idênticos:\n\n"
        "- **Utilização observada (log de eventos):** medição direta — quanto tempo os leitos "
        "ficaram de fato ocupados, calculada a partir dos timestamps de início/fim de uso do "
        "leito (`ctiBeds`) nos últimos "
        f"**{n_days} dia{'s' if n_days != 1 else ''}**.\n"
        "- **Sobrecarga estimada (índice de estabilidade):** um indicador de risco de fila, "
        "vindo do relatório consolidado da simulação, calculado a partir das taxas médias de "
        "chegada e atendimento do modelo. Ele pode divergir um pouco da utilização observada "
        "porque considera a variabilidade estatística das chegadas, não só a ocupação média.\n\n"
        "Quando as duas linhas concordam (como aqui), isso reforça a robustez da análise. "
        "**Abaixo de 85%** (verde) há folga de capacidade; **entre 85% e 100%** (amarelo) a "
        "ocupação está próxima do limite; **acima de 100%** (vermelho) a demanda supera a "
        "capacidade instalada e a fila de espera por leitos tende a crescer continuamente."
    )
else:
    st.caption(
        f"**Utilização observada (log de eventos)** é recalculada diretamente a partir dos "
        f"timestamps de início/conclusão de uso do leito (`ctiBeds`) em cada log, considerando "
        f"os últimos **{n_days} dia{'s' if n_days != 1 else ''}**. **Abaixo de 85%** (verde) há "
        "folga de capacidade; **entre 85% e 100%** (amarelo) a ocupação está próxima do limite; "
        "**acima de 100%** (vermelho) a demanda supera a capacidade instalada."
    )

st.divider()

# ---------------------------------------------------------------------
# Seleção do cenário para análise detalhada
# ---------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("Análise detalhada")
selected_capacity = st.sidebar.radio(
    "Cenário (leitos de CTI) para detalhamento",
    available_capacities,
    index=0,
    horizontal=True,
)

scenario = scenarios[selected_capacity]
CAPACITY = selected_capacity
baseline_util = scenario["baseline_util"]
source_contrib = scenario["source_contrib"]
source_beds = scenario["source_beds"]
total_beds_used = scenario["total_beds_used"]
source_arrival_rate = scenario["source_arrival_rate"]
total_cti_arrival_rate = scenario["total_cti_arrival_rate"]
mean_los = scenario["mean_los"]
cti_start = scenario["cti_start"]
pairs_df = scenario["pairs"]
sim_days = scenario.get("sim_days", n_days)

st.header(f"Detalhamento do cenário: {CAPACITY} leitos")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Leitos CTI", CAPACITY)
c2.metric("Utilização base da CTI", f"{baseline_util:.1%}")
c3.metric("Taxa atual de chegada ao CTI", f"{total_cti_arrival_rate:.1f} pacientes/dia")
c4.metric("Horizonte de análise", f"{sim_days:.0f} dias")

st.info(
    f"**Taxa atual de chegada ao CTI: {total_cti_arrival_rate:.1f} pacientes/dia.** "
    f"Isso corresponde a {len(cti_start):,} entradas no CTI durante os últimos {sim_days:.0f} dias "
    "de simulação. Esta é a taxa de referência utilizada como **1,00×** nas análises."
)

st.subheader("O que o log de eventos diz sobre este cenário")

summary = pd.DataFrame({
    "Fonte": list(SOURCE_MAP.values()),
    "Chegadas/dia": [source_arrival_rate[s] for s in SOURCE_MAP.values()],
    "Tempo Médio (ALOS) CTI (dias)": [
        mean_los[s] / 1440 if pd.notna(mean_los[s]) else np.nan for s in SOURCE_MAP.values()
    ],
    "Impacto na utilização": [source_contrib[s] for s in SOURCE_MAP.values()],
})
st.dataframe(
    summary.style.format({
        "Chegadas/dia": "{:.1f}",
        "Tempo Médio (ALOS) CTI (dias)": "{:.2f}",
        "Impacto na utilização": "{:.1%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.info(
    "Descoberta importante do modelo: os leitos de CTI (ctiBeds) são liberados quando cada "
    "bloco de processo de monitoramento da CTI é concluído. O bloco subsequente 'CTI prepara Encaminhamento' "
    "e a decisão de destino a jusante não retêm os leitos. "
    "Portanto, alterar a taxa de roteamento de destino/alta a jusante não "
    "altera a utilização do CTI no modelo atual."
)

# ---------------------------------------------------------------------
# Visualização do fluxo (HTML)
# ---------------------------------------------------------------------
st.subheader("Visualização do Fluxo do Processo")
st.caption("Visualização interativa do fluxo da simulação.")

try:
    import streamlit.components.v1 as components
    with open("cti.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
except FileNotFoundError:
    st.info("Arquivo 'cti.html' não encontrado. Coloque o arquivo HTML na mesma pasta para exibi-lo aqui.")

st.divider()

# ---------------------------------------------------------------------
# Sensibilidade da taxa de chegada
# ---------------------------------------------------------------------
st.subheader("1. Impacto do aumento da taxa de chegada em cada fonte")

st.markdown(
    f"**Taxa atual de chegada ao CTI (todas as fontes): "
    f"{total_cti_arrival_rate:.1f} pacientes/dia.**"
)

SOURCE_OPTIONS_GERAL = ["Geral (todos)"] + list(SOURCE_MAP.values())

col_fonte, col_mult = st.columns(2)
with col_fonte:
    source = st.selectbox(
        "Fonte",
        SOURCE_OPTIONS_GERAL,
        key="arrival_source",
        help="'Geral (todos)' simula todas as fontes de chegada aumentando/diminuindo ao "
             "mesmo tempo, na mesma proporção — útil para saber como o CTI se comporta "
             "quando a chegada de pacientes do hospital inteiro varia.",
    )
with col_mult:
    max_multiplier = st.slider(
        "Multiplicador máximo da taxa de chegada",
        min_value=1.25,
        max_value=3.0,
        value=2.0,
        step=0.25,
        key="arrival_max",
        help="1.0× é a linha de base. 2.0× significa duas vezes a taxa de chegada da linha de base.",
    )

is_geral_chegada = source == "Geral (todos)"
if is_geral_chegada:
    source_rate = total_cti_arrival_rate
else:
    source_rate = source_arrival_rate.get(source, 0.0)

st.info(
    f"**Taxa atual de {source}: {source_rate:.1f} pacientes/dia.** "
    f"A taxa total atual de chegada ao CTI é **{total_cti_arrival_rate:.1f} pacientes/dia**. "
    "No gráfico, **1,00× representa a taxa atual da fonte selecionada**."
    + (
        " Como 'Geral (todos)' foi selecionado, o multiplicador é aplicado simultaneamente "
        "a todas as fontes de chegada ao CTI."
        if is_geral_chegada else ""
    )
)

multipliers = np.linspace(0.5, max_multiplier, 31)

# IMPORTANTE: a projeção é feita em NÚMERO DE LEITOS necessários (grandeza absoluta,
# sem limite superior artificial), e só depois convertida em % da capacidade instalada.
# Isso evita o erro anterior, em que a utilização era limitada (clipada) em 100%: como o
# cenário de 31 leitos já parte de uma base quase saturada (~96%), o "teto" artificial
# sobrava muito pouca margem para crescer (só 4 pp), fazendo parecer — de forma
# contraintuitiva — que o cenário com MENOS leitos era o MENOS sensível a aumentos de
# demanda. Sem o teto artificial, o resultado correto aparece: cenários com menos folga
# de capacidade ultrapassam a capacidade instalada com aumentos de demanda MENORES.
if is_geral_chegada:
    base_beds = float(total_beds_used)
    other_beds = 0.0
else:
    base_beds = float(source_beds.get(source, 0))
    other_beds = total_beds_used - base_beds

projected_beds = other_beds + base_beds * multipliers
projected_util_pct = projected_beds / CAPACITY * 100

arrival_df = pd.DataFrame({
    "Multiplicador da taxa de chegada": multipliers,
    "Leitos necessários (projeção)": projected_beds,
    "Utilização do CTI (projeção)": projected_util_pct,
})

st.caption(
    f"Para {source}: **1,00× = {source_rate:.1f} pacientes/dia** (taxa atual); "
    f"**1,25× = {source_rate * 1.25:.1f} pacientes/dia** (+25%); "
    f"**2,00× = {source_rate * 2:.1f} pacientes/dia** (+100%)."
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=arrival_df["Multiplicador da taxa de chegada"],
    y=arrival_df["Leitos necessários (projeção)"],
    mode="lines+markers",
    name="Leitos necessários (projeção)",
    line=dict(width=3, color="#1f77b4"),
    customdata=arrival_df["Utilização do CTI (projeção)"],
    hovertemplate=(
        "Multiplicador: %{x:.2f}×<br>"
        "Leitos necessários: %{y:.1f}<br>"
        "Utilização projetada: %{customdata:.1f}%"
        "<extra></extra>"
    ),
))
fig.add_hline(
    y=CAPACITY,
    line_dash="dash",
    line_color="crimson",
    annotation_text=f"Capacidade instalada: {CAPACITY} leitos",
    annotation_position="top left",
)
fig.add_hrect(
    y0=CAPACITY,
    y1=max(float(projected_beds.max()), CAPACITY) * 1.15 + 1,
    fillcolor="crimson",
    opacity=0.07,
    line_width=0,
    annotation_text="Zona de sobrecarga (demanda > capacidade)",
    annotation_position="top right",
)
fig.add_trace(go.Scatter(
    x=[1.0],
    y=[other_beds + base_beds],
    mode="markers+text",
    name="Base atual (1,00×)",
    marker=dict(size=12, color="black", symbol="diamond"),
    text=["Base atual"],
    textposition="bottom center",
    hovertemplate=(
        f"Base atual (1,00×)<br>Leitos: %{{y:.1f}}"
        "<extra></extra>"
    ),
))
fig.update_layout(
    title=f"Leitos de CTI necessários quando as chegadas de {source} mudam ({CAPACITY} leitos)",
    xaxis_title="Taxa de chegada da fonte / linha de base",
    yaxis_title="Número médio de leitos necessários",
    # Formata os rótulos do eixo Y como números inteiros (sem casas decimais tipo 31.5,
    # 32.5...), já que "número de leitos" é mais legível como valor arredondado no eixo.
    yaxis=dict(tickformat=",d"),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

max_util = float(projected_util_pct.max())
if max_util > 100:
    st.warning(
        f"⚠️ No multiplicador máximo ({max_multiplier:.2f}×), a demanda projetada exigiria "
        f"**{float(projected_beds.max()):.1f} leitos**, ou seja, **{max_util:.0f}% da capacidade "
        f"instalada** de {CAPACITY} leitos — o sistema ficaria sobrecarregado (demanda além da "
        "capacidade física, refletida em fila/espera, não em utilização acima de 100%)."
    )
else:
    st.caption(
        f"No multiplicador máximo ({max_multiplier:.2f}×), a demanda projetada usaria "
        f"**{float(projected_beds.max()):.1f} leitos** ({max_util:.0f}% da capacidade de {CAPACITY} leitos)."
    )

st.markdown("**Comparando o impacto por fonte de origem (variação na chegada)**")

increase = st.slider(
    "Variação comum na taxa de chegada",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%d%%",
    key="arrival_increase",
    help="Valores positivos aumentam a taxa de chegada (mais demanda, mais leitos ocupados); "
         "valores negativos diminuem a taxa de chegada (menos demanda, menos leitos ocupados). "
         "Por exemplo, +25% multiplica a taxa de chegada base por 1,25; -25% multiplica por 0,75.",
)

# Mesmo princípio: leitos adicionais (ou a menos) necessários (absoluto, sem teto artificial) e
# só então convertido em pontos percentuais da capacidade instalada deste cenário. Funciona nos
# dois sentidos: 'increase' pode ser negativo (redução na taxa de chegada).
arrival_sensitivity = pd.DataFrame({
    "Fonte": list(source_beds.index),
    "Leitos adicionais necessários": source_beds.values * (increase / 100),
})
arrival_sensitivity["Variação na utilização do CTI (pp)"] = (
    arrival_sensitivity["Leitos adicionais necessários"] / CAPACITY * 100
)

fig2_title = (
    f"Aumento da utilização do CTI com um aumento de {increase}% na taxa de chegada ({CAPACITY} leitos)"
    if increase >= 0
    else f"Redução da utilização do CTI com uma queda de {abs(increase)}% na taxa de chegada ({CAPACITY} leitos)"
)

pp_values = arrival_sensitivity["Variação na utilização do CTI (pp)"]
beds_values = arrival_sensitivity["Leitos adicionais necessários"]
total_pp = float(pp_values.sum())

# Construído com go.Bar (em vez de px.bar + text_auto) para controlar explicitamente o
# arredondamento do texto exibido nas barras — o text_auto do Plotly Express, dependendo da
# versão instalada, pode ignorar a string de formato e mostrar o float bruto (ex.:
# "4.572295..." em vez de "+4.57%").
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=arrival_sensitivity["Fonte"],
    y=pp_values,
    marker_color=["#1f77b4" if v >= 0 else "#d62728" for v in pp_values],
    text=[f"{v:+.2f} pp" for v in pp_values],
    textposition="outside",
    customdata=beds_values,
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Variação na utilização do CTI: %{y:+.2f} pp<br>"
        "Leitos adicionais necessários: %{customdata:+.2f}"
        "<extra></extra>"
    ),
))
fig2.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1)
# Anotação com a soma das fontes: deixa explícito que a soma das barras (variação real na
# utilização) não precisa ser igual, em módulo, ao percentual do slider — a taxa de chegada
# varia X%, mas o efeito em pontos percentuais de utilização depende de quanto cada fonte já
# ocupa da capacidade instalada, então a soma tende a ficar abaixo do valor nominal do slider.
fig2.add_annotation(
    xref="paper", yref="paper", x=1, y=1.12, showarrow=False,
    align="right", font=dict(size=12, color="#555"),
    text=f"Soma das 5 fontes: {total_pp:+.2f} pp",
)
fig2.update_layout(
    title=fig2_title,
    xaxis_title=None,
    yaxis_title="Variação na utilização do CTI (pontos percentuais)",
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Para a mesma variação percentual na taxa de chegada (para mais ou para menos), cenários "
    "com **menos leitos** mostram uma variação maior em pontos percentuais de utilização — "
    "porque os mesmos leitos adicionais (ou a menos) representam uma fatia maior de uma "
    "capacidade instalada menor. Isso é consistente com a intuição: sistemas com menos folga "
    "de capacidade são mais sensíveis tanto a aumentos quanto a quedas de demanda. **Note que a "
    f"soma das 5 barras ({total_pp:+.2f} pp) não precisa ser igual, em módulo, ao "
    f"{increase}% aplicado no slider**: o slider varia a *taxa de chegada* de cada fonte, e o "
    "efeito em pontos percentuais de utilização depende de quantos leitos cada fonte já ocupa "
    "hoje em relação à capacidade instalada — por isso a soma costuma ficar abaixo do valor "
    "nominal do slider."
)

# ---------------------------------------------------------------------
# Sensibilidade da taxa de alta da CTI
# ---------------------------------------------------------------------
st.subheader("2. Impacto do aumento da taxa de alta da CTI")

st.info(
    "Não há como rastrear a alta de um paciente do CTI pela sua fonte de origem: a saída do "
    "CTI é determinada pela **condição clínica** do paciente (uma variável interna do "
    "atendimento), não pela porta de entrada. Por isso, esta análise considera sempre a taxa "
    "de alta do **CTI como um todo (Geral — todos os pacientes)**, e não por fonte individual "
    "como na seção anterior."
)

st.markdown(
    f"**Taxa atual de chegada ao CTI (todas as fontes): "
    f"{total_cti_arrival_rate:.1f} pacientes/dia.** "
    "A análise abaixo altera a **taxa de alta do CTI**, aplicada a todos os pacientes ao mesmo tempo."
)

# ALOS (Average Length of Stay) agregado, calculado diretamente do log de eventos: média das
# durações de internação no CTI (todas as fontes), em dias. É o inverso da taxa de alta e a
# grandeza mais intuitiva para gestores hospitalares (dias, em vez de um multiplicador
# adimensional).
if not pairs_df.empty and pairs_df["duration_min"].notna().any():
    baseline_alos_days = float(pairs_df["duration_min"].mean()) / 1440.0
else:
    baseline_alos_days = np.nan

alos_texto = (
    f"tempo médio de permanência (ALOS) atual de **{baseline_alos_days:.2f} dias**"
    if pd.notna(baseline_alos_days) else "tempo médio de permanência (ALOS) indisponível para este cenário"
)

st.info(
    "**Sobre a unidade da 'taxa de alta':** ela é expressa como um **multiplicador da taxa de "
    "referência** (1,00× = hoje), não em uma unidade absoluta como 'pacientes/dia', porque "
    "representa a *velocidade* de liberação dos leitos — o inverso do **tempo médio de "
    f"permanência (ALOS)** no CTI. Hoje, o CTI (Geral — todos os pacientes) tem um {alos_texto}. "
    "Um multiplicador de 2,00× significa o dobro da velocidade de liberação, ou seja, "
    f"aproximadamente **{(baseline_alos_days / 2):.2f} dias** de permanência; um multiplicador "
    f"de 0,50× significa metade da velocidade, ou aproximadamente **{(baseline_alos_days / 0.5):.2f} "
    "dias** de permanência. Os eixos superior do gráfico e os textos abaixo trazem essa "
    "conversão para dias."
)

max_discharge_multiplier = st.slider(
    "Multiplicador máximo da taxa de alta do CTI (Geral — todos)",
    min_value=1.25,
    max_value=3.0,
    value=2.5,
    step=0.25,
    key="discharge_max",
    help="A 'taxa de alta da CTI' é a taxa na qual um paciente deixa o leito de CTI. "
         "1,00× = taxa de alta de referência; 2,00× significa aproximadamente o dobro da "
         "taxa de alta e, portanto, aproximadamente metade do tempo de permanência (em dias).",
)

st.caption(
    "A 'taxa de alta da CTI' é a taxa na qual um paciente deixa o leito de CTI. "
    "**1,00× = taxa de alta de referência**; **2,00×** significa aproximadamente "
    "o dobro da taxa de alta e, portanto, aproximadamente metade do tempo de permanência "
    f"— hoje equivalente a cerca de **{baseline_alos_days:.2f} dias**."
)

discharge_multipliers = np.linspace(0.5, max_discharge_multiplier, 31)
# "Geral (todos)": a taxa de alta é aplicada a todos os leitos ocupados do cenário ao mesmo
# tempo (não há segmentação por fonte, pois a alta não é rastreável por origem).
disc_base_beds = float(total_beds_used)
disc_other_beds = 0.0

projected_beds_disc = disc_other_beds + disc_base_beds / discharge_multipliers
projected_util_disc = projected_beds_disc / CAPACITY * 100

discharge_df = pd.DataFrame({
    "Multiplicador da taxa de alta da CTI": discharge_multipliers,
    "Leitos necessários (projeção)": projected_beds_disc,
    "Utilização do CTI (projeção)": projected_util_disc,
})
# Tempo médio de permanência (dias) equivalente a cada multiplicador (ALOS_base / multiplicador).
if pd.notna(baseline_alos_days):
    discharge_df["ALOS equivalente (dias)"] = baseline_alos_days / discharge_multipliers
else:
    discharge_df["ALOS equivalente (dias)"] = np.nan

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=discharge_df["Multiplicador da taxa de alta da CTI"],
    y=discharge_df["Leitos necessários (projeção)"],
    mode="lines+markers",
    line=dict(width=3, color="#2ca02c"),
    customdata=discharge_df["ALOS equivalente (dias)"],
    hovertemplate=(
        "Multiplicador: %{x:.2f}×<br>"
        "Leitos necessários: %{y:.1f}<br>"
        "Tempo médio de permanência equivalente: %{customdata:.2f} dias"
        "<extra></extra>"
    ),
))
fig3.add_hline(
    y=CAPACITY,
    line_dash="dash",
    line_color="crimson",
    annotation_text=f"Capacidade instalada: {CAPACITY} leitos",
    annotation_position="top left",
)
fig3.add_hrect(
    y0=CAPACITY,
    y1=max(float(projected_beds_disc.max()), CAPACITY) * 1.15 + 1,
    fillcolor="crimson",
    opacity=0.07,
    line_width=0,
    annotation_text="Zona de sobrecarga (demanda > capacidade)",
    annotation_position="top right",
)
fig3.add_trace(go.Scatter(
    x=[1.0],
    y=[disc_other_beds + disc_base_beds],
    mode="markers+text",
    marker=dict(size=12, color="black", symbol="diamond"),
    text=["Base atual"],
    textposition="bottom center",
    hovertemplate=(
        f"Base atual (1,00×)<br>Leitos: %{{y:.1f}}<br>ALOS: {baseline_alos_days:.2f} dias"
        "<extra></extra>"
    ),
))

# Eixo superior espelhado, com a mesma escala do multiplicador, mas rotulado em dias de
# permanência (ALOS_base / multiplicador) — mais intuitivo para gestores hospitalares.
if pd.notna(baseline_alos_days):
    top_tick_multipliers = sorted(set(
        [round(v, 2) for v in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, max_discharge_multiplier]]
        + [round(discharge_multipliers.min(), 2), round(discharge_multipliers.max(), 2)]
    ))
    top_tick_multipliers = [
        v for v in top_tick_multipliers
        if discharge_multipliers.min() - 1e-9 <= v <= discharge_multipliers.max() + 1e-9
    ]
    fig3.add_trace(go.Scatter(
        x=top_tick_multipliers,
        y=[disc_other_beds + disc_base_beds] * len(top_tick_multipliers),
        mode="markers",
        marker=dict(size=0.1, color="rgba(0,0,0,0)"),
        xaxis="x2",
        showlegend=False,
        hoverinfo="skip",
    ))
    fig3.update_layout(
        xaxis2=dict(
            overlaying="x",
            matches="x",
            side="top",
            tickmode="array",
            tickvals=top_tick_multipliers,
            ticktext=[f"{baseline_alos_days / v:.2f} d" for v in top_tick_multipliers],
            title="Tempo médio de permanência equivalente (dias)",
        )
    )

fig3.update_layout(
    title=f"Leitos de CTI necessários quando a taxa de alta muda (Geral — todos os pacientes, {CAPACITY} leitos)",
    xaxis_title="Taxa de alta da CTI / linha de base",
    yaxis_title="Número médio de leitos necessários",
    yaxis=dict(
        range=[0, max(float(projected_beds_disc.max()), CAPACITY) * 1.2],
        tickformat=",d",
    ),
    showlegend=False,
)
st.plotly_chart(fig3, use_container_width=True)

st.caption(
    f"Hoje, o CTI tem em média **{disc_base_beds:.1f} leitos ocupados** (todas as fontes "
    f"somadas), com um tempo médio de permanência (ALOS) de aproximadamente "
    f"**{baseline_alos_days:.2f} dias**. Como a alta não pode ser segmentada por origem, a "
    "mudança na taxa de alta é aplicada de uma vez sobre esse total — quanto maior o aumento "
    "na taxa de alta, menor o tempo médio de permanência e menor o número de leitos necessários."
)

# --- Impacto agregado: variação (para mais ou para menos) na taxa de alta ---
st.markdown("**Impacto agregado da variação na taxa de alta (CTI como um todo)**")

discharge_increase = st.slider(
    "Variação comum na taxa de alta da CTI (Geral — todos)",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%d%%",
    key="discharge_increase",
    help="Valores positivos aumentam a taxa de alta (leitos são liberados mais rápido, ALOS "
         "menor); valores negativos diminuem a taxa de alta (pacientes ficam mais tempo, ALOS "
         "maior). Por exemplo, +25% multiplica a taxa de alta de todos os pacientes do CTI por "
         "1,25 (reduz o ALOS em ~1/1,25); -25% multiplica por 0,75 (aumenta o ALOS em ~1/0,75). "
         "Aplicado ao CTI como um todo, já que a alta não é rastreável por fonte de origem.",
)

discharge_factor = 1 + discharge_increase / 100
beds_freed = total_beds_used * (1 - 1 / discharge_factor)
util_change_pp = beds_freed / CAPACITY * 100
projected_alos_days = baseline_alos_days / discharge_factor if pd.notna(baseline_alos_days) else np.nan

col_disc1, col_disc2, col_disc3, col_disc4 = st.columns(4)
col_disc1.metric("Tempo médio de permanência hoje (ALOS)", f"{baseline_alos_days:.2f} dias")
col_disc2.metric(
    "Tempo médio de permanência projetado",
    f"{projected_alos_days:.2f} dias",
    delta=f"{projected_alos_days - baseline_alos_days:+.2f} dias",
    delta_color="inverse",
)
col_disc3.metric("Leitos liberados/adicionais (estimado)", f"{beds_freed:+.1f}")
col_disc4.metric("Variação na utilização do CTI", f"{util_change_pp:+.1f} pp", delta_color="inverse")

if discharge_increase >= 0:
    direcao_txt = (
        f"Com um aumento de {discharge_increase}% na taxa de alta do CTI (aplicado a todos os "
        "pacientes simultaneamente), o tempo médio de permanência cairia de "
        f"**{baseline_alos_days:.2f} dias** para aproximadamente **{projected_alos_days:.2f} dias**, "
        f"e a utilização do CTI cairia aproximadamente **{abs(util_change_pp):.1f} pontos "
        "percentuais**"
    )
else:
    direcao_txt = (
        f"Com uma redução de {abs(discharge_increase)}% na taxa de alta do CTI (aplicado a "
        "todos os pacientes simultaneamente), o tempo médio de permanência subiria de "
        f"**{baseline_alos_days:.2f} dias** para aproximadamente **{projected_alos_days:.2f} dias**, "
        f"e a utilização do CTI subiria aproximadamente **{abs(util_change_pp):.1f} pontos "
        "percentuais**"
    )

st.caption(
    direcao_txt
    + f" — de **{baseline_util:.1%}** para cerca de "
    f"**{baseline_util * 100 - util_change_pp:.1f}%** da capacidade instalada de "
    f"{CAPACITY} leitos."
)

# ---------------------------------------------------------------------
# Interpretação
# ---------------------------------------------------------------------
st.subheader("Interpretação")

st.markdown(
    f"""
- Comparando os três cenários simulados, a utilização do CTI **cai de {scenarios[available_capacities[0]]['baseline_util']:.1%}
  com {available_capacities[0]} leitos para {scenarios[available_capacities[-1]]['baseline_util']:.1%} com
  {available_capacities[-1]} leitos**, conforme mais capacidade é adicionada ao sistema.
- No cenário selecionado ({CAPACITY} leitos), o log de eventos fornece uma utilização base da CTI de
  **{baseline_util:.1%}** ao longo do horizonte de observação de **{sim_days:.0f} dias**.
- Aumentar a taxa de chegada de uma fonte aumenta a utilização do CTI porque
  essa fonte contribui com demanda adicional de tempo de leito.
- A magnitude depende da contribuição de tempo de leito do CTI observada da fonte.
- Aumentar a **taxa de alta da CTI** reduz a utilização do CTI porque
  reduz o tempo esperado de permanência no CTI.
"""
)



# Aqui está um mapeamento das perguntas que diretores tipicamente fazem, organizado por onde no dashboard você encontra a resposta:

# 1. Situação atual (visão geral, sem mexer em nada)
# "Qual a nossa utilização atual do CTI?" → métrica "Utilização base da CTI" e o gráfico de visão geral.
# "Estamos operando em risco de fila/sobrecarga hoje?" → zonas verde/amarelo/vermelho no gráfico geral (limite de 85% e 100%).
# "Quantos pacientes o CTI atende por dia hoje?" → "Taxa atual de chegada ao CTI".
# "Qual o tempo médio de internação no CTI (ALOS)?" → métrica ALOS em dias, na seção 2.
# "De onde vêm os pacientes do CTI (PS, Centro Cirúrgico, Clínica Médica, etc.)?" → tabela "O que o log de eventos diz sobre este cenário".
# 2. Capacidade e investimento (comparação de cenários)
# "Precisamos expandir o CTI? De 31 para 40 ou 50 leitos, o que muda?" → gráfico "Utilização do CTI vs. Número de Leitos" no topo.
# "Com mais leitos, ficamos financeiramente melhores ou piores?" → tabela com lucro líquido e margem de lucro por cenário.
# "Quão perto estamos do ponto de virar fila de espera?" → índice de estabilidade / linha de 100% no mesmo gráfico.
# 3. "E se..." de demanda (seção 1 — taxa de chegada)
# "Se aumentarmos os leitos cirúrgicos e isso trouxer mais pacientes ao CTI, o que acontece?" → selecionar "Centro Cirúrgico" em Fonte + variação positiva.
# "Se toda a demanda do hospital crescer ao mesmo tempo (não só uma área), o CTI aguenta?" → opção "Geral (todos)" no seletor Fonte — essa é a pergunta que motivou aquela opção.
# "Qual área de origem mais pressiona o CTI hoje?" → gráfico "Comparando o impacto por fonte de origem".
# "Até quanto a demanda pode crescer antes de estourarmos a capacidade?" → ponto onde a linha cruza a "Capacidade instalada" no primeiro gráfico da seção 1.
# "E se a demanda cair (menos cirurgias, sazonalidade baixa)?" → agora o slider de variação também cobre isso, com valores negativos.
# 4. "E se..." de giro de leito (seção 2 — taxa de alta)
# "Se conseguirmos agilizar a alta do CTI (ex.: protocolo de desospitalização mais rápido), quanta capacidade isso libera?" → variação positiva na taxa de alta.
# "Se pacientes ficarem mais tempo internados (ex.: falta de vaga na enfermaria para onde ele sairia), qual o impacto?" → variação negativa — mostra ALOS subindo e utilização piorando.
# "Conseguimos saber se a alta lenta é culpa de uma clínica específica?" → aqui você vai precisar explicar a limitação: não, a alta é por condição clínica, não por origem — por isso a análise é sempre "Geral".
# Perguntas que o board pode fazer e o dashboard não responde diretamente

# Vale já ter a resposta pronta para não ser pego de surpresa:

# Custo de abrir novos leitos (equipe, equipamento) — o dashboard mostra receita/custo/lucro da simulação, não um orçamento de capex.
# Causa raiz de por que uma fonte específica tem ALOS alto (o dashboard mostra o dado, não o motivo clínico).
# Impacto de mudanças que não estão no modelo (ex.: nova ala, novo protocolo não simulado).