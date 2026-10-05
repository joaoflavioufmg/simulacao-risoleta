from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# streamlit run cc_app.py

st.set_page_config(
    page_title="CC: Dashboard Avançado do Centro Cirúrgico",
    page_icon="🏥",
    layout="wide",
)

# Criação de 3 colunas para alinhar a logo esquerda, o título no centro e a logo direita
col_logo_esq, col_titulo, col_logo_dir = st.columns([1, 6, 1])

with col_logo_esq:
    # Substitua pelo caminho da sua logo esquerda (ex: URL, arquivo local .png/.jpg)
    # Dica: use o parâmetro width para ajustar o tamanho médio desejado (ex: 120 a 180 pixels)
    caminho_logo_esq = "../../figs/Logo_UFMG.png"  # Ajuste para o seu arquivo ou URL
    if Path(caminho_logo_esq).exists():
        st.image(caminho_logo_esq, width=180)
    else:
        # Placeholder visual caso o arquivo ainda não esteja na pasta
        st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>[ Logo Esquerda ]</div>", unsafe_allow_html=True)

with col_titulo:
    st.markdown("<h1 style='text-align: center; font-size: 26px; margin-bottom: 0px;'>Utilização do Centro Cirúrgico do Risoleta Neves — Análise de Sensibilidade</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 13px; margin-top: 5px;'>"
        "Painel interativo, calibrado com dados reais dos logs de simulação, para responder: "
        "o que acontece com a utilização do CC se o volume de cirurgias aumentar/diminuir, se o tempo médio "
        "cirúrgico aumentar/diminuir, se as cirurgias no horário estendido (19h–22h) aumentarem/diminuírem, "
        "e se as cirurgias de sábado aumentarem/diminuírem — mantendo fixas as 5 salas cirúrgicas. "
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
    caminho_logo_dir = "../../figs/logo_risoleta.png"  # Ajuste para o seu arquivo ou URL
    if Path(caminho_logo_dir).exists():
        st.image(caminho_logo_dir, width=180)
    else:
        st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>[ Logo Direita ]</div>", unsafe_allow_html=True)

st.divider()

DEFAULT_SCENARIOS = {
    14: "cc_event_log-14.csv",
    16: "cc_event_log-16.csv",
    18: "cc_event_log-18.csv",
    20: "cc_event_log-20.csv",
}

ROOMS = 5
HOURS_PADRAO = 16                     # disponibilidade diária de cada sala: 07h–23h (referência financeira do HRTN)
STD_END_H = 19                        # a partir daqui já contamos como "horário estendido" (19h–22h)
EXT_END_H = 22
SATURDAY_DOW = 3                      # dia da semana (no ciclo de 7 dias do log) com menor volume = sábado


# ---------------------------------------------------------------------
# CARREGAMENTO E EXTRAÇÃO DE MÉTRICAS REAIS DO EVENT LOG
# ---------------------------------------------------------------------
def _reduce_to_last_n_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Mantém apenas os últimos n_days de atividade simulada.

    Jornadas completas são preservadas: qualquer case_id que tenha pelo menos
    um evento dentro da janela mantém *todos* os seus eventos.
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

    if "case_id" in df.columns:
        case_ids_in_window = df.loc[df["timestamp"] >= min_ts, "case_id"].unique()
        return df[df["case_id"].isin(case_ids_in_window)].copy()

    return df[df["timestamp"] >= min_ts].copy()


@st.cache_data
def compute_cc_metrics(file_bytes: bytes, scenario_target: int, n_days: int = 365) -> dict:
    df = pd.read_csv(io.BytesIO(file_bytes))
    required_cols = {"case_id", "activity", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"colunas ausentes no CSV: {', '.join(sorted(missing))}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["case_id", "activity", "timestamp"])
    df["case_id"] = df["case_id"].astype(str)
    df["activity"] = df["activity"].astype(str)

    # Recorte temporal (últimos n_days), preservando jornadas completas
    df = _reduce_to_last_n_days(df, n_days)

    seize = df[df["activity"] == "Seize_Sala_CC"][["case_id", "timestamp"]].rename(columns={"timestamp": "start_time"})
    release = df[df["activity"] == "Release_Sala_CC"][["case_id", "timestamp"]].rename(columns={"timestamp": "end_time"})
    pairs = pd.merge(seize, release, on="case_id", how="inner")
    pairs["duration_min"] = (pairs["end_time"] - pairs["start_time"]).clip(lower=0)

    pairs["day"] = (pairs["start_time"] // 1440).astype(int)
    pairs["dow"] = pairs["day"] % 7
    pairs["hour"] = (pairs["start_time"] % 1440) / 60.0
    pairs["is_saturday"] = pairs["dow"] == SATURDAY_DOW
    pairs["is_extended"] = (~pairs["is_saturday"]) & (pairs["hour"] >= STD_END_H) & (pairs["hour"] < EXT_END_H)

    # Duração efetiva da janela pedida pelo usuário (não o span bruto do DF,
    # que pode ser um pouco maior por causa da preservação de jornadas).
    simulation_days = float(n_days)
    if not df.empty and "timestamp" in df.columns:
        observed_span_days = (df["timestamp"].max() - df["timestamp"].min()) / 1440.0
        # Se o log original for mais curto que n_days, usa o que existe
        if observed_span_days < n_days and observed_span_days > 0:
            simulation_days = float(observed_span_days)

    n_weekdays = simulation_days * 5 / 7
    n_saturdays = simulation_days * 1 / 7

    weekday_pairs = pairs[~pairs["is_saturday"]]
    saturday_pairs = pairs[pairs["is_saturday"]]
    extended_pairs = weekday_pairs[weekday_pairs["is_extended"]]

    mean_duration_min = pairs["duration_min"].mean() if not pairs.empty else 120.0

    # "Cirurgias/dia" que aparece no cabeçalho = média real do log (bate com a meta do cenário,
    # ex.: cenário "16" mostra ~16, não um número inflado de dia útil).
    overall_per_day = len(pairs) / simulation_days if simulation_days > 0 else 0.0

    extended_per_weekday = len(extended_pairs) / n_weekdays if n_weekdays > 0 else 0.0
    standard_per_weekday = max(len(weekday_pairs) / n_weekdays - extended_per_weekday, 0.0) if n_weekdays > 0 else 0.0
    saturday_per_saturday = len(saturday_pairs) / n_saturdays if n_saturdays > 0 else 0.0

    return {
        "scenario_target": scenario_target,
        "n_days": n_days,
        "pairs": pairs,
        "total_surgeries": len(pairs),
        "simulation_days": simulation_days,
        "mean_duration_min": mean_duration_min,
        "overall_per_day": overall_per_day,
        "standard_per_weekday": standard_per_weekday,   # cirurgias/dia útil fora do horário estendido (07h–19h)
        "extended_per_weekday": extended_per_weekday,    # cirurgias/dia útil entre 19h–22h
        "saturday_per_saturday": saturday_per_saturday,  # cirurgias por sábado
    }


resolved_paths = {target: path for target, path in DEFAULT_SCENARIOS.items() if Path(path).exists()}

# ---------------------------------------------------------------------
# MENU LATERAL — JANELA DE ANÁLISE + CENÁRIOS
# ---------------------------------------------------------------------
st.sidebar.header("Janela de análise")

n_days = st.sidebar.slider(
    "Janela de análise (dias)",
    min_value=1,
    max_value=365,
    value=365,
    step=1,
    help="Quantidade de dias finais do event log a serem considerados na análise. "
         "O recorte é aplicado a todos os cenários (14/16/18/20 cirurgias/dia) simultaneamente. "
         "Jornadas completas de pacientes que tocam a janela são preservadas. "
         "Valores muito altos (ex.: > 90 dias) podem tornar a aplicação mais lenta.",
)

st.sidebar.caption(
    f"Análise restrita aos **últimos {n_days} dia{'s' if n_days != 1 else ''}** "
    "de cada log de eventos."
)

st.sidebar.divider()
st.sidebar.header("Cenários (análise de CC)")
st.sidebar.caption(
    "Por padrão, o app carrega os logs de eventos e relatórios para demanda por 14, 16, 18 e 20 "
    "pacientes/dia presentes na mesma pasta. Você pode substituir qualquer cenário por um arquivo próprio."
)
with st.sidebar.expander("Substituir logs de eventos (opcional)"):
    uploaded_files = {}
    for target in sorted(DEFAULT_SCENARIOS.keys()):
        uploaded_files[target] = st.file_uploader(
            f"Cenário ~{target} cirurgias/dia",
            type="csv",
            key=f"upload_{target}",
            help="Envie um cc_event_log-*.csv com as mesmas colunas (case_id, activity, timestamp, ...) para substituir este cenário.",
        )

scenario_sources = {}  # target -> (bytes, label)
for target in sorted(DEFAULT_SCENARIOS.keys()):
    up = uploaded_files.get(target)
    if up is not None:
        scenario_sources[target] = (up.getvalue(), f"arquivo enviado ({up.name})")
    elif target in resolved_paths:
        scenario_sources[target] = (Path(resolved_paths[target]).read_bytes(), "arquivo padrão da pasta")

if not scenario_sources:
    st.error("Nenhum arquivo `cc_event_log-*.csv` encontrado na pasta e nenhum arquivo foi enviado.")
    st.stop()

available_targets = sorted(scenario_sources.keys())
all_scenarios = {}
for t in available_targets:
    try:
        all_scenarios[t] = compute_cc_metrics(scenario_sources[t][0], t, n_days=n_days)
    except Exception as exc:
        st.sidebar.error(f"Não foi possível ler o arquivo do cenário ~{t} cirurgias/dia: {exc}")
available_targets = sorted(all_scenarios.keys())
if not available_targets:
    st.error("Nenhum dos arquivos carregados pôde ser processado. Verifique se as colunas esperadas (case_id, activity, timestamp, ...) estão presentes.")
    st.stop()
scenario_labels = {t: scenario_sources[t][1] for t in available_targets}


# ---------------------------------------------------------------------
# MODELO DE CAPACIDADE (5 salas fixas, 16h/dia — referência financeira do HRTN)
# ---------------------------------------------------------------------
def simulate_utilization(base, vol_pct, dur_pct, ext_pct, sat_pct, saturday_open):
    """Aplica as 4 variações sobre as métricas reais do cenário-base e calcula a
    utilização semanal do CC, respeitando a capacidade fixa de 5 salas × 16h/dia."""
    duration = base["mean_duration_min"] * (1 + dur_pct / 100.0)
    standard = base["standard_per_weekday"] * (1 + vol_pct / 100.0)
    extended = base["extended_per_weekday"] * (1 + vol_pct / 100.0) * (1 + ext_pct / 100.0)
    saturday = base["saturday_per_saturday"] * (1 + vol_pct / 100.0) * (1 + sat_pct / 100.0)

    cap_day_min = ROOMS * HOURS_PADRAO * 60
    weekly_capacity = cap_day_min * 5 + (cap_day_min * 1 if saturday_open else 0)
    weekly_workload = (standard + extended) * duration * 5 + saturday * duration * 1

    utilization = (weekly_workload / weekly_capacity * 100) if weekly_capacity > 0 else 0.0
    surgeries_per_day = ((standard + extended) * 5 + saturday * 1) / 7.0
    return {
        "utilization": min(max(utilization, 0.0), 200.0),
        "duration": duration,
        "standard": standard,
        "extended": extended,
        "saturday": saturday,
        "surgeries_per_day": surgeries_per_day,
        "surgeries_per_weekday": standard + extended,
    }


# ---------------------------------------------------------------------
# MENU LATERAL (Controles Interativos)
# ---------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("1. Cenário Base (log real de simulação)")

selected_target = st.sidebar.selectbox(
    "Selecione o cenário",
    available_targets,
    index=0,
    format_func=lambda t: f"Meta ~{t} cirurgias/dia",
    label_visibility="collapsed",
)
current = all_scenarios[selected_target]

st.sidebar.caption(
    f"📄 {scenario_labels[selected_target].capitalize()} — **{current['total_surgeries']}** cirurgias reais simuladas ao longo de "
    f"**{current['simulation_days']:.0f} dias** — média de **{current['overall_per_day']:.1f} cirurgias/dia**, "
    f"duração média de {current['mean_duration_min']:.0f} min ({current['mean_duration_min']/60:.1f}h), "
    f"capacidade de 5 salas × {HOURS_PADRAO}h/dia (referência financeira do HRTN)."
)

st.sidebar.divider()
st.sidebar.subheader("2. Simule as Variações")

# -----------------------------------------------------------------
# SLIDERS SINCRONIZADOS (barra lateral + parte superior dos gráficos)
# -----------------------------------------------------------------
# Cada variável de simulação tem DOIS widgets equivalentes: um na barra
# lateral e outro logo acima do gráfico correspondente. Os dois ficam
# sempre sincronizados: mexer em qualquer um dos dois atualiza o outro
# automaticamente (via callback), então não importa qual o usuário usa.
SLIDER_SPECS = {
    "vol_pct": dict(
        icon="📈",
        label="Variação no volume geral de cirurgias (%)",
        short_label="① Volume geral de cirurgias (%)",
        min_value=-50, max_value=100, step=5,
        help="Aumenta ou reduz proporcionalmente a demanda cirúrgica geral (todos os turnos).",
    ),
    "dur_pct": dict(
        icon="⏱️",
        label="Variação no tempo médio cirúrgico (%)",
        short_label="② Tempo médio cirúrgico (%)",
        min_value=-30, max_value=50, step=5,
        help="Cirurgias mais rápidas liberam capacidade; mais longas consomem mais sala-tempo.",
    ),
    "ext_pct": dict(
        icon="🌙",
        label="Variação nas cirurgias entre 19h–22h (%)",
        short_label="③ Cirurgias 19h–22h (%)",
        min_value=-100, max_value=200, step=10,
        help="Ajusta especificamente a quantidade de cirurgias agendadas no horário estendido (dentro das 16h já disponíveis).",
    ),
    "sat_pct": dict(
        icon="📅",
        label="Variação nas cirurgias de sábado (%)",
        short_label="④ Cirurgias de sábado (%)",
        min_value=-100, max_value=200, step=10,
        help="Ajusta especificamente a quantidade de cirurgias agendadas aos sábados.",
    ),
}

for _name in SLIDER_SPECS:
    st.session_state.setdefault(f"{_name}_sidebar", 0)
    st.session_state.setdefault(f"{_name}_chart", 0)


def _make_sync_callback(name: str, source_suffix: str, target_suffix: str):
    """Copia o valor do widget de origem para o widget espelho, ANTES do rerun,
    para que os dois sliders (barra lateral e topo do gráfico) fiquem sempre iguais."""
    def _cb():
        st.session_state[f"{name}_{target_suffix}"] = st.session_state[f"{name}_{source_suffix}"]
    return _cb


def synced_slider(name: str, container, key_suffix: str, mirror_suffix: str, use_short_label: bool = False):
    """Renderiza um slider sincronizado (via st.session_state) com seu par."""
    spec = SLIDER_SPECS[name]
    label = f"{spec['icon']} {spec['short_label'] if use_short_label else spec['label']}"
    return container.slider(
        label,
        min_value=spec["min_value"],
        max_value=spec["max_value"],
        step=spec["step"],
        format="%d%%",
        help=spec["help"],
        key=f"{name}_{key_suffix}",
        on_change=_make_sync_callback(name, key_suffix, mirror_suffix),
    )


vol_pct = synced_slider("vol_pct", st.sidebar, "sidebar", "chart")
dur_pct = synced_slider("dur_pct", st.sidebar, "sidebar", "chart")
ext_pct = synced_slider("ext_pct", st.sidebar, "sidebar", "chart")
sat_pct = synced_slider("sat_pct", st.sidebar, "sidebar", "chart")

st.sidebar.divider()
st.sidebar.subheader("3. Capacidade Instalada aos Sábados")
saturday_open = st.sidebar.checkbox(
    "Sábado com capacidade dedicada (5 salas × 16h)", value=True,
    help="Os dados reais já mostram cirurgias ocorrendo aos sábados. Desmarque para simular o que "
         "acontece se essa demanda de sábado precisar ser absorvida de segunda a sexta.",
)
st.sidebar.caption(
    "De segunda a sexta, a capacidade instalada é sempre 5 salas × 16h/dia (07h–23h), a mesma referência "
    "usada pelo financeiro do HRTN. O horário 19h–22h já está **dentro** dessa janela — não é uma capacidade extra."
)

# ---------------------------------------------------------------------
# CÁLCULO DO CENÁRIO ATUAL
# ---------------------------------------------------------------------
result = simulate_utilization(current, vol_pct, dur_pct, ext_pct, sat_pct, saturday_open)
baseline_result = simulate_utilization(current, 0, 0, 0, 0, saturday_open)

# ---------------------------------------------------------------------
# VISUALIZAÇÃO PRINCIPAL
# ---------------------------------------------------------------------
st.header(f"Análise de Impacto — Cenário Base 'Meta ~{selected_target} cirurgias/dia'")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cirurgias/dia (média)", f"{result['surgeries_per_day']:.1f}", delta=f"{vol_pct}%")
col2.metric("Tempo médio no Centro Cirúrgico", f"{result['duration']/60:.2f}h", delta=f"{dur_pct}%", delta_color="inverse")
col3.metric(
    "Utilização do Centro Cirúrgico", f"{result['utilization']:.1f}%",
    delta=f"{result['utilization'] - baseline_result['utilization']:.1f} p.p. vs. dado real",
    delta_color="inverse" if result['utilization'] > baseline_result['utilization'] else "normal",
)
col4.metric("Capacidade Ativa", f"5 salas × {HOURS_PADRAO}h" + (" + Sábado" if saturday_open else " (seg–sex)"))

if result["utilization"] > 100:
    st.error("⚠️ **Alerta de Sobrecarga:** a demanda excede 100% da capacidade instalada (5 salas, 16h/dia), gerando filas de espera e gargalos.")
elif result["utilization"] > 85:
    st.warning("⚠️ **Atenção:** o Centro Cirúrgico opera em zona de alta ocupação, com margem de folga reduzida.")
else:
    st.success("✅ **Operação Saudável:** a taxa de ocupação está dentro de limites seguros para as 5 salas disponíveis.")

st.caption(
    f"Horizonte de análise: **últimos {current['simulation_days']:.0f} dias** do log de eventos "
    f"(configurável no slider da barra lateral)."
)

st.divider()


# ---------------------------------------------------------------------
# Visualização do fluxo (HTML)
# ---------------------------------------------------------------------
st.subheader("Visualização do Fluxo do Processo")
st.caption("Visualização interativa do fluxo da simulação.")

try:
    import streamlit.components.v1 as components
    with open("cc.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=True)
except FileNotFoundError:
    st.info("Arquivo 'cc.html' não encontrado. Coloque o arquivo HTML na mesma pasta para exibi-lo aqui.")

st.divider()

# ---------------------------------------------------------------------
# GRÁFICOS — UMA RESPOSTA VISUAL PARA CADA UMA DAS 4 PERGUNTAS
# ---------------------------------------------------------------------
st.subheader("📊 Análises com os dados das 5 Salas Cirúrgicas")


def nearest(sweep_values, value):
    """Retorna o ponto da varredura mais próximo do valor atual do slider,
    para sempre destacar uma barra/ponto mesmo quando o slider não bate exatamente
    com um dos valores fixos do eixo x."""
    return min(sweep_values, key=lambda x: abs(x - value))


c_g1, c_g2 = st.columns(2)

with c_g1:
    vol_pct = synced_slider("vol_pct", st, "chart", "sidebar", use_short_label=True)
    sweep = [-50, -25, 0, 25, 50, 75, 100]
    # Os demais parâmetros ficam nos valores ATUAIS dos outros sliders — os gráficos
    # não são independentes: reduzir a duração, por exemplo, baixa toda esta curva.
    utils = [simulate_utilization(current, v, dur_pct, ext_pct, sat_pct, saturday_open)["utilization"] for v in sweep]
    vol_marker = nearest(sweep, vol_pct)
    colors = ["#ff7f0e" if v == vol_marker else "#636efa" for v in sweep]
    fig_vol = go.Figure(go.Bar(
        x=[f"{v:+d}%" for v in sweep], y=utils, marker_color=colors,
        text=[f"{u:.0f}%" for u in utils], textposition="outside",
    ))
    fig_vol.add_hline(y=100, line_dash="dash", line_color="crimson", annotation_text="Limite de 100%")
    fig_vol.update_layout(title="① Se o volume geral (médio) de cirurgias subir/cair", xaxis_title="Variação (%) do volume de cirurgias/dia", yaxis_title="Utilização do CC (%)")
    st.plotly_chart(fig_vol, use_container_width=True)
    st.caption(f"🔍 Duração, horário estendido e sábado ficam nos valores atuais dos outros sliders — mexer neles desloca esta curva para cima ou para baixo. A barra laranja (~{vol_marker:+d}%) é a mais próxima do valor atual do slider ① ({vol_pct:+d}%).")


with c_g2:
    dur_pct = synced_slider("dur_pct", st, "chart", "sidebar", use_short_label=True)
    sweep_d = [-30, -15, 0, 15, 30, 50]
    
    utils_d = []
    absolute_durations = []
    
    for d in sweep_d:
        res_d = simulate_utilization(current, vol_pct, d, ext_pct, sat_pct, saturday_open)
        utils_d.append(res_d["utilization"])
        absolute_durations.append(res_d["duration"])  # duração em minutos
        
    dur_marker = nearest(sweep_d, dur_pct)
    
    # Prepara as listas separadas para a linha base e para o ponto de destaque (laranja)
    x_vals = [f"{d:+d}%" for d in sweep_d]
    
    normal_x, normal_y, normal_text = [], [], []
    orange_x, orange_y, orange_text = [], [], []
    
    for d, x_str, u, dur in zip(sweep_d, x_vals, utils_d, absolute_durations):
        total_minutes = int(round(dur))
        hours = total_minutes // 60
        minutes = total_minutes % 60
        time_str_formatted = f"{hours}h{minutes:02d}min" if hours > 0 else f"{minutes}min"
        decimal_hours_str = f"{dur / 60:.2f}h"
        
        if d == dur_marker:
            orange_x.append(x_str)
            orange_y.append(u)
            # Rótulo detalhado e explícito para o ponto laranja
            orange_text.append(f"<b>{u:.0f}%</b><br>Tempo: {decimal_hours_str}<br>({time_str_formatted})")
        else:
            normal_x.append(x_str)
            normal_y.append(u)
            normal_text.append(f"{u:.0f}%<br>({decimal_hours_str})")

    # Constrói o gráfico usando graph_objects para controle total de camadas e textos
    fig_dur = go.Figure()
    
    # Linha conectora de fundo
    fig_dur.add_trace(go.Scatter(
        x=x_vals, y=utils_d,
        mode="lines",
        line=dict(width=3, color="#1f77b4"),
        showlegend=False,
        hoverinfo="skip"
    ))
    
    # Pontos normais (azuis) com texto simplificado
    fig_dur.add_trace(go.Scatter(
        x=normal_x, y=normal_y,
        mode="markers+text",
        text=normal_text,
        textposition="bottom center",
        marker=dict(size=10, color="#1f77b4", line=dict(width=1, color="white")),
        textfont=dict(size=10, color="#333333"),
        showlegend=False
    ))
    
    # Ponto Laranja atual com destaque visual superior e rótulo explícito completo
    fig_dur.add_trace(go.Scatter(
        x=orange_x, y=orange_y,
        mode="markers+text",
        text=orange_text,
        textposition="top center",
        marker=dict(size=18, color="#ff7f0e", line=dict(width=2, color="white")),
        textfont=dict(size=12, color="#d9381e", family="Arial Black"),
        showlegend=False
    ))

    fig_dur.update_layout(
        title="② Se o tempo (médio) cirúrgico subir/cair",
        xaxis_title="Variação (%) do tempo cirúrgico",
        yaxis_title="Utilização do CC (%)",
        margin=dict(t=50, b=30)
    )
    
    fig_dur.add_hline(y=100, line_dash="dash", line_color="crimson", annotation_text="Limite de 100%")
    st.plotly_chart(fig_dur, use_container_width=True)
    st.caption(f"🔍 O ponto laranja destaca o valor atual do slider ② ({dur_pct:+d}%), exibindo explicitamente o tempo médio correspondente no CC em formato decimal e horas/minutos.")


c_g3, c_g4 = st.columns(2)


with c_g3:
    ext_pct = synced_slider("ext_pct", st, "chart", "sidebar", use_short_label=True)
    sweep_e = [-100, -50, 0, 50, 100, 150, 200]
    
    # Os demais parâmetros ficam nos valores ATUAIS dos outros sliders.
    utils_e = [simulate_utilization(current, vol_pct, dur_pct, e, sat_pct, saturday_open)["utilization"] for e in sweep_e]
    
    # Identifica a barra mais próxima do valor atual do slider de horário estendido
    ext_marker = nearest(sweep_e, ext_pct)
    
    # Aplica a cor laranja apenas na barra mais próxima do valor do slider, seguindo o padrão do gráfico 1
    colors_e = ["#ff7f0e" if e == ext_marker else "#2ca02c" for e in sweep_e]
    
    fig_ext = go.Figure(go.Bar(
        x=[f"{e:+d}%" for e in sweep_e], 
        y=utils_e, 
        marker_color=colors_e,
        text=[f"{u:.0f}%" for u in utils_e], 
        textposition="outside",
    ))
    
    fig_ext.add_hline(y=100, line_dash="dash", line_color="crimson", annotation_text="Limite de 100%")
    fig_ext.update_layout(
        title="③ Se o volume (médio) de cirurgias entre 19h–22h subir/cair", 
        xaxis_title="Variação (%) no Volume 19h–22h", 
        yaxis_title="Utilização do CC (%)"
    )
    st.plotly_chart(fig_ext, use_container_width=True)
    st.caption(f"🔍 Volume, duração e sábado ficam nos valores atuais dos outros sliders. A barra laranja (~{ext_marker:+d}%) é a mais próxima do valor atual do slider ③ ({ext_pct:+d}%). Como 19h–22h já está dentro das 16h disponíveis, o impacto vem do volume extra agendado ali, somado ao restante da demanda.")


with c_g4:
    sat_pct = synced_slider("sat_pct", st, "chart", "sidebar", use_short_label=True)
    rows2 = []
    # Sábado varre sua própria variável; os demais ficam nos valores ATUAIS dos outros sliders.
    for s in sweep_e:
        for aberto in [False, True]:
            u = simulate_utilization(current, vol_pct, dur_pct, ext_pct, s, aberto)["utilization"]
            rows2.append({
                "variação": f"{s:+d}%", 
                "utilização": u, 
                "sábado": "Salas abertas" if aberto else "Salas fechadas"
            })
            
    df_sab = pd.DataFrame(rows2)
    
    fig_sab = px.bar(
        df_sab, 
        x="variação", 
        y="utilização", 
        color="sábado", 
        barmode="group",
        title="④ Se o volume (médio) de cirurgias aos sábados subir/cair",
        labels={"variação": "Variação (%) no Volume de Sábado", "utilização": "Utilização do CC (%)"},
        color_discrete_map={"Salas abertas": "#2ca02c", "Salas fechadas": "#d62728"},
        text=[f"{u:.0f}%" for u in df_sab["utilização"]] # Adiciona o percentual no topo de cada barra para base de comparação
    )
    
    # Posiciona o texto de forma limpa na parte externa superior das barras agrupadas
    fig_sab.update_traces(textposition="outside")
    
    fig_sab.add_hline(y=100, line_dash="dash", line_color="crimson", annotation_text="Limite de 100%")
    
    # Ajusta margem superior para evitar corte dos textos nas barras mais altas
    fig_sab.update_layout(margin=dict(t=50, b=30))
    
    st.plotly_chart(fig_sab, use_container_width=True)
    st.caption("🔍 Sem capacidade dedicada aos sábados, as cirurgias de sábado pressionam a agenda de segunda a sexta. Os percentuais no topo de cada barra facilitam a comparação direta com o comportamento do gráfico ③.")

st.divider()

# ---------------------------------------------------------------------
# INSIGHTS COMPLEMENTARES A PARTIR DOS DADOS REAIS (todos os cenários)
# ---------------------------------------------------------------------
st.subheader("🔎 Raio-X dos dados do Centro Cirúrgico")

c_i1, c_i2 = st.columns(2)


with c_i1:
    real_rows = []
    # Percorre todos os alvos disponíveis para garantir que o eixo X seja sempre constante
    for t in sorted(available_targets):
        b = all_scenarios[t]
        u = simulate_utilization(b, 0, 0, 0, 0, saturday_open)["utilization"]
        real_rows.append({
            "cenário": f"~{t} cir/dia", 
            "target_val": t, # Auxilia na ordenação correta se necessário
            "utilização": u, 
            "selecionado": t == selected_target
        })
    
    df_real = pd.DataFrame(real_rows)
    
    # Criação do gráfico com categoria fixa para todos os cenários carregados
    fig_real = px.bar(
        df_real, 
        x="cenário", 
        y="utilização",
        color="selecionado", 
        color_discrete_map={True: "#ff7f0e", False: "#636efa"},
        text=[f"{u:.0f}%" for u in df_real["utilização"]],
        title="Utilização real observada em cada log simulado (sem ajustes)",
        labels={"cenário": "Cenário Simulado", "utilização": "Utilização do CC (%)"},
    )
    
    # Força o eixo X a exibir todas as categorias na ordem correta, impedindo que mude ao trocar de cenário
    fig_real.update_layout(
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=[f"~{t} cir/dia" for t in sorted(available_targets)]
        ),
        showlegend=False
    )
    
    fig_real.add_hline(y=100, line_dash="dash", line_color="crimson")
    st.plotly_chart(fig_real, use_container_width=True)
    st.caption("🔍 Compara os logs reais entre si com a capacidade ativa atual, mantendo o eixo X padronizado para todos os cenários.")

with c_i2:
    pairs = current["pairs"]
    hours = pairs["hour"].values
    fig_hist = px.histogram(
        x=hours, nbins=48,
        title=f"Distribuição horária real das cirurgias — cenário 'Meta ~{selected_target}'",
        labels={"x": "Hora do dia", "y": "Nº de cirurgias (todo o período)"},
    )
    fig_hist.add_vrect(x0=STD_END_H, x1=EXT_END_H, fillcolor="orange", opacity=0.15, line_width=0,
                        annotation_text="Horário estendido (19h–22h)", annotation_position="top left")
    fig_hist.add_vrect(x0=0, x1=7, fillcolor="gray", opacity=0.1, line_width=0)
    fig_hist.add_vrect(x0=EXT_END_H, x1=24, fillcolor="gray", opacity=0.1, line_width=0)
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("🔍 Mostra onde, ao longo do dia, o cenário selecionado já concentra suas cirurgias — inclusive quanto já ocorre no horário estendido.")

st.caption(
    f"ℹ️ **Calibração:** \"Cirurgias/dia\" é a média real do log (bate com a meta do cenário selecionado). "
    f"A capacidade considerada é de 5 salas × {HOURS_PADRAO}h/dia (07h–23h), a mesma referência de "
    f"disponibilidade usada pelo financeiro do HRTN, que resulta em utilização de referência na faixa de "
    f"50%–90% para os cenários simulados. "
    f"Diferente do CTI (onde a capacidade pôde ser ampliada com mais leitos), o Centro Cirúrgico está fixado em "
    f"5 salas; por isso as alavancas de ajuste aqui são de **demanda e de janela de funcionamento** "
    f"(volume, duração, horário estendido e sábado), não de número de salas. "
    f"Horizonte de análise atual: últimos **{current['simulation_days']:.0f} dias**."
)



# 1. Perguntas de diagnóstico ("onde estamos hoje")

# Respondidas pelos KPIs do topo e pelo Raio-X dos dados (parte final do app):

# "Qual a nossa utilização atual do centro cirúrgico?"
# "Quantas cirurgias fazemos por dia, em média, e qual o tempo médio de cada uma?"
# "Estamos operando com folga ou já perto do limite de capacidade?"
# "Como a utilização varia entre os diferentes cenários de demanda (14, 16, 18, 20 cirurgias/dia)?"
# "Em que horários do dia as cirurgias já se concentram? Quanto já usamos do horário estendido (19h–22h)?"
# 2. Perguntas de sensibilidade ("e se...") — o coração da ferramenta

# Cada uma bate direto em um dos 4 gráficos/sliders:

# ① Volume geral: "Se a demanda por cirurgias crescer X%, ainda conseguimos atender com as 5 salas atuais? Em que ponto ultrapassamos 100% de utilização?"
# ② Tempo médio cirúrgico: "Se conseguirmos reduzir o tempo médio de cirurgia (ex.: com nova técnica/protocolo), quanto isso libera de capacidade? E se o tempo aumentar (casos mais complexos), qual o impacto?"
# ③ Horário estendido (19h–22h): "Vale a pena incentivar mais cirurgias nesse horário para absorver crescimento de demanda, sem abrir mais salas?"
# ④ Sábado: "Se aumentarmos o volume de cirurgias aos sábados, isso sobrecarrega a semana ou temos capacidade dedicada suficiente?"
# 3. Perguntas de decisão operacional

# Respondidas pelo checkbox "Sábado com capacidade dedicada" combinado com o gráfico ④:

# "O que acontece com a operação de segunda a sexta se deixarmos de operar aos sábados?"
# "Manter sábado aberto é o que sustenta nossa margem de folga atual?"
# 4. Perguntas combinadas (cenários compostos)

# Como os sliders interagem entre si, dá pra simular perguntas mais realistas:

# "Se a demanda crescer 20% E reduzirmos o tempo médio em 10%, ainda ficamos sustentáveis?"
# "Qual a combinação de alavancas (volume, duração, horário estendido, sábado) que nos mantém abaixo de 85% de utilização?"
# 5. Perguntas sobre robustez/confiabilidade dos dados

# Respondidas pela janela de análise (slider "dias") e pelos rótulos de origem dos dados:

# "Esses números são de qual período? Dá pra comparar janelas diferentes (últimos 30 dias vs. ano inteiro)?"
# "Os cenários (14/16/18/20) são baseados em quê — são projeções ou dados observados?"
# Importante: o que a ferramenta não responde

# Vale alinhar expectativas antes que perguntem — para não parecer que a ferramenta falhou:

# Custo financeiro (ela mostra % de utilização de sala, não R$/hora, custo de equipe extra, etc.)
# Alocação por especialidade/equipe (não sabe quem opera onde, nem gargalos de equipe médica/enfermagem)
# Causalidade (o modelo aplica variações proporcionais sobre dados reais — não explica por que a demanda mudaria)
# Número de salas — está fixo em 5; a ferramenta simula demanda e janela de funcionamento, não expansão física
