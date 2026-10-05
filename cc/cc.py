# =====================================================================
# FILE: hrtn.py
# =====================================================================
# -*- coding: utf-8 -*-
import random
import math
import sys
import simpy                          # ← required for the Sala_CC capacity gate
import matplotlib.pyplot as plt
import numpy as np  
from desk.stats.factorial import FactorialExperiment
from desk.stats.replication import ReplicationFramework    
from desk.analytics.financial import FinancialAnalyzer
from desk.validation.resource_validator import ResourceValidator
from desk.core.simulation_model import SimulationModel
from desk.core.entity import Entity, EventLogger
from desk.blocks.create_block import CreateBlock
from desk.blocks.process_block import ProcessBlock, MultiProcessBlock
from desk.blocks.decide_block import DecideBlock
from desk.blocks.dispose_block import DisposeBlock
from desk.analytics.metrics import MetricsCollector
from desk.analytics.reporting import SimulationReporter
from desk.analytics.plotting import SimulationPlotter
from desk.validation.stability import StabilityAnalyzer
from desk.validation.warmup import WarmUpAnalyzer
from desk.config.simulation_config import SimulationConfig
from desk.visualization.interface import run_visualization

from desk.analytics.report_builder import MasterReportBuilder

# ================================================================
# Each ACD model is implemented here
# ================================================================
# desk-sim -m cc/cc.py --mode visualization
# desk-sim -m cc/cc.py --mode single
# desk-sim -m cc/cc.py --mode replications
# desk-sim -m cc/cc.py --mode factorial

# ================================================================
# Desk-sim: DIST-FIT Which is the best data distribution?
# ================================================================
# desk-distfit -d src/input_data/1_int_cheg.txt --max-sample 500
# desk-distfit -d src/input_data/2_adm_conf.txt --max-sample 500
# desk-distfit -d src/input_data/3_ato_anestetico.txt --max-sample 500
# desk-distfit -d src/input_data/cir_p.txt --max-sample 500
# desk-distfit -d src/input_data/cir_m.txt --max-sample 500
# desk-distfit -d src/input_data/cir_g.txt --max-sample 500
# desk-distfit -d src/input_data/3_pos_cir.txt --max-sample 500



# ================================================================
# ESCOPO GLOBAL
# ================================================================
# Scenario-planning knob: change this value to explore different
# demand levels (e.g. 14 = baseline, 18 = growth scenario, 30 = stress test).
# WEEKDAY_FACTORS below express RELATIVE weights across days-of-week and
# are intentionally independent of this number — weekends will always be
# ~55–69 % of whatever daily volume you set here.

# BASE_ARRIVALS_PER_DAY = 14    
BASE_ARRIVALS_PER_DAY = 16    # Base
# BASE_ARRIVALS_PER_DAY = 18    
# BASE_ARRIVALS_PER_DAY = 20    
# BASE_ARRIVALS_PER_DAY = 22    
# BASE_ARRIVALS_PER_DAY = 19      # Cenario 20% aumento da demanda

# Capacidades Padrão (Default)
DEFAULT_CAPACITIES = {
    "Enfermeiro": 1, 
    "Farmacia": 2, 
    "Tec_Enfermagem": 6, #  5 nas salas + 1 corredor + sala onda + 3 SRPA Tem um ferista cobrindo alguem 
    "Eq_Assistencial_CTI": 1,
    "Eq_Medica": 5,  # + 1 equipe sala da onda
    "Anestesista": 5,  # + 1 anestesista sala da onda
    "Tec_Radiologia": 2, 
    "Eq_Radiologia": 2, # Tem 4, mas usa 2 no máximo simultaneamente
    "Func_CME": 1,  
    "Eq_Higienizacao": 2
}

# ================================================================
# BACKGROUND WORKLOAD CONFIGURATION
# ================================================================
# Purpose: The simulation models ONLY the surgical pathway events.
#   In reality staff spend most of their shift on activities NOT in
#   scope (documentation, training, equipment checks, stock replen-
#   ishment, ward rounds, administrative tasks, etc.).
#
# Without background load SimPy reports utilization only for the
# modeled activities → numbers are technically correct but look
# misleadingly low when presented as "team utilization".
#
# How it works:
#   A background_worker generator runs for every unit of every
#   resource.  It repeatedly: sleeps (inter-task gap) then seizes
#   the resource for (task_duration) minutes — representing a
#   generic "unmodeled task".  Both times are Exponential so the
#   process is a simple M/G/1 background queue.
#
# Calibration guidance:
#   background_utilization_fraction  =  task_mean / (task_mean + gap_mean)
#   Choose task_mean + gap_mean so the cycle length feels realistic
#   (e.g. tasks of ~30 min interleaved with gaps of ~30 min → 50%).
#
# The numbers below are ILLUSTRATIVE estimates.
#   Replace them with values from your time-and-motion study or
#   expert-elicitation workshop before using for decision-making.
#
# Key: resource name
# task_mean_min : mean duration of one background task (minutes)
# gap_mean_min  : mean idle gap between background tasks (minutes)
# background_util = task / (task + gap)  [approximate, per unit]
# ================================================================
BACKGROUND_WORKLOAD = {
    #                      task_mean  gap_mean   ≈ bg_util
    # Enfermeiro: Transporte de pacientes, monitoramento na SRPA, preparação de salas, 
    # organização de materiais, programação cirúrgica, coordenação de equipes, 
    # gestão de faltas de materiais, comunicação com enfermarias e tratamento de incidentes.
    "Enfermeiro":        {"task": 10, "gap": 5},   
    # Farmacia: Além dos medicamentos cirúrgicos: dispensação de medicamentos, 
    # controle de estoque, gestão de medicamentos controlados, reposição de estoques em 
    # outros setores e processamento de devoluções.
    "Farmacia":          {"task": 35, "gap": 15},   
    # Tec_Enfermagem: Auxilia SRPA o tempo todo
    "Tec_Enfermagem":    {"task": 10, "gap": 15},   
    # Eq_Assistencial_CTI: Atividades no CTI
    "Eq_Assistencial_CTI":{"task":3, "gap": 2},   
    # Eq_Medica: Além da cirurgia: registro em prontuário, descrição cirúrgica, 
    # prescrição médica, evolução clínica, codificação no SISREG/SIH/SUS 
    # e comunicação com familiares.
    "Eq_Medica":         {"task": 25, "gap": 35},  
    # Anestesista: Atividades adicionais: avaliação pré-anestésica, 
    # visitas à sala de recuperação pós-anestésica (SRPA), preenchimento de 
    # documentação e reconciliação medicamentosa. 
    "Anestesista":       {"task": 25, "gap": 15},   
    # Tec_Radiologia: Se cirurgia ortopedica, 70% de uso
    "Tec_Radiologia":    {"task": 20, "gap": 15},   
    "Eq_Radiologia":     {"task": 20, "gap": 15},   
    # Func_CME: Recebimento de instrumentais contaminados: lavagem, montagem/preparo 
    # de caixas cirúrgicas, esterilização e armazenamento.
    "Func_CME":          {"task": 10, "gap": 5},   
    # Além da limpeza entre cirurgias, a equipe de higienização realiza: limpeza terminal, 
    # limpeza de corredores e áreas comuns, coleta e descarte de resíduos e limpezas emergenciais.
    "Eq_Higienizacao":   {"task": 30, "gap": 20},   
}


# ----------------------------------------------------------------
# BACKGROUND WORKLOAD SCHEDULE
# ----------------------------------------------------------------
# Controls HOW ACTIVE background workers are during each 2-hour
# time window.  This is the mechanism that makes night utilization
# drop when there are no patient arrivals.
#
# Each resource maps to a list of (start_h, end_h, load_factor).
#   load_factor = 1.0  → normal task/gap from BACKGROUND_WORKLOAD
#   load_factor = 0.5  → workers sleep 2× longer between tasks
#                        (gap_effective = gap_base / load_factor)
#   load_factor = 0.0  → workers do nothing (pure idle); only
#                        a tiny "heartbeat" gap keeps SimPy alive
#
# Resources NOT listed here run at load_factor = 1.0 all day.
#
# Effective background utilization at load_factor f:
#   util ≈ task / (task + gap_base/f)
#
# Example — Enfermeiro (task=45, gap=8) at load_factor=0.0:
#   util ≈ 0%   (no tasks, just idle gap heartbeat of 9999 min)
# ----------------------------------------------------------------
BACKGROUND_SCHEDULE = {
    # Resource           [(start_h, end_h, load_factor), ...]
    "Enfermeiro": [
        ( 0,  7, 0.05),   # night  — near idle
        ( 7, 19, 1.00),   # day    — full background load
        (19, 24, 0.20),   # evening — reduced
    ],
    "Farmacia": [
        ( 0,  7, 0.05),
        ( 7, 19, 1.00),
        (19, 24, 0.20),
    ],
    "Tec_Enfermagem": [
        ( 0,  7, 0.10),
        ( 7, 19, 1.00),
        (19, 24, 0.25),
    ],
    "Eq_Assistencial_CTI": [
        ( 0,  7, 0.10),
        ( 7, 19, 1.00),
        (19, 24, 0.20),
    ],
    "Eq_Medica": [
        ( 0,  7, 0.05),
        ( 7, 19, 1.00),
        (19, 24, 0.20),
    ],
    "Anestesista": [
        ( 0,  7, 0.05),
        ( 7, 19, 1.00),
        (19, 24, 0.15),
    ],
    "Tec_Radiologia": [
        ( 0,  7, 0.05),
        ( 7, 19, 1.00),
        (19, 24, 0.20),
    ],
    "Eq_Radiologia": [
        ( 0,  7, 0.05),
        ( 7, 19, 1.00),
        (19, 24, 0.20),
    ],
    "Func_CME": [
        ( 0,  7, 0.05),
        ( 7, 19, 1.00),
        (19, 24, 0.20),
    ],
    "Eq_Higienizacao": [
        ( 0,  7, 0.10),
        ( 7, 19, 1.00),
        (19, 24, 0.30),
    ],
}

# Set to False to disable background load and restore original behaviour.
ENABLE_BACKGROUND_WORKLOAD = True
# ENABLE_BACKGROUND_WORKLOAD = False

# ---------------------------------------------------------------
# Time-varying resource staffing schedule
# Each resource maps to a list of (start_h, end_h, capacity).
# Resources NOT listed here keep their default capacity unchanged.
# ---------------------------------------------------------------
RESOURCE_SCHEDULE = {
    "Eq_Medica": [
        ( 0,  7, 3), # A confirmar
        ( 7, 19, 5), # Inicio do dio (eletivas + urgencias)        
        (19, 24, 3),        
    ],
    "Anestesista": [
        ( 0,  7, 3), # A confirmar
        ( 7, 19, 5), # Inicio do dio (eletivas + urgencias)        
        (19, 24, 3),        
    ],
    "Enfermeiro": [
        (0,   7, 1),
        (7,  13, 1), # Inicio do dio
        (13, 19, 2),
        (19, 24, 1),
    ],
    "Farmacia": [
        (0,   6, 2),
        (6,  18, 2), # Dio todo        
        (18, 24, 2), # Troca a equipe mas mantem a quantidade até 7AM
    ],
    "Tec_Enfermagem": [
        ( 0,  7, 4), # Ao todo sao 7, mas sao demandados 3 ou 4        
        ( 7, 19, 6), # Inicio do dio
        (19, 24, 4),        
    ],
    "Tec_Radiologia": [
        (0,   7, 1),
        (7,  19, 2), # Dio todo        
        (19, 24, 1), # Troca a equipe mas mantem a quantidade até 7AM
    ],
    "Eq_Radiologia":[
        (0,   7, 1),
        (7,  19, 2), # Dio todo        
        (19, 24, 1), # Troca a equipe mas mantem a quantidade até 7AM
    ],
    "Eq_Higienizacao":[
        (0,   7, 1),
        (7,  19, 2), # Dio todo        
        (19, 24, 1), # Troca a equipe mas mantem a quantidade até 7AM
    ],
    # add other resources as needed ...
}

# ---------------------------------------------------------------
# Generic time-dependent arrival
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Time-varying arrival profiles — SEPARATE for weekdays and weekends
# ---------------------------------------------------------------
# WHY TWO PROFILES?
# ─────────────────
# On weekdays the Centro Cirúrgico operates ~06h–22h with a broad
# morning–afternoon peak.  On weekends only urgent/emergency cases
# are performed, concentrated in the morning (≈06h–14h) with a
# very steep fall-off afterward.
# Using a single 24-slot weekday profile with only a reduced 'wf'
# factor still spreads arrivals across the full 24-slot weekday
# window; the inversion walk then crosses day boundaries mid-slot
# and can pick up the wrong weekday factor for part of the weekend
# day — producing weekday-level rates on Saturday/Sunday.
# Separate profiles eliminate this entirely: the slot boundaries
# themselves are correct for each day type.

# Weekday profile (Monday–Friday): broad operating window
ARRIVAL_SLOTS_WEEKDAY = [
    ( 0,  2, 0.035),   # 00–02h:  3.5%
    ( 2,  4, 0.009),   # 02–04h:  0.9%
    ( 4,  6, 0.010),   # 04–06h:  1.0%
    ( 6,  8, 0.186),   # 06–08h: 18.6%
    ( 8, 10, 0.111),   # 08–10h: 11.1%
    (10, 12, 0.151),   # 10–12h: 15.1%
    (12, 14, 0.108),   # 12–14h: 10.8%
    (14, 16, 0.125),   # 14–16h: 12.5%
    (16, 18, 0.106),   # 16–18h: 10.6%
    (18, 20, 0.035),   # 18–20h:  3.5%
    (20, 22, 0.063),   # 20–22h:  6.3%
    (22, 24, 0.061),   # 22–00h:  6.1%
]

# Weekend profile (Saturday–Sunday): concentrated morning window,
# minimal activity outside 06h–16h.
# Fractions must also sum to 1.0 — they represent the SHAPE of
# the within-day distribution; the VOLUME is governed by wf × base.
ARRIVAL_SLOTS_WEEKEND = [
    ( 0,  6, 0.030),   # 00–06h:  3.0%  — overnight urgencies only
    ( 6,  8, 0.210),   # 06–08h: 21.0%  — morning ramp-up
    ( 8, 10, 0.280),   # 08–10h: 28.0%  — peak
    (10, 12, 0.220),   # 10–12h: 22.0%  — late morning
    (12, 14, 0.140),   # 12–14h: 14.0%  — afternoon taper
    (14, 16, 0.080),   # 14–16h:  8.0%  — low afternoon
    (16, 24, 0.040),   # 16–24h:  4.0%  — evening/night minimal
]
# Validate: 0.030+0.210+0.280+0.220+0.140+0.080+0.040 = 1.000 ✓

# Keep the old name as an alias so any external reference still works.
ARRIVAL_SLOTS = ARRIVAL_SLOTS_WEEKDAY

# ---------------------------------------------------------------
# Day-of-week volume factors  (INDEPENDENT of BASE_ARRIVALS_PER_DAY)
# ---------------------------------------------------------------
# These are PURE RELATIVE WEIGHTS derived from observed historical means.
# The denominator is the fixed historical global mean (≈14.38) and must
# NOT reference BASE_ARRIVALS_PER_DAY — otherwise changing the base for
# scenario planning would cancel out in the product BASE × wf and every
# scenario would produce identical volumes.
#
# How it works:
#   expected_arrivals_day_d = BASE_ARRIVALS_PER_DAY × wf[d]
#
# Changing BASE scales ALL days proportionally while the weekend
# penalty (wf≈0.55–0.69) is preserved in every scenario.
#
# Source data (13 weeks of observations):
#   Segunda=16.38  Terça=17.46  Quarta=16.17  Quinta=17.31
#   Sexta=15.54    Sábado=9.92  Domingo=7.85
#   Historical global mean = (16.38+17.46+16.17+17.31+15.54+9.92+7.85)/7 ≈ 14.38
_HIST_GLOBAL_MEAN = 14.375714285714286   # fixed — do NOT replace with BASE_ARRIVALS_PER_DAY

WEEKDAY_FACTORS = {
    0: 16.38 / _HIST_GLOBAL_MEAN,  # Segunda-feira  (Monday)    ≈ 1.139
    1: 17.46 / _HIST_GLOBAL_MEAN,  # Terça-feira    (Tuesday)   ≈ 1.214
    2: 16.17 / _HIST_GLOBAL_MEAN,  # Quarta-feira   (Wednesday) ≈ 1.125  ← start_day_of_week=2
    3: 17.31 / _HIST_GLOBAL_MEAN,  # Quinta-feira   (Thursday)  ≈ 1.204
    4: 15.54 / _HIST_GLOBAL_MEAN,  # Sexta-feira    (Friday)    ≈ 1.081
    5:  9.92 / _HIST_GLOBAL_MEAN,  # Sábado         (Saturday)  ≈ 0.690  ← weekend
    6:  7.85 / _HIST_GLOBAL_MEAN,  # Domingo        (Sunday)    ≈ 0.546  ← weekend
}
# Scenario examples with BASE_ARRIVALS_PER_DAY:
#   BASE=14.38 → Sat≈9.92,  Sun≈7.85  (historical baseline)
#   BASE=16    → Sat≈11.0,  Sun≈8.7   (growth scenario)
#   BASE=20    → Sat≈13.8,  Sun≈10.9  (stress scenario)
# The weekend:weekday ratio is ~0.60–0.69 in every scenario.

# ================================================================
HOURS = 60  # Time conversion factor (base time: Minutos)
DAYS = 1440
YEARS = 525600
# ================================================================


def make_nhpp_interarrival(
    env,
    arrival_slots,
    base_arrivals_per_day,
    weekday_factors,
    start_day_of_week=2,
    arrival_slots_weekend=None,
    weekend_days=(5, 6)):
    """
    Piecewise-constant Non-Homogeneous Poisson Process — CORRECT implementation
    with separate weekday / weekend arrival profiles.

    WHY THE PREVIOUS VERSION WAS WRONG
    ────────────────────────────────────
    The old code used a single ``arrival_slots`` profile for every day.
    The inversion walk steps forward slot-by-slot; when it crossed a midnight
    boundary it called ``_lambda_at`` with the new time and picked up the
    correct weekday factor (wf)—but ``_slot_end_absolute`` still used the
    WEEKDAY slot boundaries to decide where the next boundary was.  On a
    weekend day the slot widths from the weekday profile were used, so the
    walk could land inside a slot whose *boundary* placed ``t`` at a point
    where the day-index integer changed back to a weekday, inadvertently
    applying a weekday ``wf`` for part of the weekend day.  Over many days
    this inflated the simulated weekend volume toward weekday levels.

    A second issue: ``BASE_ARRIVALS_PER_DAY`` was set to 16 while
    ``WEEKDAY_FACTORS`` were calibrated with denominator 13.33 (a different
    reference mean), so the expected daily arrivals were consistently
    over-estimated for every day type.

    FIXES
    ──────
    1. ``arrival_slots_weekend`` — a separate profile for days in
       ``weekend_days`` (default: Python weekdays 5=Sat, 6=Sun).
       Both ``_lambda_at`` AND ``_slot_end_absolute`` select the right
       profile based on the actual weekday, ensuring the slot boundaries
       are always consistent with the rate computation.
    2. ``BASE_ARRIVALS_PER_DAY`` is now set to the true historical daily
       mean so that ``wf = historical_mean_d / global_mean`` gives:
           E[arrivals on day d] = BASE × wf[d] = historical_mean_d  ✓

    ALGORITHM — Piecewise Inversion Method
    ────────────────────────────────────────────────
    1. Draw E ~ Exponential(1): one "unit of expected arrivals to consume".
    2. Starting at the current simulation time t, walk forward slot by slot.
    3. For each slot, compute the expected arrivals remaining in that slot from
       position t: Δ = λ_slot × (slot_end − t).
    4. If E ≤ Δ  →  the next arrival lands inside this slot at t + E / λ_slot.
       If E > Δ  →  subtract Δ, advance t to the next slot boundary, repeat.
    5. Correctly inherits the day-of-week factor AND the correct slot profile
       as t crosses midnight.

    Parameters
    ──────────
    env                   : simpy.Environment
    arrival_slots         : list[(start_h, end_h, fraction)]  weekday profile
    base_arrivals_per_day : float  historical global daily mean (≈14.38)
    weekday_factors       : dict {weekday_int: float}  wf = day_mean / global_mean
    start_day_of_week     : int  Python weekday (0=Mon…6=Sun) of simulation t=0
    arrival_slots_weekend : list[(start_h, end_h, fraction)] or None
                            Weekend profile; if None falls back to arrival_slots.
    weekend_days          : tuple of weekday ints treated as weekend (default (5,6))
    """

    MINUTES_PER_DAY = 1440

    _slots_weekday = arrival_slots
    _slots_weekend = arrival_slots_weekend if arrival_slots_weekend is not None else arrival_slots

    # Safety validation — both profiles must partition [0, 24)
    for label, slots in (("weekday", _slots_weekday), ("weekend", _slots_weekend)):
        total = sum(f for _, _, f in slots)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Arrival slot fractions ({label}) must sum to 1.0, got {total:.6f}"
            )

    def _weekday_and_slots(absolute_minute):
        """Return (weekday_int, wf, slot_profile) for the given absolute minute."""
        day_index = int(absolute_minute // MINUTES_PER_DAY)
        weekday   = (start_day_of_week + day_index) % 7
        wf        = weekday_factors.get(weekday, 1.0)
        slots     = _slots_weekend if weekday in weekend_days else _slots_weekday
        return weekday, wf, slots

    def _lambda_at(absolute_minute):
        """Return the arrival rate (arrivals/min) at a given absolute minute.

        Selects the correct slot profile (weekday vs weekend) based on the
        computed calendar weekday, ensuring the weekend volume-factor (wf) is
        applied consistently for the FULL weekend day.
        """
        _, wf, slots = _weekday_and_slots(absolute_minute)
        hour = (absolute_minute % MINUTES_PER_DAY) / 60.0
        for start_h, end_h, fraction in slots:
            if start_h <= hour < end_h:
                slot_min = (end_h - start_h) * 60
                return max(base_arrivals_per_day * wf * fraction / slot_min, 1e-9)
        return 1e-9  # safety fallback

    def _slot_end_absolute(absolute_minute):
        """Return the absolute minute at which the current slot ends.

        Uses the SAME profile as _lambda_at (weekday or weekend) so that the
        slot boundaries seen by the inversion walk are always consistent with
        the rate that was applied, preventing day-index drift across midnight.
        """
        day_start = int(absolute_minute // MINUTES_PER_DAY) * MINUTES_PER_DAY
        _, _, slots = _weekday_and_slots(absolute_minute)
        hour = (absolute_minute % MINUTES_PER_DAY) / 60.0
        for start_h, end_h, _ in slots:
            if start_h <= hour < end_h:
                return day_start + end_h * 60
        # Fallback: advance to the start of the next day
        return day_start + MINUTES_PER_DAY

    def interarrival():
        # ── Step 1: draw one Exp(1) variate ───────────────────────────────────
        # This represents the "amount of expected arrivals" before the next event.
        e = random.expovariate(1.0)

        # ── Step 2: walk forward through slots consuming 'e' ──────────────────
        t = env.now
        for _guard in range(10000):          # guard against infinite loops
            lam      = _lambda_at(t)
            slot_end = _slot_end_absolute(t)
            remaining_in_slot = slot_end - t

            # Expected arrivals from t to the end of this slot
            delta = lam * remaining_in_slot

            if e <= delta:
                # Next arrival falls inside this slot
                dt = e / lam
                return t + dt - env.now      # return the interarrival duration

            # Arrival is beyond this slot — subtract and move to next slot
            e -= delta
            t  = slot_end                    # advance to next slot boundary

        # Safety fallback (should never be reached with reasonable inputs)
        return t - env.now

    return interarrival


def build_model(final_simulation_time=None, event_logger=None, verbose=True,
                        entity_filter=None, resource_filter=None,
                        event_type_filter=None, time_range=None): 
    """Build the simulation model with refactored structure.
    Args:
        event_logger: Optional event logger
        verbose: Enable event tracing
        entity_filter: Optional entity filter for tracing
        resource_filter: Optional resource filter for tracing
        event_type_filter: Optional event type filter for tracing
        time_range: Optional time range for tracing
    """    
    
    HOURS = 60  # Time conversion factor (base time: minutes)
    DAYS = 1440
    YEARS = 525600
    
    model = SimulationModel(verbose=verbose,
        entity_filter=entity_filter,
        resource_filter=resource_filter,
        event_type_filter=event_type_filter,
        time_range=time_range)  

    def truncated_lognormal(mu, sigma, min_los, max_los):       
        while True:
            x = random.lognormvariate(mu, sigma)
            if min_los <= x <= max_los:
                return x
                


    # Unidade básica para todos os tempos: minutos
    def distribution(tipo):
        # Arrival rates (per day - per minute if DAYS = 1440)
        # arrival_rate_cc  = 1/53 # per minute        

        distributions = {            
            # Arrivals            
            #'arrival_cc': random.expovariate(arrival_rate_cc),

            # 1. Aguarda cirurgia
            # '1_aguarda_CC': random.triangular(40, 160, 280),
                        
            # 1. Preparação da sala. Aviso cirúrgico (P:1,6)
            '1_aviso_cirurgico': 5,

            # 1. Montagem dos kits (materiais e medicação) (P:2)
            '1_montagem_kits': 5,

            # 1. Rec kit, Sep. Instrumentos, Check list, Org., Registro Info (P:3,4,5,8,9)
            # '1_organiza_sala_e_kits': random.triangular(20, 25, 30),
            '1_organiza_sala_e_kits': random.triangular(8, 10, 15),

            # 2. Transporte do paciente do CTI ao Centro Cirúrgico (P:11)
            # '2_transporte_CTI_CC': random.triangular(20, 80, 150),
            '2_transporte_CTI_CC': random.triangular(20, 40, 60),

            # 2. Transporte do paciente de sua origem ao Centro Cirúrgico (P:11)
            '2_transporte_origem_CC': random.triangular(10, 15, 20),

            # 2. Admissão e Conferência do Paciente (P:12,13,14,15)
            #####################CONFERIR#######################            
            # 2_adm_conf.txt filtrado (retirada outliers: > 5 min, < 75 min)
            # '2_adm_e_conf_paciente': random.triangular(15, 25, 30),
            # '2_adm_e_conf_paciente': 1/3 * (5.98926 + 7696.99 * random.betavariate(1.18591, 616.092)),
            '2_adm_e_conf_paciente': 1/3 * (4.9 + 100 * random.betavariate(1.18591, 616.092)),

            # 2. Admissão do Paciente (P:16,17)
            # '2_adm_paciente': random.triangular(10, 15, 20),
            # '2_adm_paciente': 1/3 * (5.98926 + 7696.99 * random.betavariate(1.18591, 616.092)),
            '2_adm_paciente': 1/3 * (4.9 + 100 * random.betavariate(1.18591, 616.092)),

            # 2. Checklist de Cirurgia Segura Físico (P:18)
            # '3_checklist_pre_cir': 5,
            # '3_checklist_pre_cir': 1/3 * (5.98926 + 7696.99 * random.betavariate(1.18591, 616.092)),
            '3_checklist_pre_cir': 1/3 * (4.9 + 100 * random.betavariate(1.18591, 616.092)),

            # 3. Ato Anestésico (P:19)

            # '3_ato_anestetico': random.triangular(20, 50, 90),                       
            # 3_ato_anestetico.txt filtrado (retirada outliers: > 5 min, < 90 min)
             '3_ato_anestetico':5.86479 + 2023.26 * random.betavariate(1.72019, 161.458),   

            # 3. Pr.Cirúrgico Pequeno (P:20,21,22)
            # '3_cirurgia_pequena': random.triangular(20, 70, 120),
            '3_cirurgia_pequena': random.lognormvariate(3.7377, 0.879545),

            # 3. Pr.Cirúrgico Médio (P:20,21,22)
            # '3_cirurgia_media': random.triangular(120, 180, 240),
            '3_cirurgia_media': random.gammavariate(2.2796, 45.7462),

            # 3. Pr.Cirúrgico Grande (P:20,21,22)
            # '3_cirurgia_grande': random.triangular(240, 480, 720),            
            '3_cirurgia_grande': random.lognormvariate(4.19377, 0.862757),

            # 3. Conferências e registros (P:23,24,25,26,27,28,29,30)
            # '3_conf_registros_pos_cir': random.triangular(20, 30, 40),            
            # como são equivalentes a 3 atividades, tem-se 0.7* (...), o shape nao muda.
            # 2_pos_cir.txt filtrado (retirada outliers: > 5 min, < 90 min)            
            '3_conf_registros_pos_cir': 0.7 * random.weibullvariate(16.4986, 1.4622),

            # 5. Remoção de resíduos e descarte de materiais. (P:37)
            # '5_remocao_residuos': random.triangular(5, 6, 7),
            '5_remocao_residuos': 0.15 * random.weibullvariate(16.4986, 1.4622),

            # 5. Separação de materiais sujos p/ processamento. (P:38)
            # '5_sep_mat_sujos': random.triangular(5, 8, 10),
            '5_sep_mat_sujos': 0.15 * random.weibullvariate(16.4986, 1.4622),

            # 5. Limpeza do Local. (P:39)
            '5_limpeza_local': random.triangular(10, 12, 15),

            # 6. Processamento de materiais. (P:40,41,42,43,44)
            #'6_processamento_materiais': random.triangular(20, 28, 34),

            # From this part on.. it is not being used.....
            # 4. Pós-oper Encaminha paciente para SRPA (P:31)
            '4_encaminha_srpa': random.triangular(10, 20, 30),

            # 4. Monit, Med, Permanência e av. Alta SRPA  (P:32,33,34)
            '4_monit_permanencia_alta_srpa': random.triangular(480, 1680, 2880),
            # '4_monit_permanencia_alta_srpa': random.triangular(240, 840, 1440),            

            # 4. Avaliação para Alta da SRPA (P:35)
            '4_avalia_alta_srpa': random.triangular(5, 8, 10),

            # 4. Transferência para Unidade de Internação (P:36)
            '4_transfere_internacao': random.triangular(10, 15, 20)
        }
        return distributions.get(tipo, 0.0)
    
    

    Enfermeiro = model.add_resource("Enfermeiro", 1, "regular") 
    Farmacia = model.add_resource("Farmacia", 2, "regular") 
    Tec_Enfermagem = model.add_resource("Tec_Enfermagem", 6, "regular") 
    Eq_Assistencial_CTI = model.add_resource("Eq_Assistencial_CTI", 1, "regular") 
    Eq_Medica = model.add_resource("Eq_Medica", 5, "regular")     
    Anestesista = model.add_resource("Anestesista", 5, "regular")    
    Tec_Radiologia = model.add_resource("Tec_Radiologia", 2, "regular") 
    Eq_Radiologia = model.add_resource("Eq_Radiologia", 2, "regular") 
    Func_CME = model.add_resource("Func_CME", 1, "regular") 
    Eq_Higienizacao = model.add_resource("Eq_Higienizacao", 2, "regular")
    # Tec_Enfermagem_X = model.add_resource("Tec_Enfermagem_X", 1, "regular")  
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BACKGROUND WORKLOAD — unmodelled staff activities
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Design rationale
    # ─────────────────
    # The model's scope is the surgical pathway only.  However, SimPy computes
    # resource utilization as:
    #
    #   utilization = total_time_resource_is_busy / total_simulation_time
    #
    # When background activities (paperwork, training, stocking …) are absent
    # from the model, utilization numbers reflect ONLY the surgical pathway,
    # giving misleadingly low values (e.g. 2-8%) even though staff are busy for
    # most of their shift.
    #
    # This generator models those activities as a continuous background process:
    # each resource unit repeatedly performs tasks of random duration separated
    # by random gaps.  Parameters come from BACKGROUND_WORKLOAD (see global
    # config above); disable with ENABLE_BACKGROUND_WORKLOAD = False.
    #
    # Metrics impact
    # ──────────────
    # • Resource "Time Busy" increases → utilization rises to realistic levels.
    # • Surgical-pathway queue times are also affected (background tasks compete
    #   for resource slots).  If background load is high, a few surgical events
    #   may queue slightly — which is also realistic and desirable.
    #
    # Reporting note
    # ──────────────
    # Report results as:
    #   "Utilization (surgical pathway only): X%"
    #   "Utilization (total incl. background activities): Y%"
    #
    # ═══════════════════════════════════════════════════════════════════════════

    _resource_objects = {
        "Enfermeiro":         Enfermeiro,
        "Farmacia":           Farmacia,
        "Tec_Enfermagem":     Tec_Enfermagem,
        "Eq_Assistencial_CTI":Eq_Assistencial_CTI,
        "Eq_Medica":          Eq_Medica,
        "Anestesista":        Anestesista,
        "Tec_Radiologia":     Tec_Radiologia,
        "Eq_Radiologia":      Eq_Radiologia,
        "Func_CME":           Func_CME,
        "Eq_Higienizacao":    Eq_Higienizacao,
    }

    def _bg_load_factor(res_name, current_minute):
        """
        Return the background load_factor for res_name at current_minute.

        Reads BACKGROUND_SCHEDULE; falls back to 1.0 if the resource has no
        schedule or the current hour falls outside every defined slot.
        A load_factor of 0.0 is clamped to a tiny epsilon so the worker never
        hangs forever waiting on a zero-rate timeout.
        """
        slots = BACKGROUND_SCHEDULE.get(res_name)
        if not slots:
            return 1.0
        hour = (current_minute % 1440) / 60.0
        for start_h, end_h, factor in slots:
            if start_h <= hour < end_h:
                return max(factor, 1e-6)
        return 1.0   # fallback: last slot or unscheduled hour

    def background_worker(env, simpy_resource, res_name, unit_idx,
                          task_mean, gap_mean, evt_logger):
        """
        Simulates one unit of a resource performing unmodelled background tasks
        AND writes start/complete rows to the EventLogger so that downstream
        analysis scripts (cc_event_log_analysis.py, resource_2h_slots.py) can
        account for this workload when computing resource utilization.

        Time-of-day awareness
        ─────────────────────
        Before each inter-task gap the worker reads BACKGROUND_SCHEDULE to get
        the current load_factor f ∈ (0, 1].  The effective gap becomes:

            gap_effective = Exp(gap_base / f)

        At f=1.0  → normal rate, gap mean = gap_base  (full daytime load)
        At f=0.2  → gap mean = gap_base / 0.2 = 5× longer  (evening)
        At f=0.05 → gap mean = gap_base / 0.05 = 20× longer (near idle at night)

        This makes the number of tasks — and therefore resource-busy minutes —
        track the operational intensity of each shift without adding extra
        SimPy processes or re-launching workers.

        CSV row format produced (one pair per task):
            case_id   : BG_{res_name}_u{unit_idx}_t{task_counter}  (unique per task)
            activity  : BG_{res_name}
            lifecycle : "start" / "complete"
            resource  : {res_name}   (must match DEFAULT_CAPACITIES keys)
            timestamp : env.now at seize / env.now after task
        """
        task_counter = 0
        activity_name = f"BG_{res_name}"

        while True:
            # ── 1. Time-aware inter-task idle gap ─────────────────────────────
            # Read schedule at the START of the gap; if the worker sleeps across
            # a shift boundary the gap will simply be slightly inaccurate — this
            # is acceptable for a background approximation.
            f = _bg_load_factor(res_name, env.now)
            effective_gap_mean = gap_mean / f          # larger gap → fewer tasks at night
            gap = random.expovariate(1.0 / effective_gap_mean)
            yield env.timeout(gap)

            # ── 2. Seize one unit of the resource ─────────────────────────────
            task_counter += 1
            case_id = f"BG_{res_name}_u{unit_idx}_t{task_counter}"

            with simpy_resource.request() as req:
                yield req  # blocks until a unit is free

                # ── 3. Log task START ──────────────────────────────────────────
                if evt_logger is not None:
                    evt_logger.log_event(
                        case_id=case_id,
                        activity=activity_name,
                        timestamp=env.now,
                        lifecycle="start",
                        resource=res_name,
                    )

                # ── 4. Hold the resource for the task duration ─────────────────
                # Task duration is NOT scaled — once started, a task takes as
                # long as it takes regardless of the hour.
                task_duration = random.expovariate(1.0 / task_mean)
                yield env.timeout(task_duration)

                # ── 5. Log task COMPLETE ───────────────────────────────────────
                if evt_logger is not None:
                    evt_logger.log_event(
                        case_id=case_id,
                        activity=activity_name,
                        timestamp=env.now,
                        lifecycle="complete",
                        resource=res_name,
                    )
            # Resource released automatically at end of 'with' block

    if ENABLE_BACKGROUND_WORKLOAD:
        for res_name, params in BACKGROUND_WORKLOAD.items():
            res_obj = _resource_objects.get(res_name)
            if res_obj is None:
                continue
            # Launch one independent background worker per resource UNIT so
            # that the total background load scales with staffing levels and
            # workers can run truly in parallel (each holds at most 1 unit).
            capacity = DEFAULT_CAPACITIES.get(res_name, 1)
            for _unit in range(capacity):
                model.env.process(
                    background_worker(
                        env=model.env,
                        simpy_resource=res_obj,
                        res_name=res_name,
                        unit_idx=_unit,
                        task_mean=params["task"],
                        gap_mean=params["gap"],
                        evt_logger=event_logger,   # ← passed through to EventLogger
                    )
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # OPERATING ROOM CAPACITY GATE  —  5 concurrent patients maximum
    # ═══════════════════════════════════════════════════════════════════════════
    #
    #   P12a15 starts  -  Sala_CC seized  -  P12a15 ends  -  Sala_CC RELEASED
    #   ... (all intermediate blocks: adm, surgery, cleanup)  ← UNCONSTRAINED
    #   P39 starts     -  Sala_CC seized  -  P39 ends     -  Sala_CC RELEASED
    #
    # Between P12a15 and P39 the slot was free, allowing unlimited concurrency.
    #
    # WHY simpy.Container IS THE CORRECT TOOL
    # ─────────────────────────────────────────
    # simpy.Resource  -  request token must be released by the SAME SimPy
    #                    process that acquired it (held via 'with req:').
    #                    Each DESK block is its own process, so the token
    #                    cannot survive a block transition.
    #
    # simpy.Container -  Container.get(1)  permanently decrements the count
    #                    Container.put(1)  permanently increments the count
    #                    get/put are STATELESS one-shot events: they can occur
    #                    in completely DIFFERENT SimPy processes, spanning the
    #                    entire P12a15 - P39 chain.
    #
    # FLOW CORRECT
    # ────────────────────
    #   ... - P11a/b - [seize_sala_cc] ─(blocks if 5 rooms busy)─►
    #           - P12a15 - adm - surgery - cleanup - P39
    #           - [release_sala_cc] - discharge
    #
    # sala_cc.level (0–5) = available rooms; in_use = 5 − level  (max 5)
    # ═══════════════════════════════════════════════════════════════════════════
    sala_CC = simpy.Container(model.env, capacity=5, init=5)

    # WIP log: each entry is (absolute_time, rooms_occupied).
    # Populated exclusively by SalaCC_Seize (on get) and SalaCC_Release
    # (on put), so it always reflects the true number of patients inside
    # the OR -- bounded by sala_CC.capacity (5).
    # plot_wip_over_time() must NOT be used for this metric: it counts all
    # entities from creation to disposal (pre-OR queue + OR + post-OR
    # discharge) and therefore routinely exceeds 5.
    model.wip_cc_log = []   # list[tuple[float, int]]

    # ── Custom gateway blocks ──────────────────────────────────────────────────
    # These two thin classes are the ONLY place Sala_CC logic lives.
    # They do not interact with model.add_resource — they operate on the raw
    # simpy.Container directly, which is the correct SimPy pattern for
    # cross-block resource spanning.
    #
    # Implementation note
    # ───────────────────
    # The classes subclass ProcessBlock so they integrate naturally with DESK's
    # model.add_block(), connect_to(), and event_logger machinery.
    # They override only the entity-processing coroutine (run / _run / process —
    # whichever name your DESK build uses; see the comment inside each class).
    # ─────────────────────────────────────────────────────────────────────────

    class SalaCC_Seize(ProcessBlock):
        """
        Gateway block placed between P11a/b and adm_conf_paciente_P12a15.

        Behaviour
        ─────────
        • Calls sala_cc.get(1).  If the container level is 0 (all 5 rooms
          occupied), the entity waits here until one room becomes free.
        • Once sala_cc.get(1) succeeds, the level is decremented by 1 and
          stays decremented until the matching SalaCC_Release fires put(1).
        • Zero processing delay — the block is purely a gate, not a service.

        The slot is held for the full duration of the patient's stay
        (P12a15 through P39), not just while this block is active.
        """
        def __init__(self, name, env, container, event_logger=None):
            super().__init__(name, env, delay_time=lambda: 0.0, resource=None, event_logger=event_logger)
            self._sala_cc = container

        def process_entity(self, entity: Entity):
            entity.route_history.append(self.name)
            
            # Trace queue entry in console
            self._trace('queue', entity, "Sala_CC", f"Waiting for operating room. Available: {self._sala_cc.level}")
            
            # 1. BLOCKS the process pipeline here until a room slot becomes free
            yield self._sala_cc.get(1)

            # Record OR WIP: rooms now occupied (1..5).
            # This is the ONLY correct place to measure surgical-centre WIP;
            # it counts patients physically inside the OR, not in queue or recovery.
            model.wip_cc_log.append((self.env.now, sala_CC.capacity - sala_CC.level))

            # print(
            #     f"[{self.env.now:.2f}] "
            #     f"ACQUIRED {entity.id} | "
            #     f"Available={self._sala_cc.level} "
            #     f"Occupied={5 - self._sala_cc.level}"
            # )
            
            # Trace successful allocation
            free_rooms = sala_CC.level
            occupied = sala_CC.capacity - free_rooms
            # self._trace('service_start', entity, "Sala_CC", f"Operating room secured. Remaining: {self._sala_cc.level}")
            self._trace('service_start', entity, "Sala_CC", f"Entered CC. Occupied rooms: " f"{occupied}/{self._sala_cc.capacity}")

            if self.event_logger:
                self.event_logger.log_event(
                    case_id=entity.id,
                    activity=self.name,
                    timestamp=self.env.now,
                    lifecycle='complete',
                    resource="Sala_CC"
                )
            
            # 2. Forward entity to the next block (adm_conf_paciente_P12a15)
            self.env.process(self.send_to_next(entity))            
            # self.send_to_next(entity)
            yield self.env.timeout(0)

    class SalaCC_Release(ProcessBlock):
        """
        Release block placed between limpeza_organizacao_P39 and discharge_srpa.

        Behaviour
        ─────────
        • Calls sala_cc.put(1), which increments the container level back.
        • put() never blocks (capacity is always ≥ in-use count).
        • This unblocks the next patient waiting in SalaCC_Seize, if any.
        • Zero processing delay — purely a signal, not a service step.
        """
        def __init__(self, name, env, container, event_logger=None):
            super().__init__(name, env, delay_time=lambda: 0.0, resource=None, event_logger=event_logger)
            self._sala_cc = container

        def process_entity(self, entity: Entity):
            entity.route_history.append(self.name)
            
            # 1. RETURN the operating room slot back to the pool
            yield self._sala_cc.put(1)

            # Record OR WIP after release: level went up by 1, occupied went down by 1.
            model.wip_cc_log.append((self.env.now, sala_CC.capacity - sala_CC.level))

            # print(
            #     f"[{self.env.now:.2f}] "
            #     f"RELEASED {entity.id} | "
            #     f"Available={self._sala_cc.level} "
            #     f"Occupied={5 - self._sala_cc.level}"
            # )
            
            # Trace successful release
            # self._trace('service_end', entity, "Sala_CC", f"Operating room released. Available: {self._sala_cc.level}")
            # Trace successful allocation
            free_rooms = sala_CC.level
            occupied = sala_CC.capacity - free_rooms
            self._trace('service_start', entity, "Sala_CC", f"Released CC. Occupied rooms: " f"{occupied}/{self._sala_cc.capacity}")

            
            if self.event_logger:
                self.event_logger.log_event(
                    case_id=entity.id,
                    activity=self.name,
                    timestamp=self.env.now,
                    lifecycle='complete',
                    resource="Sala_CC"
                )
            
            # 2. Forward entity to the next block (e.g., discharge or recovery area)
            self.env.process(self.send_to_next(entity))            
            # self.send_to_next(entity)
            yield self.env.timeout(0)

    # ── Instantiate the two gateway blocks ────────────────────────────────────
    seize_sala_cc = SalaCC_Seize(
        "Seize_Sala_CC", model.env,
        container=sala_CC,
        event_logger=event_logger,
    )

    release_sala_cc = SalaCC_Release(
        "Release_Sala_CC", model.env,
        container=sala_CC,
        event_logger=event_logger,
    )



    def make_resource_scheduler(env, resource_map, schedule):
        """
        Returns a SimPy generator that adjusts resource capacities at
        each slot boundary.  Run it as a background process:

            model.env.process(make_resource_scheduler(
                env=model.env,
                resource_map={
                    "Eq_Medica":      Eq_Medica,
                    "Enfermeiro":     Enfermeiro,
                    "Tec_Enfermagem": Tec_Enfermagem,
                },
                schedule=RESOURCE_SCHEDULE,
            ))

        Notes
        -----
        • Capacity *increase*: pending requests are served immediately.
        • Capacity *decrease*: in-service entities are NOT preempted;
        the lower cap takes effect as servers become free (standard
        SimPy behaviour — no extra logic needed).
        • resource._capacity is SimPy's internal attribute; there is no
        public setter in SimPy 4.x, so direct assignment is the
        accepted pattern for dynamic staffing.

        Parameters
        ----------
        env          : simpy.Environment  (model.env)
        resource_map : dict {name: simpy_resource}
        schedule     : dict {name: [(start_h, end_h, capacity), ...]}
        days         : int   minutes per simulated day (default 1440)
        """

        DAYS=1440

        def _current_capacity(slots, current_hour):
            """Return the capacity for the active slot."""
            for start_h, end_h, cap in slots:
                if start_h <= current_hour < end_h:
                    return cap
            return slots[-1][2]                         # fallback: last slot

        def _minutes_to_next_boundary(now, schedule):
            """Minutes until the nearest upcoming slot boundary."""
            day_minute = now % DAYS
            # Collect every unique boundary (in minutes) across all resources
            boundaries = sorted({
                h * 60
                for slots in schedule.values()
                for start_h, end_h, _ in slots
                for h in (start_h, end_h)
            })
            for b in boundaries:
                if b > day_minute:
                    return b - day_minute
            # Wrap around midnight to the first boundary of the next day
            return (DAYS - day_minute) + boundaries[0]

        def _scheduler():
            while True:
                current_hour = (env.now % DAYS) / 60.0

                # ── Apply capacity for every scheduled resource ────────────────
                for name, slots in schedule.items():
                    if name in resource_map:
                        new_cap = _current_capacity(slots, current_hour)
                        resource_map[name]._capacity = new_cap

                # ── Sleep until the next boundary ──────────────────────────────
                wait = _minutes_to_next_boundary(env.now, schedule)
                yield env.timeout(wait)

        return _scheduler()

    # ── Start the staffing scheduler ───────────────────────────────────────
    model.env.process(make_resource_scheduler(
        env=model.env,
        resource_map={
            "Eq_Medica":      Eq_Medica,
            "Enfermeiro":     Enfermeiro,
            "Tec_Enfermagem": Tec_Enfermagem,
        },
        schedule=RESOURCE_SCHEDULE,
    ))


    # 1. Criação do Bloco usando a função de chegada calibrada por hora/dia
   
    # ============================ ACTIVITIES ====================
    # Create block
    arrivals_cc = CreateBlock(
        "Cheg_CC", model.env,        
        inter_arrival_time=make_nhpp_interarrival(
        env=model.env,
        arrival_slots=ARRIVAL_SLOTS,
        base_arrivals_per_day=BASE_ARRIVALS_PER_DAY,
        weekday_factors=WEEKDAY_FACTORS,
        start_day_of_week=2
        ),
        entity_prefix="CC_Patient",
        max_arrivals=None, # Infinito
        first_creation=0.0,
        # priority_generator=patient_severity,
        event_logger=event_logger
    )
    
    # # More reliable attribute injection
    # def inject_surgery_complexity(entity):
    #     entity.surgery_complexity = generate_surgery_complexity()
    #     # Also store in attributes dict (some frameworks use this)
    #     if not hasattr(entity, 'attributes'):
    #         entity.attributes = {}
    #     entity.attributes['surgery_complexity'] = entity.surgery_complexity
     
    # arrivals_cc.assign_attributes_callback = inject_surgery_complexity
    # # Keep the original too as backup
    # arrivals_cc.assign_attributes(surgery_complexity=generate_surgery_complexity)  

    # # Definindo a função que gera o atributo dinamicamente baseado na hora do relógio da simulação
    def generate_surgery_size():
        current_hour = (model.env.now % 1440) / 60.0
        u = random.random()
        # -------------------------------------------------
        # DAY SHIFT - elective/planned surgeries dominate
        # -------------------------------------------------
        if 7 <= current_hour < 19:
            if u < 0.30:    return "Pequena"
            elif u < 0.62:  return "Media"
            else:           return "Grande"
        # -------------------------------------------------
        # NIGHT SHIFT - emergency/fast procedures dominate
        # -------------------------------------------------
        else:
            if u < 0.20:    return "Pequena"
            elif u < 0.40:  return "Media"
            else:           return "Grande" # 60% das cirurgias sao de 2h (grande-porte)


    # def inject_surgery_size(entity):
    #     surgery_size = generate_surgery_size()
    #     # Direct attribute
    #     entity.surgery_size = surgery_size
    #     # Optional DESK compatibility
    #     if not hasattr(entity, 'attributes'):
    #         entity.attributes = {}
    #     entity.attributes['surgery_size'] = surgery_size

    # def get_surgery_size(e):
    #     if hasattr(e, 'surgery_size'):
    #         return e.surgery_size
    #     if hasattr(e, 'attributes'):
    #         return e.attributes.get('surgery_size', 'Pequena')
    #     return 'Pequena'
    

    # def is_pequeno(e, ctx):
    #     return get_surgery_size(e) == "Pequena"

    # def is_medio(e, ctx):
    #     return get_surgery_size(e) == "Media"

    # def is_grande(e, ctx):
    #     return get_surgery_size(e) == "Grande"

    def get_surgery_size(e):
        return e.get_attribute("surgery_size", "MISSING")

    def is_pequeno(e, ctx):
        s = get_surgery_size(e)
        # print("SIZE=", s)
        return s == "Pequena"

    def is_medio(e, ctx):
        s = get_surgery_size(e)
        return s == "Media"

    def is_grande(e, ctx):
        s = get_surgery_size(e)
        return s == "Grande"

    # Register ONLY callback
    # arrivals_cc.assign_attributes_callback = inject_surgery_size
    arrivals_cc.assign_attributes(surgery_size=generate_surgery_size)
    

     
    


    # # # ProcessBlock block: Process with NO resource
    # delay_ag_cc = ProcessBlock(
    #     "Pac_aguarda_lib_CC", model.env,
    #     resource=Tec_Enfermagem_X,        
    #     delay_time=lambda: distribution('1_aguarda_CC'),        
    #     event_logger=event_logger
    # )
    # delay_ag_cc.set_resource_name('Tec_Enfermagem_X') 
    
    # ProcessBlock block: Process with ONE resource
    prep_sala_P16 = ProcessBlock(
        "Aviso_Cir", model.env,
        resource=Enfermeiro,        
        delay_time=lambda: distribution('1_aviso_cirurgico'),
        resource_units=1,                 
        event_logger=event_logger
    )
    prep_sala_P16.set_resource_name('Enfermeiro')

    # ProcessBlock block: Process with ONE resource
    prep_sala_P2 = ProcessBlock(
        "Monta_Kits", model.env,
        resource=Farmacia,        
        delay_time=lambda: distribution('1_montagem_kits'),
        resource_units=1,                 
        event_logger=event_logger
    )
    prep_sala_P2.set_resource_name('Farmacia') 

    # ProcessBlock block: Process with ONE resource
    prep_sala_P3a9 = ProcessBlock(
        "Org_Sala_Kits", model.env,
        resource=Tec_Enfermagem,        
        delay_time=lambda: distribution('1_organiza_sala_e_kits'),
        resource_units=1,                 
        event_logger=event_logger
    )
    prep_sala_P3a9.set_resource_name('Tec_Enfermagem') 

    # ProcessBlock block: Process with ONE resource
    adm_conf_paciente_P11a = ProcessBlock(
        "Transp_CTI_CC", model.env,
        resource=Eq_Assistencial_CTI,        
        delay_time=lambda: distribution('2_transporte_CTI_CC'),
        resource_units=1,                 
        event_logger=event_logger
    )
    adm_conf_paciente_P11a.set_resource_name('Eq_Assistencial_CTI')

    # ProcessBlock block: Process with ONE resource
    adm_conf_paciente_P11b = ProcessBlock(
        "Transp_Ori_CC", model.env,
        resource=Tec_Enfermagem,        
        delay_time=lambda: distribution('2_transporte_origem_CC'),
        resource_units=1,                 
        event_logger=event_logger
    )
    adm_conf_paciente_P11b.set_resource_name('Tec_Enfermagem')

    # ProcessBlock block: Process with ONE resource    
    adm_conf_paciente_P12a15 = ProcessBlock(
        "Adm_Conf_Pac", model.env,
        resource=Tec_Enfermagem,        
        delay_time=lambda: distribution('2_adm_e_conf_paciente'),
        resource_units=1,          
        event_logger=event_logger
    )
    adm_conf_paciente_P12a15.set_resource_name('Tec_Enfermagem')


    # ProcessBlock block: Process with ONE resource
    adm_paciente_P1617 = ProcessBlock(
        "Adm_Paciente", model.env,
        resource=Tec_Enfermagem,        
        delay_time=lambda: distribution('2_adm_paciente'),
        resource_units=1,                 
        event_logger=event_logger
    )
    adm_paciente_P1617.set_resource_name('Tec_Enfermagem')

    # ProcessBlock block: Process with ONE resource    
    # proc_cirurgico_P18 = ProcessBlock(
    proc_cirurgico_P18 = MultiProcessBlock(
        "Check_Cir_Seg", model.env,
        # resource=Eq_Medica,        
        resource_requirements={                        
            Eq_Medica: 1,            
            Tec_Enfermagem: 1,
            Anestesista: 1
        },
        delay_time=lambda: distribution('3_checklist_pre_cir'),
        # resource_units=1,                 
        event_logger=event_logger
    )
    # proc_cirurgico_P18.set_resource_name('Eq_Medica')    
    proc_cirurgico_P18.set_resource_names({                
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem',
        Anestesista: 'Anestesista'
    })  

    # ProcessBlock block: Process with ONE resource
    proc_cirurgico_P19 = ProcessBlock(
        "Ato_Anest", model.env,
        resource=Anestesista,        
        delay_time=lambda: distribution('3_ato_anestetico'),
        resource_units=1,                 
        event_logger=event_logger
    )
    proc_cirurgico_P19.set_resource_name('Anestesista')  

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 25% of distribution('3_cirurgia_pequena')
    proc_cirurgico_P_P20_025 = MultiProcessBlock(
        "Cir_Pequena_025", model.env,        
        resource_requirements={                        
            Eq_Medica: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.25*distribution('3_cirurgia_pequena'),        
        event_logger=event_logger
    )    
    proc_cirurgico_P_P20_025.set_resource_names({                
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })  

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 15% of distribution('3_cirurgia_pequena')
    proc_cirurgico_P_P20aP22_015 = MultiProcessBlock(
        "Cir_Pequena_015", model.env,        
        resource_requirements={            
            Anestesista: 1,
            Eq_Medica: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.15*distribution('3_cirurgia_pequena'),        
        event_logger=event_logger
    )    
    proc_cirurgico_P_P20aP22_015.set_resource_names({        
        Anestesista: 'Anestesista',
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })  

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 15% of distribution('3_cirurgia_pequena')
    proc_cirurgico_P_P20aP22_015_Radio = MultiProcessBlock(
        "Cir_Pequena_015R", model.env,        
        resource_requirements={            
            Anestesista: 1,
            Eq_Medica: 1,
            Tec_Radiologia: 1,
            Eq_Radiologia: 1,
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.15*distribution('3_cirurgia_pequena'),        
        event_logger=event_logger
    )    
    proc_cirurgico_P_P20aP22_015_Radio.set_resource_names({        
        Anestesista: 'Anestesista',
        Eq_Medica: 'Eq_Medica',
        Tec_Radiologia: 'Tec_Radiologia',
        Eq_Radiologia: 'Eq_Radiologia',
        Tec_Enfermagem: 'Tec_Enfermagem'
    }) 

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 60% of distribution('3_cirurgia_pequena')
    proc_cirurgico_P_P20_060 = MultiProcessBlock(
        "Cir_Pequena_060", model.env,        
        resource_requirements={                        
            Eq_Medica: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.60*distribution('3_cirurgia_pequena'),        
        event_logger=event_logger
    )    
    proc_cirurgico_P_P20_060.set_resource_names({                
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 25% of distribution('3_cirurgia_media')
    proc_cirurgico_M_P20_025 = MultiProcessBlock(
        "Cir_Media_025", model.env,        
        resource_requirements={                        
            Eq_Medica: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.25*distribution('3_cirurgia_media'),        
        event_logger=event_logger
    )    
    proc_cirurgico_M_P20_025.set_resource_names({        
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 15% of distribution('3_cirurgia_media')
    proc_cirurgico_M_P20aP22_015 = MultiProcessBlock(
        "Cir_Media_015", model.env,        
        resource_requirements={            
            Anestesista: 1,
            Eq_Medica: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.15*distribution('3_cirurgia_media'),        
        event_logger=event_logger
    )    
    proc_cirurgico_M_P20aP22_015.set_resource_names({        
        Anestesista: 'Anestesista',
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 15% of distribution('3_cirurgia_media')
    proc_cirurgico_M_P20aP22_015_Radio = MultiProcessBlock(
        "Cir_Media_015R", model.env,        
        resource_requirements={            
            Anestesista: 1,
            Eq_Medica: 1,
            Tec_Radiologia: 1,
            Eq_Radiologia: 1,
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.15*distribution('3_cirurgia_media'),        
        event_logger=event_logger
    )    
    proc_cirurgico_M_P20aP22_015_Radio.set_resource_names({        
        Anestesista: 'Anestesista',
        Eq_Medica: 'Eq_Medica',
        Tec_Radiologia: 'Tec_Radiologia',
        Eq_Radiologia: 'Eq_Radiologia',
        Tec_Enfermagem: 'Tec_Enfermagem'
    })

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 60% of distribution('3_cirurgia_media')
    proc_cirurgico_M_P20_060 = MultiProcessBlock(
        "Cir_Media_060", model.env,        
        resource_requirements={                        
            Eq_Medica: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.60*distribution('3_cirurgia_media'),        
        event_logger=event_logger
    )    
    proc_cirurgico_M_P20_060.set_resource_names({        
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 25% of distribution('3_cirurgia_grande')
    proc_cirurgico_G_P20_025 = MultiProcessBlock(
        "Cir_Grande_025", model.env,        
        resource_requirements={                        
            Eq_Medica: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.25*distribution('3_cirurgia_grande'),        
        event_logger=event_logger
    )    
    proc_cirurgico_G_P20_025.set_resource_names({                
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 15% of distribution('3_cirurgia_grande')
    proc_cirurgico_G_P20aP22_015 = MultiProcessBlock(
        "Cir_Grande_015", model.env,        
        resource_requirements={            
            Anestesista: 1,
            Eq_Medica: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.15*distribution('3_cirurgia_grande'),        
        event_logger=event_logger
    )    
    proc_cirurgico_G_P20aP22_015.set_resource_names({        
        Anestesista: 'Anestesista',
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 15% of distribution('3_cirurgia_grande')
    proc_cirurgico_G_P20aP22_015_Radio = MultiProcessBlock(
        "Cir_Grande_015R", model.env,        
        resource_requirements={            
            Anestesista: 1,
            Eq_Medica: 1,
            Tec_Radiologia: 1,
            Eq_Radiologia: 1,
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.15*distribution('3_cirurgia_grande'),        
        event_logger=event_logger
    )    
    proc_cirurgico_G_P20aP22_015_Radio.set_resource_names({        
        Anestesista: 'Anestesista',
        Eq_Medica: 'Eq_Medica',
        Tec_Radiologia: 'Tec_Radiologia',
        Eq_Radiologia: 'Eq_Radiologia',
        Tec_Enfermagem: 'Tec_Enfermagem'
    })

    # MultiProcessBlock block: Process with MULTIPLE resources
    # >> Atention: 60% of distribution('3_cirurgia_grande')
    proc_cirurgico_G_P20_060 = MultiProcessBlock(
        "Cir_Grande_060", model.env,        
        resource_requirements={                        
            Eq_Medica: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: 0.60*distribution('3_cirurgia_grande'),        
        event_logger=event_logger
    )    
    proc_cirurgico_G_P20_060.set_resource_names({                
        Eq_Medica: 'Eq_Medica',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })


    # MultiProcessBlock block: Process with MULTIPLE resources
    # Conferência e registro
    proc_cirurgico_pos_P23aP30 = MultiProcessBlock(
        "Conf_Registros", model.env,        
        resource_requirements={            
            Anestesista: 1,
            Eq_Medica: 1,
            Tec_Radiologia: 1,            
            Tec_Enfermagem: 1
        },        
        delay_time=lambda: distribution('3_conf_registros_pos_cir'),        
        event_logger=event_logger
    )    
    proc_cirurgico_pos_P23aP30.set_resource_names({        
        Anestesista: 'Anestesista',
        Eq_Medica: 'Eq_Medica',
        Tec_Radiologia: 'Tec_Radiologia',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })
    
    # ProcessBlock block: Process with ONE resource
    limpeza_organizacao_P37 = ProcessBlock(
        "Remov_Residuos", model.env,
        resource=Tec_Enfermagem,        
        delay_time=lambda: distribution('5_remocao_residuos'),
        resource_units=1,                 
        event_logger=event_logger
    )
    limpeza_organizacao_P37.set_resource_name('Tec_Enfermagem') 

    # ProcessBlock block: Process with ONE resource
    # limpeza_organizacao_P38 = ProcessBlock(
    limpeza_organizacao_P38 = MultiProcessBlock(
        "Sep_MatSujo", model.env,
        # resource=Func_CME,        
        resource_requirements={                        
            Func_CME: 1,            
            Tec_Enfermagem: 1
        },
        delay_time=lambda: distribution('5_sep_mat_sujos'),
        # resource_units=1,                 
        event_logger=event_logger
    )
    # limpeza_organizacao_P38.set_resource_name('Func_CME')
    limpeza_organizacao_P38.set_resource_names({                
        Func_CME: 'Func_CME',        
        Tec_Enfermagem: 'Tec_Enfermagem'
    })

    # ProcessBlock block: Process with ONE resource
    limpeza_organizacao_P39 = ProcessBlock(
        "Limpa_Sala_CC", model.env,
        resource=Eq_Higienizacao,        
        delay_time=lambda: distribution('5_limpeza_local'),
        resource_units=1,                 
        event_logger=event_logger
    )
    limpeza_organizacao_P39.set_resource_name('Eq_Higienizacao')
   
    
    # MultiProcessBlock block: Process with MULTIPLE resources
    #proc_materiais_P40aP44 = MultiProcessBlock(
    #    "Proc_Mat_Sujos", model.env,        
    #    resource_requirements={            
    #        Tec_Enfermagem: 1,            
    #        Func_CME: 1
    #    },        
    #    delay_time=lambda: distribution('6_processamento_materiais'),        
    #    event_logger=event_logger
    #)    
    #proc_materiais_P40aP44.set_resource_names({        
    #    Tec_Enfermagem: 'Tec_Enfermagem',       
    #    Func_CME: 'Func_CME'
    #})

    # # ProcessBlock block: Process with NO resource
    # ps_prepare = ProcessBlock(
    #     "PS prepara Encaminhamento", model.env,        
    #     delay_time=lambda: distribution('ps_prepare'),        
    #     event_logger=event_logger
    # )    
    # ============================================================
    
    # ============================ DECISIONS ====================
    # NOTE: arriv_CC_busy_decision was REMOVED.
    # The pre-check (sala_CC.level > 0) was redundant AND harmful:
    #   • When level == 0, the DecideBlock had no fallback route, so
    #     arriving patients were silently dropped instead of queuing.
    #   • Freed rooms were never claimed because nobody was in the
    #     Container's get-queue (all had been discarded upstream).
    # The simpy.Container.get(1) in SalaCC_Seize already blocks and
    # queues atomically — that IS the gate; no pre-check is needed.

    origem_paciente_decision = DecideBlock(
        "Origem_Paciente", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    porte_cirurgia_decision = DecideBlock(
        "Porte_Cir", model.env,
        decision_type="condition_generic",
        event_logger=event_logger
    )    

    faz_ex_radio_cir_p_decision = DecideBlock(
        "Faz_Rad_Cir_P", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    faz_ex_radio_cir_m_decision = DecideBlock(
        "Faz_Rad_Cir_M", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    faz_ex_radio_cir_g_decision = DecideBlock(
        "Faz_Rad_Cir_G", model.env,
        decision_type="probability",
        event_logger=event_logger
    )
    # ============================================================

    
    # ============================ DISPOSALS ====================            
    # discharge_arrival = DisposeBlock("Saida_CC_Busy", model.env, event_logger=event_logger)     
    discharge_srpa = DisposeBlock("Saida_SRPA", model.env, event_logger=event_logger)     
    # ============================================================

    # ============================ INCLUDE ALL BLOCKS ====================    
    # Add blocks to model
    for block in [arrivals_cc,
                #   delay_ag_cc,
                #   discharge_arrival,
                  seize_sala_cc,
                  prep_sala_P16, prep_sala_P2, prep_sala_P3a9,
                  origem_paciente_decision, 
                  adm_conf_paciente_P11a,                   
                  adm_conf_paciente_P11b,                   
                  adm_conf_paciente_P12a15, adm_paciente_P1617, 
                  proc_cirurgico_P18, proc_cirurgico_P19, 
                  porte_cirurgia_decision,
                  faz_ex_radio_cir_p_decision, faz_ex_radio_cir_m_decision, faz_ex_radio_cir_g_decision,
                  proc_cirurgico_P_P20_025, proc_cirurgico_P_P20aP22_015, proc_cirurgico_P_P20aP22_015_Radio, proc_cirurgico_P_P20_060, 
                  proc_cirurgico_M_P20_025, proc_cirurgico_M_P20aP22_015, proc_cirurgico_M_P20aP22_015_Radio, proc_cirurgico_M_P20_060, 
                  proc_cirurgico_G_P20_025, proc_cirurgico_G_P20aP22_015, proc_cirurgico_G_P20aP22_015_Radio, proc_cirurgico_G_P20_060,
                  proc_cirurgico_pos_P23aP30,
                  limpeza_organizacao_P37, limpeza_organizacao_P38, limpeza_organizacao_P39,
                  release_sala_cc,
                  discharge_srpa
                  ]:
        model.add_block(block)
    # ====================================================================
    
    # ============================ CONNECT ALL BLOCKS ====================    
    # FIX: arrivals go directly to the Container gate.
    # The yield container.get(1) inside SalaCC_Seize is the atomic gate;
    # it queues entities natively when all 5 rooms are occupied and
    # releases them one-by-one as each Release_Sala_CC fires put(1).
    arrivals_cc.connect_to(seize_sala_cc)

    # (removed arriv_CC_busy_decision routing — see comment above)

    seize_sala_cc.connect_to(prep_sala_P16)
    prep_sala_P16.connect_to(prep_sala_P2)
    prep_sala_P2.connect_to(prep_sala_P3a9)
    prep_sala_P3a9.connect_to(origem_paciente_decision)

    origem_paciente_decision.add_route("Pac_CTI", adm_conf_paciente_P11a, probability=0.136) # 13.6 %
    origem_paciente_decision.add_route("Pac_Outros", adm_conf_paciente_P11b, probability=0.864)

    adm_conf_paciente_P11a.connect_to(adm_conf_paciente_P12a15)    
    adm_conf_paciente_P11b.connect_to(adm_conf_paciente_P12a15)            
    adm_conf_paciente_P12a15.connect_to(adm_paciente_P1617)
    adm_paciente_P1617.connect_to(proc_cirurgico_P18)
    proc_cirurgico_P18.connect_to(proc_cirurgico_P19)
    proc_cirurgico_P19.connect_to(porte_cirurgia_decision)

    # porte_cirurgia_decision.add_route("Cir Pequeno", proc_cirurgico_P_P20_025, probability=0.4)
    # porte_cirurgia_decision.add_route("Cir Medio", proc_cirurgico_M_P20_025, probability=0.4)
    # porte_cirurgia_decision.add_route("Cir Grande", proc_cirurgico_G_P20_025, probability=0.2)

        
    # """Assign shift based on START time of the activity"""
    # if 7 <= hour < 13: "Manhã"
    # elif 13 <= hour < 19: "Tarde"
    # else:  "Noite"

    # Surgeries / Entries into Centro Cirúrgico by Shift (Real World)
    # Shift           % of Cases      Observation
    # Manhã (7h–13h)  ~46%            Dominant
    # Tarde (13h–19h) ~31%            Strong
    # Noite (19h–7h)  ~23%            Minority


    porte_cirurgia_decision.add_route("Cir Pequeno", proc_cirurgico_P_P20_025, condition_generic=is_pequeno)
    porte_cirurgia_decision.add_route("Cir Medio", proc_cirurgico_M_P20_025, condition_generic=is_medio)
    porte_cirurgia_decision.add_route("Cir Grande", proc_cirurgico_G_P20_025, condition_generic=is_grande)

    # ====================
    proc_cirurgico_P_P20_025.connect_to(faz_ex_radio_cir_p_decision)
    # For orthopedic/vascular surgery: 40%−70%
    #  For general surgery: 10%−30%
    faz_ex_radio_cir_p_decision.add_route("Cir_P_Sem_Radio", proc_cirurgico_P_P20aP22_015, probability=0.60)
    faz_ex_radio_cir_p_decision.add_route("Cir_P_Com_Radio", proc_cirurgico_P_P20aP22_015_Radio, probability=0.40)

    proc_cirurgico_P_P20aP22_015.connect_to(proc_cirurgico_P_P20_060)
    proc_cirurgico_P_P20aP22_015_Radio.connect_to(proc_cirurgico_P_P20_060)

    # ====================
    proc_cirurgico_M_P20_025.connect_to(faz_ex_radio_cir_m_decision)

    faz_ex_radio_cir_m_decision.add_route("Cir_M_Sem_Radio", proc_cirurgico_M_P20aP22_015, probability=0.60)
    faz_ex_radio_cir_m_decision.add_route("Cir_M_Com_Radio", proc_cirurgico_M_P20aP22_015_Radio, probability=0.40)

    proc_cirurgico_M_P20aP22_015.connect_to(proc_cirurgico_M_P20_060)
    proc_cirurgico_M_P20aP22_015_Radio.connect_to(proc_cirurgico_M_P20_060)
    # ====================

    proc_cirurgico_G_P20_025.connect_to(faz_ex_radio_cir_g_decision)

    faz_ex_radio_cir_g_decision.add_route("Cir_G_Sem_Radio", proc_cirurgico_G_P20aP22_015, probability=0.60)
    faz_ex_radio_cir_g_decision.add_route("Cir_G_Com_Radio", proc_cirurgico_G_P20aP22_015_Radio, probability=0.40)

    proc_cirurgico_G_P20aP22_015.connect_to(proc_cirurgico_G_P20_060)
    proc_cirurgico_G_P20aP22_015_Radio.connect_to(proc_cirurgico_G_P20_060)
    # ====================

    proc_cirurgico_P_P20_060.connect_to(proc_cirurgico_pos_P23aP30)
    proc_cirurgico_M_P20_060.connect_to(proc_cirurgico_pos_P23aP30)
    proc_cirurgico_G_P20_060.connect_to(proc_cirurgico_pos_P23aP30)

    proc_cirurgico_pos_P23aP30.connect_to(limpeza_organizacao_P37)  
    

    limpeza_organizacao_P37.connect_to(limpeza_organizacao_P38)
    limpeza_organizacao_P38.connect_to(limpeza_organizacao_P39)
    limpeza_organizacao_P39.connect_to(release_sala_cc)
    release_sala_cc.connect_to(discharge_srpa)
    # limpeza_organizacao_P39.connect_to(discharge_srpa)

    #proc_materiais_P40aP44.connect_to(discharge_srpa)
    

    # ================================================================
    # CONFIGURE FINANCIAL ATTRIBUTES
    # ================================================================    
    # Assign costs to each activity
    # arrivals_cc.assign_attributes(
    #     cost=lambda: random.uniform(0, 1)  # costs $0
    # )
    # Assign costs to each activity: Aviso_Cir
    prep_sala_P16.assign_attributes(
        cost=lambda: random.uniform(150, 175)  # costs $175
    ) 
     # Assign costs to each activity: Monta_Kits
    prep_sala_P2.assign_attributes(
        cost=lambda: random.uniform(150, 175)  # costs $175
    )    
     # Assign costs to each activity: Org_Sala_Kits
    prep_sala_P3a9.assign_attributes(
        cost=lambda: random.uniform(300, 360)  # costs $351
    ) 
     # Assign costs to each activity: Transp_CTI_CC
    adm_conf_paciente_P11a.assign_attributes(
        cost=lambda: random.uniform(0, 1)  # costs $0
    ) 
     # Assign costs to each activity
    adm_conf_paciente_P11b.assign_attributes(
        cost=lambda: random.uniform(0, 1)  # costs $0
    ) 
     # Assign costs to each activity: Adm_Conf_Pac
    adm_conf_paciente_P12a15.assign_attributes(
        cost=lambda: random.uniform(150, 175)  # costs $175
    ) 
     # Assign costs to each activity: Adm_Paciente
    adm_paciente_P1617.assign_attributes(
        cost=lambda: random.uniform(300, 350)  # costs $350
    ) 
     # Assign costs to each activity: Check_Cir_Seg
    proc_cirurgico_P18.assign_attributes(
        cost=lambda: random.uniform(150, 175)  # costs $175
    ) 
     # Assign costs to each activity: Ato_Anest
    proc_cirurgico_P19.assign_attributes(
        cost=lambda: random.uniform(700, 1000)  # costs $1054
    ) 
    # =======================
    # Assign costs to each activity: Cir_Pequena_025
    proc_cirurgico_P_P20_025.assign_attributes(
        cost=lambda: 0.25*random.uniform(3000, 3500)  # Surgery costs $3867
    )
    # Assign costs to each activity: Cir_Pequena_015
    proc_cirurgico_P_P20aP22_015.assign_attributes(
        cost=lambda: 0.15*random.uniform(3000, 3500)  # Surgery costs $3867
    )
    # Assign costs to each activity: Radiologia
    proc_cirurgico_P_P20aP22_015_Radio.assign_attributes(
        cost=lambda: random.uniform(0, 1)  # Surgery costs $1054 (em paralelo)
    )
    # Assign costs to each activity Cir_Pequena_060
    proc_cirurgico_P_P20_060.assign_attributes(
        cost=lambda: 0.60*random.uniform(3000, 3500)  # Surgery costs $3867
    )
    # =======================
    # =======================
    # Assign costs to each activity
    proc_cirurgico_M_P20_025.assign_attributes(
        cost=lambda: 0.25*random.uniform(3000, 3500)  # Surgery costs $4042
    )
    # Assign costs to each activity
    proc_cirurgico_M_P20aP22_015.assign_attributes(
        cost=lambda: 0.15*random.uniform(3000, 3500)  # Surgery costs $4042
    )
    # Assign costs to each activity
    proc_cirurgico_M_P20aP22_015_Radio.assign_attributes(
        cost=lambda: random.uniform(0, 1)  # Surgery costs $100-175
    )
    # Assign costs to each activity
    proc_cirurgico_M_P20_060.assign_attributes(
        cost=lambda: 0.60*random.uniform(3000, 3500)  # Surgery costs $4042
    )
    # =======================
    # =======================
    # Assign costs to each activity
    proc_cirurgico_G_P20_025.assign_attributes(
        cost=lambda: 0.25*random.uniform(3000, 3500)  # Surgery costs $4218
    )
    # Assign costs to each activity
    proc_cirurgico_G_P20aP22_015.assign_attributes(
        cost=lambda: 0.15*random.uniform(3000, 3500)  # Surgery costs $4218
    )
    # Assign costs to each activity
    proc_cirurgico_G_P20aP22_015_Radio.assign_attributes(
        cost=lambda: random.uniform(0, 1)  # Surgery costs $100-175
    )
    # Assign costs to each activity
    proc_cirurgico_G_P20_060.assign_attributes(
        cost=lambda: 0.60*random.uniform(3000, 3500)  # Surgery costs $4218
    )
    # =======================

    # Assign costs to each activity: Remov_Residuos
    limpeza_organizacao_P37.assign_attributes(
        cost=lambda: random.uniform(150, 175)  # costs $175
    ) 
    # Assign costs to each activity: Sep_MatSujo
    limpeza_organizacao_P38.assign_attributes(
        cost=lambda: random.uniform(150, 175)  # costs $175
    ) 
    # Assign costs to each activity: limpeza_organizacao
    limpeza_organizacao_P39.assign_attributes(
        cost=lambda: random.uniform(450, 550)  # costs $527
    ) 
    # Assign costs to each activity
    #proc_materiais_P40aP44.assign_attributes(
    #    cost=lambda: random.uniform(20, 30)  # costs $20-30
    #) 
    
    # Assign revenue at discharge (based on patient complexity)
    def calculate_revenue():
        """Revenue varies by patient complexity"""
        return random.uniform(800, 1100) # Revenue: $ 1514
    
    discharge_srpa.assign_attributes(revenue=calculate_revenue)                
    # ================================================================
    
    return model


def simulation_wrapper(seed=None, until=None, warm_up_period=None):
    """Wrapper function for replication framework."""
    
    from desk.core.entity import EventLogger
    
    event_logger = EventLogger()

    HOURS = 60  # Time conversion factor (base time: minutes)
    DAYS = 1440
    YEARS = 525600

    # Create configuration
    config = SimulationConfig(
        duration=31*DAYS,
        warm_up_period=3*DAYS,        
        seed=123,
        check_stability=True
    )

    # model = build_model(event_logger)
    model = build_model(config.duration, event_logger, verbose=False)    

    # model.run_simulation(
    #     until=until or 24*60,
    #     seed=seed,
    #     warm_up_period=warm_up_period or 2*60
    # )
    model.run_simulation(
        validate_resources=False,
        until=until,
        seed=seed,
        warm_up_period=warm_up_period
    )
    
    return model

# ================================================================
# For full simulation
# ================================================================
# Run replications
def run_replications():
    replication_framework = ReplicationFramework(
        simulation_function=simulation_wrapper,
        n_replications=30
    )

    HOURS = 60  # Time conversion factor (base time: minutes)
    DAYS = 1440
    YEARS = 525600
    
    replication_framework.run_replications(
        base_seed=12345,
        # until=365*DAYS,
        # warm_up_period=30*DAYS
        until=31*DAYS,
        warm_up_period=3*DAYS
    )

    # Access results
    df = replication_framework.get_results_dataframe()
    print(df.describe())
# ================================================================
    

# ================================================================
# Factorial Analysis
# ================================================================
def factorial_analysis():
    """Example of factorial analysis with hospital simulation."""

    HOURS = 60  # Time conversion factor (base time: minutes)
    DAYS = 1440
    YEARS = 525600
    
    # Define simulation function wrapper
    def simulation_wrapper(arrival_rate=12, Eq_Medica=4, 
                                    seed=None, until=None, warm_up_period=0, verbose=False, **kwargs):
        """Wrapper that adapts parameters for factorial analysis."""

        # ############################################################
        # # O modelo de simulação é importado aqui
        # ############################################################
        
        # This would need to be modified in your actual model to accept these parameters
        # For now, this is a template showing how to structure it
        model = build_model(verbose=False)
        model.run_simulation(validate_resources=False, until=until, seed=seed, warm_up_period=warm_up_period)
        return model
    
    # Create factorial analysis
    factorial = FactorialExperiment(
        simulation_function=simulation_wrapper,
        base_seed=12345
    )
    
    # Add factors
    factorial.add_factor(
        factor_name='arrival_rate',
        parameter_path='CreateBlock.inter_arrival_time',
        levels=[10, 12, 14],  # Minutes between arrivals
        description='Intervalo entre chegadas de pacientes (min)'
    )
    
    # factorial.add_factor(
    #     factor_name='num_physicians',
    #     parameter_path='Resource.physicians.capacity',
    #     levels=[1, 5, 10],
    #     description='Número de médicos'
    # )

    factorial.add_factor(
        factor_name='num_Eq_Medica',
        parameter_path='Resource.Eq_Medica.capacity',
        levels=[3, 4, 5],
        description='Número de equipes Medicas'
    )
    
    # factorial.add_factor(
    #     factor_name='num_nurses',
    #     parameter_path='Resource.nurses.capacity',
    #     levels=[8, 10, 12],
    #     description='Número de enfermeiros'
    # )
    
    # Run experiment
    factorial.run_factorial_experiment(
        n_replications=5,
        # simulation_time=365*DAYS,  # 40 hours
        # warm_up_period=30*DAYS,    # 7 hours
        simulation_time=31*DAYS,  # 40 hours
        warm_up_period=3*DAYS,    # 7 hours
        verbose=True
    )
    
    # Analyze results
    factorial.print_summary()
    factorial.plot_correlation_matrix()
    factorial.plot_main_effects('system_time_avg')
    factorial.plot_interaction_effects('system_time_avg', 'arrival_rate', 'num_Eq_Medica')
    
    # Export
    factorial.export_results()

    print("\n\nFactorial analysis examples completed!")
    print("Check the generated CSV files and plots for detailed results.")
    
    return factorial
# ================================================================

def pause_simulation(message="Continue? (Enter=yes / n=no): "):
    answer = input(message)
    if answer.lower().startswith('n'):
        print(f"Simulation stopped!")
        sys.exit()  # stops the simulation


def main():
    """Main example demonstrating refactored usage."""
    
    HOURS = 60  # Time conversion factor (base time: Minutos)
    DAYS = 1440
    YEARS = 525600
    
    # Create event logger
    event_logger = EventLogger()
    
    # Build model
    print("Building Detailed CC model...")

    # Create configuration
    config = SimulationConfig(
        # warm_up_period=0
        # until=20
        # duration=24*HOURS,
        # warm_up_period=2*HOURS,
        duration=31*DAYS,
        warm_up_period=3*DAYS,        
        seed=321,
        check_stability=True
    )
    config.validate()
    
    # model = build_model(event_logger)
    # model = build_model(config.duration, event_logger, verbose=True)
    model = build_model(config.duration, event_logger, verbose=False)


    # --- NEW: create the builder right after the model exists ---
    builder = MasterReportBuilder(model, run_name="cc")
    
    
    
    # Check stability BEFORE running (optional)
    print("\nChecking system stability...")
    stability_analyzer = StabilityAnalyzer(model)
    stability = stability_analyzer.check_system_stability()
    model.stability_result = stability


    def _replay_stability_result():
        # Cheap: just re-prints the cached result, no re-sampling.
        print(f"🎯 STABILITY INDEX: {stability:.2f}")
        stability_analyzer._print_stability_assessment(stability)
 
    builder.add_section("SYSTEM STABILITY CHECK", _replay_stability_result)

    
    # Run simulation
    print("\nRunning simulation (replication)...")
    model.run_simulation(
        validate_resources=True,  # Default True
        until=config.duration,
        seed=config.seed,
        warm_up_period=config.warm_up_period
    )
    
    
    # # === ANALYSIS PHASE (using separate modules) ===
    # #     
    # # ========================================
    # # Trace specific patient
    # # ========================================    
    # print("\n" + "="*80)
    # print("FILTER: Journey of CC_Patient_1")
    # print("="*80)    
    # pause_simulation()
    # model.trace_entity('CC_Patient_1')        
    
    # # ========================================
    # # Replay with filters
    # # ========================================
    # print("\n" + "="*80)
    # print("FILTER: Replay - First 3 patients only")
    # print("="*80)    
    # pause_simulation()
    # model.replay_trace(entity_pattern = r'^Patient_[1-3]$')
    
    # # ========================================
    # # Trace specific resource
    # # ========================================
    # print("\n" + "="*80)
    # print("FILTER: Replay - CC Tec_Enfermagem interactions only")
    # print("="*80)    
    # pause_simulation()
    # model.replay_trace(resource_filter={'Tec_Enfermagem'})
    
    # # ========================================
    # # Trace specific event types
    # # ========================================
    # print("\n" + "="*80)
    # print("FILTER: Replay - Queue and service events only")
    # print("="*80)    
    # pause_simulation()
    # model.replay_trace(event_type_filter={'queue', 'service_start', 'service_end'})
    
    # # ========================================
    # # Trace time window
    # # ========================================
    # print("\n" + "="*80)
    # print("FILTER: Replay - Events between t=20 and t=40")
    # print("="*80)    
    # pause_simulation()
    # model.replay_trace(time_range=(20, 40))
    
    # # ========================================
    # # Combined filters
    # # ========================================
    # print("\n" + "="*80)
    # print("FILTER: Replay - Patient_1 at physicians (queue + service)")
    # print("="*80)    
    # pause_simulation()
    # model.replay_trace(
    #     entity_filter={'Patient_1'},
    #     resource_filter={'physicians'},
    #     event_type_filter={'queue', 'service_start', 'service_end'}
    # )    

    # # ========================================
    # # Multiple patient journeys
    # # ========================================
    # print("\n" + "="*80)
    # print("FILTER: Detailed journeys of first 3 patients")
    # print("="*80)    
    # pause_simulation()
    # model.trace_entities(['Patient_1', 'Patient_2', 'Patient_3'])
    

    # ========================================
    # Trace statistics
    # ========================================
    model.print_trace_statistics()
    # pause_simulation()

        
    print("\n" + "="*60)
    print("SIMULATION COMPLETE - ANALYZING RESULTS")
    print("="*60)
    
    # 2. Detailed reporting
    print("\n" + "="*70)
    print("REPORTING SCOPE NOTE")
    print("="*70)
    if ENABLE_BACKGROUND_WORKLOAD:
        print(
            "  Resource utilization figures INCLUDE background workload:\n"
            "  documentation, training, stocking, equipment checks, and\n"
            "  other activities NOT part of the surgical pathway model.\n"
            "  Background parameters: BACKGROUND_WORKLOAD (see global config).\n"
            "  To see surgical-pathway-only utilization, set\n"
            "  ENABLE_BACKGROUND_WORKLOAD = False and re-run."
        )
    else:
        print(
            "  ⚠️  Background workload is DISABLED (ENABLE_BACKGROUND_WORKLOAD=False).\n"
            "  Utilization figures reflect ONLY modeled surgical pathway activities.\n"
            "  Staff perform many other tasks not captured here.\n"
            "  Enable BACKGROUND_WORKLOAD for a more realistic utilization picture."
        )
    print("="*70)


    # --- CHANGED: reporting tail now goes through the builder ---
    reporter = SimulationReporter(model)
    warmup_analyzer = WarmUpAnalyzer(model)
    financial_analyzer = FinancialAnalyzer(model)
 
    builder.add_section("SIMULATION RESULTS", reporter.print_results)
    builder.add_section("WARM-UP PERIOD ANALYSIS", warmup_analyzer.analyze_warm_up_period)
    builder.add_section("ACTIVITY METRICS", reporter._print_activity_metrics)
    builder.add_section("RESOURCE METRICS", reporter._print_resource_metrics)
    builder.add_section("ENTITY COUNTS", reporter._print_entity_counts)
    builder.add_section("BLOCK STATISTICS", reporter._print_block_statistics)
    builder.add_section("FINANCIAL BALANCE SHEET", financial_analyzer.print_financial_summary)
        
    # Runs every section, mirrors to terminal as before, and writes ONE file.
    report_path = builder.run_and_save(output_dir="results")    
    print(f"Report successfully saved to: {report_path}")
    # -> results/checkout_model_v3_report_20260618_143000.txt



    # print("\nResults...")
    # reporter.print_results()
    # print("\nAnalyzing warm-up period...")
    # warmup_analyzer.analyze_warm_up_period()
    # print("\nActivity metrics...")
    # reporter._print_activity_metrics()
    # print("\nResourse summary...")
    # reporter._print_resource_metrics()
    # reporter._print_entity_counts()
    # reporter._print_block_statistics()
    # print("\nFinancial analysys...")
    # financial_analyzer.print_financial_summary()


    # Plotting    
    plotter = SimulationPlotter(model)
    
    # # Plot resource utilization over time    
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='Enfermeiro', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='Farmacia', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='Tec_Enfermagem', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='Eq_Assistencial_CTI', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='Eq_Medica', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='Anestesista', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='Tec_Radiologia', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='Func_CME', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='Eq_Higienizacao', moving_average_window=50)    
    # plotter.plot_wip_over_time() 
    # plotter.plot_system_time_distribution()
    # plotter.plot_activity_metrics()
    plotter.plot_resources_utilization()
    financial_analyzer.plot_financial_breakdown()

    
    

    # Export event log
    print("\nExporting event log...")
    df = event_logger.export_to_csv("results/cc_event_log.csv")
    print(f"\nFirst 10 events:")
    print(df.head(10))
    
    # 6. Direct metrics access (if needed)
    metrics = MetricsCollector(model)
    entity_metrics = metrics.get_entity_metrics_summary()
    resource_metrics = metrics.get_resource_metrics_summary()
    
    print(f"\nAverage system time: {entity_metrics['tempo_medio_sistema']:.2f} min")    
    print(f"Random seed for this run: {config.seed}")    

    
    return model, event_logger












# ===========================================
# Simulation Kit
# ===========================================
def run_single_replication():
    return main()


def run_replications_cli():
    run_replications()


def run_factorial_cli():
    return factorial_analysis()


def run_visualization_cli(simulation_time=15*DAYS):
    return run_visualization(build_model, simulation_time=simulation_time)
# ===========================================