# =====================================================================
# FILE: hrtn.py
# =====================================================================
import random
import math
import sys
import simpy
from desk.stats.factorial import FactorialExperiment
from desk.stats.replication import ReplicationFramework    
from desk.analytics.financial import FinancialAnalyzer
from desk.validation.resource_validator import ResourceValidator
from desk.core.simulation_model import SimulationModel
from desk.core.entity import EventLogger
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
# desk-sim -m hrtn/hrtn.py --mode visualization
# desk-sim -m hrtn/hrtn.py --mode single
# desk-sim -m hrtn/hrtn.py --mode replications
# desk-sim -m hrtn/hrtn.py --mode factorial (no)

# ================================================================
# Desk-sim: DIST-FIT Which is the best data distribution?
# ================================================================
# desk-distfit -d hrtn/input/ps_los_dias.txt --remove-outliers --max-sample 500

# ================================================================
HOURS = 60  # Time conversion factor (base time: Minutos)
DAYS = 1440
YEARS = 525600
# ================================================================


# =====================================================================
# NHPP ARRIVAL GENERATOR — Piecewise-constant Non-Homogeneous Poisson
# Process, with independent, separately-fitted profiles for the two
# hospital entry points: Pronto-Socorro (PS) and Maternidade (MAT).
# =====================================================================
#
# Each 2h slot is assumed LOCALLY HOMOGENEOUS POISSON: within a slot,
# arrivals follow independent Exponential(lambda_slot) interarrival
# times, where lambda_slot is constant for the duration of that slot.
# The rate changes only at slot boundaries (piecewise-constant lambda(t)).
#
# Both PS and MAT get their OWN slot-fraction tables, base daily rate,
# and per-weekday volume factors, fitted directly from the historical
# arrival logs (ps_entradas.txt / mat_entradas.txt).
#
# Fitting method (per stream):
#   base_arrivals_per_day = total_arrivals / n_calendar_days_in_log
#   weekday_factors[d]    = mean_daily_arrivals_on_weekday_d / base_arrivals_per_day
#   slot fractions        = arrivals_in_2h_slot / total_arrivals,
#                            computed SEPARATELY for weekday (Mon-Fri)
#                            and weekend (Sat-Sun) subsets, each
#                            renormalized to sum to exactly 1.0.
#
# weekday_factors keys follow Python's weekday() / pandas dayofweek
# convention: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun.
# =====================================================================

# ---- Pronto-Socorro (PS) --------------------------------------------
# (2023) 65625/365
# (2024) 66515/365
# (2025) 59307/365
BASE_ARRIVALS_PER_DAY_PS = 162 

WEEKDAY_FACTORS_PS = {
    0: 1.183916,  # Mon
    1: 1.049006,  # Tue
    2: 1.028956,  # Wed
    3: 1.007191,  # Thu
    4: 0.968607,  # Fri
    5: 0.897970,  # Sat
    6: 0.863270,  # Sun
}

ARRIVAL_SLOTS_PS_WEEKDAY = [
    ( 0,  2, 0.043928), ( 2,  4, 0.027665), ( 4,  6, 0.028345),
    ( 6,  8, 0.072299), ( 8, 10, 0.124425), (10, 12, 0.129739),
    (12, 14, 0.113356), (14, 16, 0.113646), (16, 18, 0.100216),
    (18, 20, 0.098988), (20, 22, 0.083638), (22, 24, 0.063755),
]

ARRIVAL_SLOTS_PS_WEEKEND = [
    ( 0,  2, 0.060226), ( 2,  4, 0.039436), ( 4,  6, 0.032481),
    ( 6,  8, 0.055959), ( 8, 10, 0.095489), (10, 12, 0.122444),
    (12, 14, 0.105658), (14, 16, 0.106541), (16, 18, 0.108985),
    (18, 20, 0.104380), (20, 22, 0.090714), (22, 24, 0.077688),
]

# ---- Maternidade (MAT) ------------------------------------------------
# (2023) 19394/365
# (2024) 18140/365
# (2025) 18854/365
BASE_ARRIVALS_PER_DAY_MAT = 52

WEEKDAY_FACTORS_MAT = {
    0: 1.108033,  # Mon
    1: 1.073349,  # Tue
    2: 1.027320,  # Wed
    3: 1.030110,  # Thu
    4: 1.052148,  # Fri
    5: 0.907646,  # Sat
    6: 0.799806,  # Sun
}

ARRIVAL_SLOTS_MAT_WEEKDAY = [
    ( 0,  2, 0.039402), ( 2,  4, 0.029965), ( 4,  6, 0.030492),
    ( 6,  8, 0.053884), ( 8, 10, 0.103216), (10, 12, 0.120510),
    (12, 14, 0.125149), (14, 16, 0.140281), (16, 18, 0.114464),
    (18, 20, 0.091652), (20, 22, 0.086520), (22, 24, 0.064464),
]

ARRIVAL_SLOTS_MAT_WEEKEND = [
    ( 0,  2, 0.053897), ( 2,  4, 0.035804), ( 4,  6, 0.036077),
    ( 6,  8, 0.054061), ( 8, 10, 0.089811), (10, 12, 0.121679),
    (12, 14, 0.115065), (14, 16, 0.120531), (16, 18, 0.119875),
    (18, 20, 0.087078), (20, 22, 0.091232), (22, 24, 0.074888),
]

# Weekday-of-week that simulation time t=0 corresponds to.
# ps_entradas.txt / mat_entradas.txt both start on 01/01/2024, a Monday,
# so 0 (Monday) is used as the default. Adjust if a run's t=0 should
# represent a different calendar day.
SIM_START_DAY_OF_WEEK = 0


def make_nhpp_interarrival(
    env,
    arrival_slots,
    base_arrivals_per_day,
    weekday_factors,
    start_day_of_week=2,
    arrival_slots_weekend=None,
    weekend_days=(5, 6)):
    """
    Piecewise-constant Non-Homogeneous Poisson Process, with separate
    weekday / weekend arrival profiles.

    Each 2h slot is treated as LOCALLY HOMOGENEOUS POISSON: within a
    slot, interarrival times are independent Exponential(lambda_slot)
    draws, with lambda_slot fixed for the duration of that slot. Both
    the rate (_lambda_at) and the slot boundary (_slot_end_absolute)
    consult the SAME day-type profile (weekday vs weekend), so the
    inversion walk never mixes weekday slot widths with weekend days
    (or vice versa) when it crosses a midnight boundary.

    ALGORITHM — Piecewise Inversion Method
    ────────────────────────────────────────────────
    1. Draw E ~ Exponential(1): one "unit of expected arrivals to consume".
    2. Starting at the current simulation time t, walk forward slot by slot.
    3. For each slot, compute the expected arrivals remaining in that slot
       from position t: Delta = lambda_slot * (slot_end - t).
    4. If E <= Delta -> the next arrival lands inside this slot at
       t + E / lambda_slot.
       If E > Delta  -> subtract Delta, advance t to the next slot
       boundary, repeat.
    5. Correctly inherits the day-of-week factor AND the correct slot
       profile as t crosses midnight.

    Parameters
    ──────────
    env                   : simpy.Environment
    arrival_slots         : list[(start_h, end_h, fraction)]  weekday profile
    base_arrivals_per_day : float  historical global daily mean
    weekday_factors       : dict {weekday_int: float}  wf = day_mean / global_mean
    start_day_of_week     : int  Python weekday (0=Mon...6=Sun) of simulation t=0
    arrival_slots_weekend : list[(start_h, end_h, fraction)] or None
                            Weekend profile; if None falls back to arrival_slots.
    weekend_days          : tuple of weekday ints treated as weekend (default (5,6))
    """

    MINUTES_PER_DAY = 1440

    _slots_weekday = arrival_slots
    _slots_weekend = arrival_slots_weekend if arrival_slots_weekend is not None else arrival_slots

    # Safety validation — both profiles must partition [0, 24).
    # Tolerance is 1e-4 (not 1e-6) because fractions are stored rounded
    # to 6 decimals for readability, which can leave a ~1e-6 sum residual
    # per slot — this is fine; anything above 1e-4 indicates a real error.
    for label, slots in (("weekday", _slots_weekday), ("weekend", _slots_weekend)):
        total = sum(f for _, _, f in slots)
        if abs(total - 1.0) > 1e-4:
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
        """Return the arrival rate (arrivals/min) at a given absolute minute."""
        _, wf, slots = _weekday_and_slots(absolute_minute)
        hour = (absolute_minute % MINUTES_PER_DAY) / 60.0
        for start_h, end_h, fraction in slots:
            if start_h <= hour < end_h:
                slot_min = (end_h - start_h) * 60
                return max(base_arrivals_per_day * wf * fraction / slot_min, 1e-9)
        return 1e-9  # safety fallback

    def _slot_end_absolute(absolute_minute):
        """Return the absolute minute at which the current slot ends."""
        day_start = int(absolute_minute // MINUTES_PER_DAY) * MINUTES_PER_DAY
        _, _, slots = _weekday_and_slots(absolute_minute)
        hour = (absolute_minute % MINUTES_PER_DAY) / 60.0
        for start_h, end_h, _ in slots:
            if start_h <= hour < end_h:
                return day_start + end_h * 60
        # Fallback: advance to the start of the next day
        return day_start + MINUTES_PER_DAY

    def interarrival():
        # Step 1: draw one Exp(1) variate — "amount of expected arrivals"
        # to consume before the next event.
        e = random.expovariate(1.0)

        # Step 2: walk forward through slots consuming 'e'.
        t = env.now
        for _guard in range(10000):          # guard against infinite loops
            lam      = _lambda_at(t)
            slot_end = _slot_end_absolute(t)
            remaining_in_slot = slot_end - t

            delta = lam * remaining_in_slot

            if e <= delta:
                dt = e / lam
                return t + dt - env.now      # return the interarrival duration

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
    
    def truncated_weibull(alpha, beta, min_los, max_los):
        while True:
            x = random.weibullvariate(alpha, beta)
            if min_los <= x <= max_los:
                return x

    def truncated_beta(a, b, loc, scale, min_los, max_los):
        while True:
            x = loc + scale * random.betavariate(a, b)
            if min_los <= x <= max_los:
                return x
    
    def truncated_gamma(alpha, theta, min_los, max_los):
        while True:
            x = random.gammavariate(alpha, theta)
            if min_los <= x <= max_los:
                return x

    def truncated_normal(mu, sigma, min_los, max_los):        
        while True:
            x = random.normalvariate(mu, sigma)
            if min_los <= x <= max_los:
                return x
                
    # # Unidade básica para todos os tempos: minutos
    # def distribution(tipo):
    #     arrival_rate_ps=169.978814/DAYS # por minuto  λ (lambda): 169.978814 patients/day
    #     arrival_rate_mat=49.994530/DAYS # por minuto  λ (lambda): 49.994530 patients/day
    #     return {            
    #         'arrival_ps': random.expovariate(arrival_rate_ps),
    #         'arrival_mat': random.expovariate(arrival_rate_mat),
    #         'amb_monitoring': random.lognormvariate(-0.071, 0.176)*DAYS,
    #         'cc_monitoring': random.lognormvariate(0.003, 0.182)*DAYS,
    #         'clic_monitoring': random.lognormvariate(-0.255, 4.597)*DAYS,
    #         'clim_monitoring': random.lognormvariate(-0.747, 6.191)*DAYS,
    #         'cti_monitoring': random.lognormvariate(-0.194, 4.707)*DAYS,
    #         'mat_monitoring': random.lognormvariate(0.001, 0.137)*DAYS, 
    #         'ps_monitoring': random.lognormvariate(-0.000, 0.335)*DAYS, 
    #     }.get(tipo,0.0)

    # ------------------------------------------------------------------
    # NHPP arrival generators — one independent instance per entry point.
    # Each 2h slot is locally homogeneous Poisson (fixed-rate exponential
    # interarrival times within the slot); the rate is piecewise-constant
    # across slots and varies by weekday/weekend profile + weekday_factor.
    # Fitted directly from ps_entradas.txt / mat_entradas.txt.
    # ------------------------------------------------------------------
    interarrival_ps = make_nhpp_interarrival(
        model.env,
        arrival_slots=ARRIVAL_SLOTS_PS_WEEKDAY,
        base_arrivals_per_day=BASE_ARRIVALS_PER_DAY_PS,
        weekday_factors=WEEKDAY_FACTORS_PS,
        start_day_of_week=SIM_START_DAY_OF_WEEK,
        arrival_slots_weekend=ARRIVAL_SLOTS_PS_WEEKEND,
        weekend_days=(5, 6),
    )

    interarrival_mat = make_nhpp_interarrival(
        model.env,
        arrival_slots=ARRIVAL_SLOTS_MAT_WEEKDAY,
        base_arrivals_per_day=BASE_ARRIVALS_PER_DAY_MAT,
        weekday_factors=WEEKDAY_FACTORS_MAT,
        start_day_of_week=SIM_START_DAY_OF_WEEK,
        arrival_slots_weekend=ARRIVAL_SLOTS_MAT_WEEKEND,
        weekend_days=(5, 6),
    )

    def distribution(tipo):
        distributions = {            
            # Arrivals — piecewise-constant NHPP, separate profiles per stream
            'arrival_ps':  interarrival_ps,
            'arrival_mat': interarrival_mat,            
            
            # Length of stay (days)
            # Ambulatorial: 6,52h/1440 min/dia ~ 0,04 dias
            'amb_monitoring': lambda: truncated_lognormal(
                mu=-2.34334, sigma=0.62805,
                min_los=0.001, max_los=3
            )*DAYS,

            # Centro Cirúrgico
            'cc_monitoring': lambda: truncated_lognormal(
                mu=-2.01659, sigma=1.07589,
                min_los=0.01, max_los=1.0
            )*DAYS,

            # Sala de Recuperação Pós Anestesica
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC4256808/
            'srpa_monitoring': lambda: truncated_normal(
                mu=0.23, sigma=0.24,
                min_los=0.16, max_los=48.0
            )*DAYS,

            # Clínica Cirúrgica
            # random.weibullvariate(3.43116, 0.858684)
            'clic_monitoring': lambda: truncated_weibull(                
                alpha=3.43116, beta=0.858684,
                min_los=0.01, max_los=16
            )*DAYS,

            # Clínica Médica
            # 0.000694444 + 22.4936 * random.betavariate(0.662594, 3.41671)
            'clim_monitoring': lambda: truncated_beta(
                a=0.662594, b=3.41671,
                loc=0.000694444, scale=22.4936,
                min_los=0.01, max_los=18
            )*DAYS,

            # CTI / UTI
            # random.weibullvariate(5.25394, 1.10367)
            'cti_monitoring': lambda: truncated_weibull(                
                alpha=5.25394, beta=1.10367,
                min_los=0.01, max_los=50
            )*DAYS,

            # Maternidade
            # random.lognormvariate(-2.65333, 0.947878)
            'mat_monitoring': lambda: truncated_lognormal(
                mu=-2.65333, sigma=0.947878,
                min_los=0.01, max_los=10
            )*DAYS,            

            # Pronto-Socorro
            # random.gammavariate(1.35708, 0.224386)
            'ps_monitoring': lambda: truncated_gamma(
                alpha=1.35708, theta=0.224386,
                min_los=0.001, max_los=4.5
            )*DAYS,

            # # Imagem / Diagnóstico
            # 'img_monitoring': lambda: truncated_lognormal(
            #     mu=-2.83, sigma=2.05,
            #     min_los=0.001, max_los=15
            # ),

            'discharge_delay': lambda: truncated_normal(
                mu=(0.14)*3.137, sigma=(0.14)*0.357,
                min_los=0.01, max_los=5
            )*DAYS
        }

        return distributions.get(tipo, lambda: 0.0)()
    
    
    
    # ─────────────────────────────────────────────────────────────────────
    # CENTRO CIRURGICO (CC): 5 operating rooms  ->  13 SRPA recovery beds
    # ─────────────────────────────────────────────────────────────────────
    # Simplified design: two ordinary ProcessBlocks + model.add_resource.
    # The previous Container + custom-block chain (blocking-after-service)
    # is replaced so that standard resource_data logging and
    # plot_resource_use_over_time work out of the box.
    #
    # Trade-off: a patient now releases the OR as soon as surgery finishes
    # and then waits for an SRPA bed (no longer holds the OR while waiting).
    # The total CC episode duration is still drawn once and split by
    # CC_OR_TIME_FRACTION so the sum matches the fitted distribution.
    #
    # ⚠ ASSUMPTION — recalibrate when real data is available.
    # The historical fit for 'cc_monitoring' covers the WHOLE CC episode
    # (surgery + recovery). Change CC_OR_TIME_FRACTION to recalibrate.
    # CC_OR_TIME_FRACTION = 0.65  # fraction of total CC time spent IN the OR

    cc_rooms = model.add_resource("CC_Rooms", 5, "regular")
    srpaBeds = model.add_resource("srpaBeds", 13, "regular")

    # PS: 10-Cuidados_críticos, 7-emergência, 12-Retaguarda, 
    # PS: 12-Amarela_interno, 22-ortopedia, 35-Amarelo_externo, 
    # PS: 10-Cir_geral_plastica, 2-isolamento 
    # PS: Leitos: 10+7+22+10+2=51 Buffer: 12+12+35=59
    psBeds = model.add_resource("psBeds", 110, "regular") 
    # CLIM: 16-feminino, 16-masculino, 20-misto (1º andar)
    # CLIM: 28-misto, 28-AVC (5º andar)
    # CLIM: 28-misto, 28-Cuidados_paleativos (6º andar)
    climBeds = model.add_resource("climBeds", 164, "regular") 
    ctiBeds = model.add_resource("ctiBeds", 31, "regular")
    # CLIC: 12-(2º andar-b), 31-(3º andar-b), 24-(4º andar-a), 28-(4º andar-b)
    clicBeds = model.add_resource("clicBeds", 95, "regular")     
    # MAT: 12-Conjunto, 3-admissão, 3-PPP, 2-PP(2º andar-a)
    # MAT: 11-Conjunto, 4-mae_canguru, 6-UCI
    # MAP: Leitos: 12+3+3+2+11+4+6=38 Buffer: 3
    matBeds = model.add_resource("matBeds", 30, "regular")
    # AMB: 9 espaços de consulta, 4 para procedimentos e 100 cadeiras, para espera
    # Observou-se 5 espaços de consulta em média, na prática
    ambRooms = model.add_resource("ambRooms", 5, "regular") 
    # imgEquips = model.add_resource("imgEquips", 500, "regular")
    
    
    # ============================ ACTIVITIES ====================
    # Patient severity generator
    def patient_severity():
        severity_dist = [0.01, 0.21, 0.32, 0.44, 0.01, 0.02]
        return random.choices([1, 2, 3, 4, 5, 6], weights=severity_dist)[0]
    
    # Create block
    arrivals_ps = CreateBlock(
        "Chegadas_PS", model.env,
        # inter_arrival_time=lambda: random.expovariate(1/4),
        inter_arrival_time=lambda: distribution('arrival_ps'),
        entity_prefix="PS_Patient",
        max_arrivals=None, # Infinito
        first_creation=0.0,
        priority_generator=patient_severity,
        event_logger=event_logger
    )

    arrivals_mat = CreateBlock(
        "Chegadas_MAT", model.env,
        # inter_arrival_time=lambda: random.expovariate(1/4),
        inter_arrival_time=lambda: distribution('arrival_mat'),
        entity_prefix="MAT_Patient",
        max_arrivals=None, # Infinito
        first_creation=0.0,
        # priority_generator=patient_severity,
        event_logger=event_logger
    )

    # # MultiProcessBlock block: Process with MULTIPLE resources
    # admission = MultiProcessBlock(
    #     "Admission", model.env,        
    #     resource_requirements={            
    #         nurses: 1,
    #         nursingTech: 1,
    #         physicians: 1
    #     },        
    #     delay_time=lambda: distribution('admission'),        
    #     event_logger=event_logger
    # )    
    # admission.set_resource_names({        
    #     nurses: 'nurses',
    #     nursingTech: 'nursingTech',
    #     physicians: 'physicians'
    # })    

    # ProcessBlock block: Process with ONE resource
    amb_monitoring = ProcessBlock(
        "AMB", model.env,
        resource=ambRooms,        
        delay_time=lambda: distribution('amb_monitoring'),
        resource_units=1,                 # 1 ambBed per service
        buffer_capacity=100, # 100 cadeiras para espera
        event_logger=event_logger
    )
    amb_monitoring.set_resource_name('ambRooms')

    # ── Centro Cirúrgico: two ordinary ProcessBlocks (OR + SRPA) ──────────
    # Surgery (OR) then recovery (SRPA). Total episode duration is drawn
    # once and split by CC_OR_TIME_FRACTION so the sum still matches the
    # fitted distribution. No blocking-after-service (patient releases the
    # OR when surgery finishes, then queues for an SRPA bed).
    cc_monitoring = ProcessBlock(
        "CC", model.env,
        resource=cc_rooms,
        delay_time=lambda: distribution('cc_monitoring'),
        resource_units=1,
        buffer_capacity=10,          # queue in front of the ORs
        event_logger=event_logger
    )
    cc_monitoring.set_resource_name('CC_Rooms')

    srpa_monitoring = ProcessBlock(
        "SRPA", model.env,
        resource=srpaBeds,
        delay_time=lambda: distribution('srpa_monitoring'),
        resource_units=1,
        buffer_capacity=10,          # queue in front of SRPA beds
        event_logger=event_logger
    )
    srpa_monitoring.set_resource_name('srpaBeds')

    cc_monitoring.connect_to(srpa_monitoring)

    clic_monitoring = ProcessBlock(
        "CLIC", model.env,
        resource=clicBeds,        
        delay_time=lambda: distribution('clic_monitoring'),
        resource_units=1,                 # 1 ctiBed per service
        buffer_capacity=95, # 95 leitos
        event_logger=event_logger
    )
    clic_monitoring.set_resource_name('clicBeds')

    # ProcessBlock block: Process with ONE resource
    clim_monitoring = ProcessBlock(
        "CLIM", model.env,
        resource=climBeds,        
        delay_time=lambda: distribution('clim_monitoring'),
        resource_units=1,                 # 1 ctiBed per service
        buffer_capacity=164, # 164 leitos
        event_logger=event_logger
    )
    clim_monitoring.set_resource_name('climBeds')

    # ProcessBlock block: Process with ONE resource
    cti_monitoring = ProcessBlock(
        "CTI", model.env,
        resource=ctiBeds,        
        delay_time=lambda: distribution('cti_monitoring'),
        resource_units=1,                 # 1 ctiBed per service
        buffer_capacity=31, # 31 leitos CTI
        event_logger=event_logger
    )
    cti_monitoring.set_resource_name('ctiBeds')

    # ProcessBlock block: Process with ONE resource
    mat_monitoring = ProcessBlock(
        "MAT", model.env,
        resource=matBeds,        
        delay_time=lambda: distribution('mat_monitoring'),
        resource_units=1,                 # 1 matBed per service
        buffer_capacity=30, # Leitos 38 Buffer: 
        event_logger=event_logger
    )
    mat_monitoring.set_resource_name('matBeds')    

    # ProcessBlock block: Process with ONE resource
    ps_monitoring = ProcessBlock(
        "PS", model.env,
        resource=psBeds,        
        delay_time=lambda: distribution('ps_monitoring'),
        resource_units=1,                 # 1 psBed per service
        buffer_capacity=59, # Cap: 51 + Buffer: 59 leitos Pronto-Socorro
        event_logger=event_logger
    )
    ps_monitoring.set_resource_name('psBeds') 

    discharge_delay = ProcessBlock(
        "ALTA_HOSP", model.env,
        # resource=psBeds,        
        delay_time=lambda: distribution('discharge_delay'),
        # resource_units=0,                 # 1 ctiBed per service
        event_logger=event_logger
    )

    
    
    # # ProcessBlock block: Process with NO resource
    # ps_prepare = ProcessBlock(
    #     "PS prepara Encaminhamento", model.env,        
    #     delay_time=lambda: distribution('ps_prepare'),        
    #     event_logger=event_logger
    # )    
    # ============================================================
    
    # ============================ DECISIONS ====================
    # admission_decision = DecideBlock(
    #     "Origem", model.env,
    #     decision_type="probability",
    #     event_logger=event_logger
    # )

    # discharge_decision = DecideBlock(
    #     "Encaminha", model.env,
    #     decision_type="probability",
    #     event_logger=event_logger
    # )

    ps_decision = DecideBlock(
        "PS_Encaminha", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    mat_decision = DecideBlock(
        "MAT_Encaminha", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    clim_decision = DecideBlock(
        "CLIM_Encaminha", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    amb_decision = DecideBlock(
        "AMB_Encaminha", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    cc_decision = DecideBlock(
        "CC_Encaminha", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    clic_decision = DecideBlock(
        "CLIC_Encaminha", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    cti_decision = DecideBlock(
        "CTI_Encaminha", model.env,
        decision_type="probability",
        event_logger=event_logger
    )
    # ============================================================

    
    # ============================ DISPOSALS ====================            
    discharge_ah = DisposeBlock("Saida_AltaHosp", model.env, event_logger=event_logger) 
    discharge_des = DisposeBlock("Saida_Desist", model.env, event_logger=event_logger)  
    discharge_text = DisposeBlock("Saida_Transf_Ext", model.env, event_logger=event_logger) 
    discharge_death = DisposeBlock("Saida_Obito", model.env, event_logger=event_logger)     
    # ============================================================


    # ============================ INCLUDE ALL BLOCKS ====================    
    # Add blocks to model
    for block in [arrivals_ps, arrivals_mat, 
                  amb_monitoring, cc_monitoring, srpa_monitoring,
                  clic_monitoring,
                  clim_monitoring, cti_monitoring, mat_monitoring,
                  ps_monitoring, 
                  ps_decision, mat_decision, clim_decision, amb_decision,
                  cc_decision, clic_decision, cti_decision,
                  discharge_delay,
                  discharge_ah, discharge_des, discharge_text, discharge_death
                  ]:
        model.add_block(block)
    # ====================================================================
    
    # ============================ CONNECT ALL BLOCKS ====================    
    
    # ====================== Connect flow: Pronto Socorro==================
    arrivals_ps.connect_to(ps_monitoring)
    ps_monitoring.connect_to(ps_decision)

    # CC-share doubled: probabilities rescaled by k=1.939 on the CC route,
    # all other routes reduced proportionally so the block still sums to 1.
    ps_decision.add_route("Alta Hospitalar", discharge_delay, probability=0.5426)    
    discharge_delay.connect_to(discharge_ah)    
    
    # ps_decision.add_route("Alta Hospitalar", discharge_ah, probability=0.6541)
    ps_decision.add_route("Desistencia/Evasao", discharge_des, probability=0.1172)    
    ps_decision.add_route("para a Clinica Medica", clim_monitoring, probability=0.1686)    
    ps_decision.add_route("para a Clinica Cirurgica", clic_monitoring, probability=0.0804)    
    ps_decision.add_route("para o Centro Cirurgico", cc_monitoring, probability=0.0621)    
    ps_decision.add_route("Transferencia Externa", discharge_text, probability=0.0126)
    ps_decision.add_route("para o CTI", cti_monitoring, probability=0.0126)    
    ps_decision.add_route("Obito", discharge_death, probability=0.0019)
    ps_decision.add_route("para o Ambulatorio", amb_monitoring, probability=0.0010)
    ps_decision.add_route("para a Maternidade", mat_monitoring, probability=0.0010)

    # ====================== Connect flow: Maternidade ==================
    arrivals_mat.connect_to(mat_monitoring)
    mat_monitoring.connect_to(mat_decision)

    # CC-share doubled (same k=1.939 as ps_decision; see note above).
    mat_decision.add_route("Alta Hospitalar", discharge_delay, probability=0.9583)
    discharge_delay.connect_to(discharge_ah)

    # mat_decision.add_route("Alta Hospitalar", discharge_ah, probability=0.9465)    
    mat_decision.add_route("Desistencia/Evasao", discharge_des, probability=0.0170)
    mat_decision.add_route("Transferencia Externa", discharge_text, probability=0.0090)
    mat_decision.add_route("para a Clinica Cirurgica", clic_monitoring, probability=0.0020)
    mat_decision.add_route("para a Clinica Medica", clim_monitoring, probability=0.0020)    
    mat_decision.add_route("para o Pronto Socorro", ps_monitoring, probability=0.0010)
    mat_decision.add_route("para o Centro Cirurgico", cc_monitoring, probability=0.0058)    
    mat_decision.add_route("para o Ambulatorio", amb_monitoring, probability=0.0040)
    mat_decision.add_route("para o CTI", cti_monitoring, probability=0.0010)    
    mat_decision.add_route("Obito", discharge_death, probability=0.00)
    
    # ====================== Connect flow: Clinica Medica ==================    
    clim_monitoring.connect_to(clim_decision)

    # CC-share doubled (same k=1.939 as ps_decision; see note above).
    clim_decision.add_route("Alta Hospitalar", discharge_delay, probability=0.6330)
    discharge_delay.connect_to(discharge_ah)

    # clim_decision.add_route("Alta Hospitalar", discharge_ah, probability=0.6797)
    clim_decision.add_route("Obito", discharge_death, probability=0.0813)
    clim_decision.add_route("Transferencia Externa", discharge_text, probability=0.0774)
    clim_decision.add_route("para a Clinica Cirurgica", clic_monitoring, probability=0.0300)
    clim_decision.add_route("para o Centro Cirurgico", cc_monitoring, probability=0.0640)
    clim_decision.add_route("para o Pronto Socorro", ps_monitoring, probability=0.0774)    
    clim_decision.add_route("Desistencia/Evasao", discharge_des, probability=0.0116)
    clim_decision.add_route("para o CTI", cti_monitoring, probability=0.0184)    
    clim_decision.add_route("para o Ambulatorio", amb_monitoring, probability=0.0048)
    clim_decision.add_route("para a Maternidade", mat_monitoring, probability=0.0019)

    # ====================== Connect flow: Centro Cirurgico ==================
    # CC (surgery) -> SRPA (recovery) -> decision. Already connected
    # cc_monitoring -> srpa_monitoring above; now exit of SRPA to decision.
    srpa_monitoring.connect_to(cc_decision)

    cc_decision.add_route("Alta Hospitalar", discharge_delay, probability=0.051)
    discharge_delay.connect_to(discharge_ah)

    # cc_decision.add_route("Alta Hospitalar", discharge_ah, probability=0.0622)
    cc_decision.add_route("para a Clinica Cirurgica", clic_monitoring, probability=0.675)
    cc_decision.add_route("para o CTI", cti_monitoring, probability=0.149)
    cc_decision.add_route("para a Clinica Medica", clim_monitoring, probability=0.094)        
    cc_decision.add_route("para o Pronto Socorro", ps_monitoring, probability=0.017)
    cc_decision.add_route("para a Maternidade", mat_monitoring, probability=0.005)
    cc_decision.add_route("Obito", discharge_death, probability=0.004)
    cc_decision.add_route("para o Ambulatorio", amb_monitoring, probability=0.001)
    cc_decision.add_route("Transferencia Externa", discharge_text, probability=0.003)        
    cc_decision.add_route("Desistencia/Evasao", discharge_des, probability=0.001)
    
    # ====================== Connect flow: Clinica Cirurgica ==================    
    clic_monitoring.connect_to(clic_decision)

    # CC-share doubled (same k=1.939 as ps_decision; see note above).
    clic_decision.add_route("Alta Hospitalar", discharge_delay, probability=0.6382)
    discharge_delay.connect_to(discharge_ah)

    # clic_decision.add_route("Alta Hospitalar", discharge_ah, probability=0.6697)    
    clic_decision.add_route("Transferencia Externa", discharge_text, probability=0.1566)            
    clic_decision.add_route("para o Ambulatorio", amb_monitoring, probability=0.0425)
    clic_decision.add_route("para a Clinica Medica", clim_monitoring, probability=0.0396)
    clic_decision.add_route("Desistencia/Evasao", discharge_des, probability=0.0164)
    clic_decision.add_route("para o CTI", cti_monitoring, probability=0.0271)
    clic_decision.add_route("para o Centro Cirurgico", cc_monitoring, probability=0.0659)    
    clic_decision.add_route("Obito", discharge_death, probability=0.0039)    
    clic_decision.add_route("para o Pronto Socorro", ps_monitoring, probability=0.0097)    
    clic_decision.add_route("para a Maternidade", mat_monitoring, probability=0.0010)

    # ====================== Connect flow: CTI ==================    
    cti_monitoring.connect_to(cti_decision)

    # CC-share doubled (same k=1.939 as ps_decision; see note above).
    cti_decision.add_route("Alta Hospitalar", discharge_delay, probability=0.00)
    discharge_delay.connect_to(discharge_ah)

    # cti_decision.add_route("Alta Hospitalar", discharge_ah, probability=0.0014)
    cti_decision.add_route("para a Clinica Medica", clim_monitoring, probability=0.4081)
    cti_decision.add_route("para a Clinica Cirurgica", clic_monitoring, probability=0.3391)
    cti_decision.add_route("Obito", discharge_death, probability=0.1626)    
    cti_decision.add_route("Transferencia Externa", discharge_text, probability=0.0463)
    cti_decision.add_route("para a Maternidade", mat_monitoring, probability=0.0099)
    cti_decision.add_route("para o Centro Cirurgico", cc_monitoring, probability=0.0291)    
    cti_decision.add_route("para o Pronto Socorro", ps_monitoring, probability=0.0049)
    

    # ====================== Connect flow: Ambulatorio ==================    
    amb_monitoring.connect_to(amb_decision)

    amb_decision.add_route("Alta Hospitalar", discharge_delay, probability=0.989)
    discharge_delay.connect_to(discharge_ah)

    # amb_decision.add_route("Alta Hospitalar", discharge_ah, probability=0.9907)
    amb_decision.add_route("Desistencia/Evasao", discharge_des, probability=0.003)
    amb_decision.add_route("para a Clinica Cirurgica", clic_monitoring, probability=0.006)    
    amb_decision.add_route("para a Clinica Medica", clim_monitoring, probability=0.001)
    amb_decision.add_route("Transferencia Externa", discharge_text, probability=0.0005)
    amb_decision.add_route("para a Maternidade", mat_monitoring, probability=0.0005)  
    

    # ================================================================
    # CONFIGURE FINANCIAL ATTRIBUTES
    # https://www.hrtn.fundep.ufmg.br/detalhe-da-materia/info/18929/hospital-em-numeros/
    # ================================================================    
    # Assign costs to each activity
    arrivals_ps.assign_attributes(
        cost=lambda: random.uniform(0, 0)  # Triage costs $20-30
    )
    # Assign costs to each activity
    arrivals_mat.assign_attributes(
        cost=lambda: random.uniform(0, 0)  # Triage costs $20-30
    )    
    
    # https://www.hrtn.fundep.ufmg.br/detalhe-da-materia/info/18929/hospital-em-numeros/
    ps_monitoring.assign_attributes(
        cost=lambda: random.uniform(595, 600)
    )

    # https://www.hrtn.fundep.ufmg.br/detalhe-da-materia/info/18929/hospital-em-numeros/
    amb_monitoring.assign_attributes(
        cost=lambda: random.uniform(196, 200)
    )
   
    cc_monitoring.assign_attributes(
        cost=lambda: random.uniform(1925, 1930)
    )

    clic_monitoring.assign_attributes(
        cost=lambda: random.uniform(900, 1000)
    ) 

    clim_monitoring.assign_attributes(
        cost=lambda: random.uniform(900, 1000)
    )

    cti_monitoring.assign_attributes(
        cost=lambda: random.uniform(3000, 3300)
    )    

    mat_monitoring.assign_attributes(
        cost=lambda: random.uniform(1200, 1300)
    )  

# Consumo de materiais em 2023
# No ano de 2023 o valor total de consumo de materiais foi de R$ 16.952.446,03 
# (para item hospitalar, material de diagnóstico, material de CME, instrumental, 
# itens de higienização e limpeza, material administrativo, plástico e embalagem, 
# Equipamentos de Proteção Individual, material de segurança, enxoval e uniformes 
# e material de informática). 
# 
# Consumo de medicamentos em 2023
# Dispensamos 4.352.348 unidades de medicamentos no ano de 2023, o que representou 
# um custo total de R$ 13.204.599,72. 

# Produção X Faturamento em 2023
# Valor médio da AIH faturada em 2023: R$ 1.298,29
# Produção AIH em 2023: R$ 27.754.786,04
# AIHs faturadas em 2023: 21.378

# Roupa lavada (em média, em 2023)
# 747.600 kg ano: Roupa Limpa
# 780.000 kg ano: Roupa Suja

# Número de refeições servidas em 2023
# 652.285 Para pacientes
# 144.946 Para acompanhantes
# 590.596 Para trabalhadores

# Total de acadêmicos/estagiários em 2023: 1.447
# Total de residentes externos em 2023: 321
    
    # Assign revenue at discharge (based on patient complexity)
    def calculate_revenue():
        """Revenue varies by patient complexity"""
        return random.uniform(1200, 1300)*0.17
    
    discharge_ah.assign_attributes(revenue=calculate_revenue)            
    discharge_text.assign_attributes(revenue=calculate_revenue)
    discharge_death.assign_attributes(revenue=calculate_revenue)    
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
        duration=365*DAYS,
        warm_up_period=30*DAYS,        
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
        until=36*DAYS,
        warm_up_period=3*DAYS
    )

    # Access results
    df = replication_framework.get_results_dataframe()
    print(df.describe())
# ================================================================
    

# # ================================================================
# # Factorial Analysis
# # ================================================================
# def factorial_analysis():
#     """Example of factorial analysis with hospital simulation."""

#     HOURS = 60  # Time conversion factor (base time: minutes)
#     DAYS = 1440
#     YEARS = 525600
    
#     # Define simulation function wrapper
#     def simulation_wrapper(arrival_rate=6, num_psBeds=175, 
#                                     seed=None, until=None, warm_up_period=0, verbose=False, **kwargs):
#         """Wrapper that adapts parameters for factorial analysis."""

#         # ############################################################
#         # # O modelo de simulação é importado aqui
#         # ############################################################
        
#         # This would need to be modified in your actual model to accept these parameters
#         # For now, this is a template showing how to structure it
#         model = build_model(verbose=False)
#         model.run_simulation(validate_resources=False, until=until, seed=seed, warm_up_period=warm_up_period)
#         return model
    
#     # Create factorial analysis
#     factorial = FactorialExperiment(
#         simulation_function=simulation_wrapper,
#         base_seed=12345
#     )
    
#     # Add factors
#     factorial.add_factor(
#         factor_name='arrival_rate',
#         parameter_path='CreateBlock.inter_arrival_time',
#         levels=[4, 8, 20],  # Minutes between arrivals
#         description='Intervalo entre chegadas de pacientes (min)'
#     )
    
#     # factorial.add_factor(
#     #     factor_name='num_physicians',
#     #     parameter_path='Resource.physicians.capacity',
#     #     levels=[1, 5, 10],
#     #     description='Número de médicos'
#     # )

#     factorial.add_factor(
#         factor_name='num_psBeds',
#         parameter_path='Resource.psBeds.capacity',
#         levels=[100, 175, 250],
#         description='Número de leitos PS'
#     )
    
#     # factorial.add_factor(
#     #     factor_name='num_nurses',
#     #     parameter_path='Resource.nurses.capacity',
#     #     levels=[8, 10, 12],
#     #     description='Número de enfermeiros'
#     # )
    
#     # Run experiment
#     factorial.run_factorial_experiment(
#         n_replications=5,
#         # simulation_time=365*DAYS,  # 40 hours
#         # warm_up_period=30*DAYS,    # 7 hours
#         simulation_time=36*DAYS,  # 40 hours
#         warm_up_period=3*DAYS,    # 7 hours
#         verbose=True
#     )
    
#     # Analyze results
#     factorial.print_summary()
#     factorial.plot_correlation_matrix()
#     factorial.plot_main_effects('system_time_avg')
#     factorial.plot_interaction_effects('system_time_avg', 'arrival_rate', 'num_psBeds')
    
#     # Export
#     factorial.export_results()

#     print("\n\nFactorial analysis examples completed!")
#     print("Check the generated CSV files and plots for detailed results.")
    
#     return factorial
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
    print("Building HRTN model...")

    # Create configuration
    config = SimulationConfig(
        # warm_up_period=0
        # until=20
        # duration=24*HOURS,
        # warm_up_period=2*HOURS,
        duration=365*DAYS,
        warm_up_period=31*DAYS,        
        seed=321,
        check_stability=True
    )
    config.validate()
    
    # model = build_model(event_logger)
    # model = build_model(config.duration, event_logger, verbose=True)
    model = build_model(config.duration, event_logger, verbose=False)
    
    
    # --- NEW: create the builder right after the model exists ---
    # ===================================================================
    builder = MasterReportBuilder(model, run_name="hrtn")
    # ===================================================================
    

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
    # print("FILTER: Journey of PS_Patient_1")
    # print("="*80)    
    # pause_simulation()
    # model.trace_entity('PS_Patient_1')        
    
    # # # ========================================
    # # # Replay with filters
    # # # ========================================
    # # print("\n" + "="*80)
    # # print("FILTER: Replay - First 3 patients only")
    # # print("="*80)    
    # # pause_simulation()
    # # model.replay_trace(entity_pattern = r'^Patient_[1-3]$')
    
    # # ========================================
    # # Trace specific resource
    # # ========================================
    # print("\n" + "="*80)
    # print("FILTER: Replay - PS beds interactions only")
    # print("="*80)    
    # pause_simulation()
    # model.replay_trace(resource_filter={'psBeds'})
    
    # # # ========================================
    # # # Trace specific event types
    # # # ========================================
    # # print("\n" + "="*80)
    # # print("FILTER: Replay - Queue and service events only")
    # # print("="*80)    
    # # pause_simulation()
    # # model.replay_trace(event_type_filter={'queue', 'service_start', 'service_end'})
    
    # # # ========================================
    # # # Trace time window
    # # # ========================================
    # # print("\n" + "="*80)
    # # print("FILTER: Replay - Events between t=20 and t=40")
    # # print("="*80)    
    # # pause_simulation()
    # # model.replay_trace(time_range=(20, 40))
    
    # # # ========================================
    # # # Combined filters
    # # # ========================================
    # # print("\n" + "="*80)
    # # print("FILTER: Replay - Patient_1 at physicians (queue + service)")
    # # print("="*80)    
    # # pause_simulation()
    # # model.replay_trace(
    # #     entity_filter={'Patient_1'},
    # #     resource_filter={'physicians'},
    # #     event_type_filter={'queue', 'service_start', 'service_end'}
    # # )    

    # # # ========================================
    # # # Multiple patient journeys
    # # # ========================================
    # # print("\n" + "="*80)
    # # print("FILTER: Detailed journeys of first 3 patients")
    # # print("="*80)    
    # # pause_simulation()
    # # model.trace_entities(['Patient_1', 'Patient_2', 'Patient_3'])
    
    # ========================================
    # Trace statistics
    # ========================================
    model.print_trace_statistics()
    # pause_simulation()

        
    print("\n" + "="*60)
    print("SIMULATION COMPLETE - ANALYZING RESULTS")
    print("="*60)
    
    # 2. Detailed reporting
    reporter = SimulationReporter(model)
    # reporter.print_results()
    
    # 3. Warm-up analysis
    print("\nAnalyzing warm-up period...")
    warmup_analyzer = WarmUpAnalyzer(model)
    # warmup_analyzer.analyze_warm_up_period()

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
    
    # 4. Plotting
    print("\nPlotting resourse use over time...")
    plotter = SimulationPlotter(model)
    
    # Plot resource utilization over time
    plotter.plot_resource_use_over_time(show_warm_up=True, resource='psBeds', moving_average_window=50)
    plotter.plot_resource_use_over_time(show_warm_up=True, resource='CC_Rooms', moving_average_window=50)
    plotter.plot_resource_use_over_time(show_warm_up=True, resource='srpaBeds', moving_average_window=50)
    plotter.plot_resource_use_over_time(show_warm_up=True, resource='clicBeds', moving_average_window=50)
    plotter.plot_resource_use_over_time(show_warm_up=True, resource='climBeds', moving_average_window=50)
    plotter.plot_resource_use_over_time(show_warm_up=True, resource='ctiBeds', moving_average_window=50)
    plotter.plot_resource_use_over_time(show_warm_up=True, resource='matBeds', moving_average_window=50)
    plotter.plot_wip_over_time()
    plotter.plot_system_time_distribution()

    # Plot activity metrics
    print("\nPlotting activity metrics...")
    # reporter._print_activity_metrics()
    plotter.plot_activity_metrics()

        
    # Plot resource utilization summary
    print("\nPlotting resourse summary...")
    plotter.plot_resources_utilization()
    # reporter._print_resource_metrics()
    # reporter._print_entity_counts()
    # reporter._print_block_statistics()

    
    # Financial analysis
    # print("\nPlotting financial analysys...")
    # financial_analyzer = FinancialAnalyzer(model)
    # financial_analyzer.print_financial_summary()
    financial_analyzer.plot_financial_breakdown()

    # 5. Export event log
    print("\nExporting event log...")
    df = event_logger.export_to_csv("results/hrtn_event_log.csv")
    print(f"\nFirst 10 events:")
    print(df.head(10))
    
    # 6. Direct metrics access (if needed)
    metrics = MetricsCollector(model)
    entity_metrics = metrics.get_entity_metrics_summary()
    resource_metrics = metrics.get_resource_metrics_summary()
    
    print(f"\nAverage system time: {entity_metrics['tempo_medio_sistema']:.2f} min")
    # print(f"Nurses utilization: "
    #       f"{resource_metrics['nurses']['taxa_utilizacao']:.1%}")
    print(f"Random seed for this run: {config.seed}")
    
    return model, event_logger



# ===========================================
# Simulation Kit
# ===========================================
def run_single_replication():
    return main()


def run_replications_cli():
    run_replications()


# def run_factorial_cli():
#     return factorial_analysis()


def run_visualization_cli(simulation_time=5*DAYS):
    return run_visualization(build_model, simulation_time=simulation_time)
# ===========================================