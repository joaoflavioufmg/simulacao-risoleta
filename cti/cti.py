# =====================================================================
# FILE: cti.py
# =====================================================================
import random
import math
import sys
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
# desk-sim -m cti/cti.py --mode visualization
# desk-sim -m cti/cti.py --mode single
# desk-sim -m cti/cti.py --mode replications
# desk-sim -m cti/cti.py --mode factorial

# ================================================================
HOURS = 60  # Time conversion factor (base time: Minutos)
DAYS = 1440
YEARS = 525600
# ================================================================

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

    # Unidade básica para todos os tempos: minutos
    def distribution(tipo):
        arrival_rate=3.898943/DAYS # por minuto  λ (lambda): 3.898943 patients/day
        return {            
            'arrival': random.expovariate(arrival_rate),
            'cti_monitoring_ps': random.gammavariate(alpha=1.194434, beta=6.789006)*DAYS, # Pronto-Socorro (39.78%)
            'cti_monitoring_cc': random.lognormvariate(mu=math.log(3.136917), sigma = 1.373309)*DAYS, # Centro-Cirurgico (36.04%)
            'cti_monitoring_clim': random.gammavariate(alpha=1.057302, beta=7.952208)*DAYS, # Clinica-Medica (9.00%)
            'cti_monitoring_clic': random.gammavariate(alpha=0.806172, beta=7.535330)*DAYS, # Clinica-Cirurgica (7.86%)
            # 'cti_monitoring_cti': random.gammavariate(alpha=0.884660, beta=8.718526)*DAYS, # CTI (6.08%)
            'cti_monitoring_mat': random.gammavariate(alpha=1.667830, beta=1.245742)*DAYS, # Maternidade (1.23%)
            'cti_prepare': random.uniform(30, 60)
        }.get(tipo,0.0)    
    
    
    # ctiBeds = model.add_resource("ctiBeds", 31, "regular") 
    # ctiBeds = model.add_resource("ctiBeds", 35, "regular") 
    # ctiBeds = model.add_resource("ctiBeds", 40, "regular") 
    ctiBeds = model.add_resource("ctiBeds", 50, "regular") 
    # nurses = model.add_resource("nurses", 4, "regular") 
    # nursingTech = model.add_resource("nursingTech", 18, "regular") 
    # physicians = model.add_resource("physicians", 4, "regular") 
    # eMulti = model.add_resource("eMulti", 2, "regular") 
    
    # ============================ ACTIVITIES ====================
    # Create block
    arrivals = CreateBlock(
        "Chegadas", model.env,
        # inter_arrival_time=lambda: random.expovariate(1/4),
        inter_arrival_time=lambda: distribution('arrival'),
        entity_prefix="Patient",
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
    cti_monitoring_ps = ProcessBlock(
        "CTI (PS)", model.env,
        resource=ctiBeds,        
        delay_time=lambda: distribution('cti_monitoring_ps'),
        resource_units=1,                 # 1 ctiBed per service
        event_logger=event_logger
    )
    cti_monitoring_ps.set_resource_name('ctiBeds')

    # ProcessBlock block: Process with ONE resource
    cti_monitoring_cc = ProcessBlock(
        "CTI (CC)", model.env,
        resource=ctiBeds,        
        delay_time=lambda: distribution('cti_monitoring_cc'),
        resource_units=1,                 # 1 ctiBed per service
        event_logger=event_logger
    )
    cti_monitoring_cc.set_resource_name('ctiBeds')    

    # ProcessBlock block: Process with ONE resource
    cti_monitoring_clim = ProcessBlock(
        "CTI (CLIM)", model.env,
        resource=ctiBeds,        
        delay_time=lambda: distribution('cti_monitoring_clim'),
        resource_units=1,                 # 1 ctiBed per service
        event_logger=event_logger
    )
    cti_monitoring_clim.set_resource_name('ctiBeds')    

    # ProcessBlock block: Process with ONE resource
    cti_monitoring_clic = ProcessBlock(
        "CTI (CLIC)", model.env,
        resource=ctiBeds,        
        delay_time=lambda: distribution('cti_monitoring_clic'),
        resource_units=1,                 # 1 ctiBed per service
        event_logger=event_logger
    )
    cti_monitoring_clic.set_resource_name('ctiBeds')    

    # # ProcessBlock block: Process with ONE resource
    # cti_monitoring_cti = ProcessBlock(
    #     "CTI (CTI)", model.env,
    #     resource=ctiBeds,        
    #     delay_time=lambda: distribution('cti_monitoring_cti'),
    #     resource_units=1,                 # 1 ctiBed per service
    #     event_logger=event_logger
    # )
    # cti_monitoring_cti.set_resource_name('ctiBeds')    

    # ProcessBlock block: Process with ONE resource
    cti_monitoring_mat = ProcessBlock(
        "CTI (MAT)", model.env,
        resource=ctiBeds,        
        delay_time=lambda: distribution('cti_monitoring_mat'),
        resource_units=1,                 # 1 ctiBed per service
        event_logger=event_logger
    )
    cti_monitoring_mat.set_resource_name('ctiBeds') 

    # ProcessBlock block: Process with NO resource
    cti_prepare = ProcessBlock(
        "CTI prepara Encaminhamento", model.env,        
        delay_time=lambda: distribution('cti_prepare'),        
        event_logger=event_logger
    )    
    # ============================================================
    
    # ============================ DECISIONS ====================
    admission_decision = DecideBlock(
        "Origem", model.env,
        decision_type="probability",
        event_logger=event_logger
    )

    discharge_decision = DecideBlock(
        "Encaminha", model.env,
        decision_type="probability",
        event_logger=event_logger
    )
    # ============================================================

    
    # ============================ DISPOSALS ====================            
    # discharge_clim = DisposeBlock("Discharge_CLIM", model.env, event_logger=event_logger)    
    # discharge_clic = DisposeBlock("Discharge_CLIC", model.env, event_logger=event_logger)    
    # discharge_death = DisposeBlock("Discharge_Death", model.env, event_logger=event_logger)
    # discharge_text = DisposeBlock("Discharge_Transf_Ext", model.env, event_logger=event_logger)    
    # discharge_cc = DisposeBlock("Discharge_CC", model.env, event_logger=event_logger)    
    # discharge_mat = DisposeBlock("Discharge_MAT", model.env, event_logger=event_logger)    
    # discharge_ps = DisposeBlock("Discharge_PS", model.env, event_logger=event_logger)        
    # discharge_ah = DisposeBlock("DischargeAltaHosp", model.env, event_logger=event_logger)

    discharge_clim = DisposeBlock("Saida_CLIM", model.env, event_logger=event_logger)    
    discharge_clic = DisposeBlock("Saida_CLIC", model.env, event_logger=event_logger)    
    discharge_death = DisposeBlock("Saida_Death", model.env, event_logger=event_logger)
    discharge_text = DisposeBlock("Saida_Transf_Ext", model.env, event_logger=event_logger)    
    discharge_cc = DisposeBlock("Saida_CC", model.env, event_logger=event_logger)    
    discharge_mat = DisposeBlock("Saida_MAT", model.env, event_logger=event_logger)    
    discharge_ps = DisposeBlock("Saida_PS", model.env, event_logger=event_logger)        
    discharge_ah = DisposeBlock("Saida_AltaHosp", model.env, event_logger=event_logger)
    # ============================================================


    # ============================ INCLUDE ALL BLOCKS ====================    
    # Add blocks to model
    for block in [arrivals, cti_monitoring_ps, cti_monitoring_cc, cti_monitoring_clim,
                  cti_monitoring_clic, cti_monitoring_mat, 
                  cti_prepare, admission_decision, discharge_decision,
                  discharge_clim, discharge_clic, discharge_death, discharge_text,
                  discharge_cc, discharge_mat, discharge_ps, discharge_ah]:
        model.add_block(block)
    # ====================================================================
    
    # ============================ CONNECT ALL BLOCKS ====================    
    # Connect flow
    arrivals.connect_to(admission_decision)

    # admission_decision.add_route("from_PS", cti_monitoring_ps, probability=0.3978)
    # admission_decision.add_route("from_CC", cti_monitoring_cc, probability=0.3604)
    # admission_decision.add_route("from_CLIM", cti_monitoring_clim, probability=0.0900)
    # admission_decision.add_route("from_CLIC", cti_monitoring_clic, probability=0.0786)
    # admission_decision.add_route("from_CTI", cti_monitoring_cti, probability=0.0608)
    # admission_decision.add_route("from_MAT", cti_monitoring_mat, probability=0.0123)

    admission_decision.add_route("do Pronto Socorro", cti_monitoring_ps, probability=0.4501)
    admission_decision.add_route("do Centro Cirurgico", cti_monitoring_cc, probability=0.3906)
    admission_decision.add_route("da Clinica Medica", cti_monitoring_clim, probability=0.0745)
    admission_decision.add_route("da Clinica Cirurgica", cti_monitoring_clic, probability=0.0721)
    # admission_decision.add_route("do CTI", cti_monitoring_cti, probability=0.0608)
    admission_decision.add_route("da Maternidade", cti_monitoring_mat, probability=0.0127)

    cti_monitoring_ps.connect_to(cti_prepare)
    cti_monitoring_cc.connect_to(cti_prepare)
    cti_monitoring_clim.connect_to(cti_prepare)
    cti_monitoring_clic.connect_to(cti_prepare)
    # cti_monitoring_cti.connect_to(cti_prepare)
    cti_monitoring_mat.connect_to(cti_prepare)
    
    cti_prepare.connect_to(discharge_decision)

    # discharge_decision.add_route("to_CLIM", discharge_clim, probability=0.3685)
    # discharge_decision.add_route("to_CLIC", discharge_clic, probability=0.3253)
    # discharge_decision.add_route("to_DEATH", discharge_death, probability=0.1713)
    # discharge_decision.add_route("to_CTI", cti_monitoring_cti, probability=0.0614)
    # discharge_decision.add_route("to_EXT", discharge_text, probability=0.0458)
    # discharge_decision.add_route("to_CC", discharge_cc, probability=0.0117)
    # discharge_decision.add_route("to_MAT", discharge_mat, probability=0.0101)
    # discharge_decision.add_route("to_PS", discharge_ps, probability=0.0045)
    # discharge_decision.add_route("to_AH", discharge_ah, probability=0.0013)

    discharge_decision.add_route("para a Clinica Medica", discharge_clim, probability=0.3906)
    discharge_decision.add_route("para a Clinica Cirurgica", discharge_clic, probability=0.3344)
    discharge_decision.add_route("Obito", discharge_death, probability=0.2032)
    # discharge_decision.add_route("para o CTI", cti_monitoring_cti, probability=0.0614)
    discharge_decision.add_route("Transferencia Externa", discharge_text, probability=0.0517)
    discharge_decision.add_route("para a Maternidade", discharge_mat, probability=0.0102)
    discharge_decision.add_route("para o Centro Cirurgico", discharge_cc, probability=0.0067)    
    discharge_decision.add_route("para o Pronto Socorro", discharge_ps, probability=0.0018)
    discharge_decision.add_route("Alta Hospitalar", discharge_ah, probability=0.0014)
    
    
    # ================================================================
    # CONFIGURE FINANCIAL ATTRIBUTES
    # ================================================================    
    # Assign costs to each activity
    arrivals.assign_attributes(
        cost=lambda: random.uniform(0, 0)  # Triage costs $20-30
    )
    
    cti_monitoring_ps.assign_attributes(
        cost=lambda: random.uniform(3000, 3300)  # Consultation costs $100-200
    )
    
    cti_monitoring_cc.assign_attributes(
        cost=lambda: random.uniform(3000, 3300)  # Medication costs $100-200
    )

    cti_monitoring_clim.assign_attributes(
        cost=lambda: random.uniform(3000, 3300)  # Medication costs $100-200
    )
    
    cti_monitoring_clic.assign_attributes(
        cost=lambda: random.uniform(3000, 3300)  # Medication costs $100-200
    )

    # cti_monitoring_cti.assign_attributes(
    #     cost=lambda: random.uniform(15000, 30000)  # Medication costs $100-200
    # )

    cti_monitoring_mat.assign_attributes(
        cost=lambda: random.uniform(3000, 3300)  # Medication costs $100-200
    )
    
    # Assign revenue at discharge (based on patient complexity)
    def calculate_revenue():
        """Revenue varies by patient complexity"""
        return random.uniform(1200, 1300)
    
    discharge_clim.assign_attributes(revenue=calculate_revenue)
    discharge_clic.assign_attributes(revenue=calculate_revenue)
    discharge_death.assign_attributes(revenue=calculate_revenue)
    discharge_text.assign_attributes(revenue=calculate_revenue)
    discharge_cc.assign_attributes(revenue=calculate_revenue)
    discharge_mat.assign_attributes(revenue=calculate_revenue)
    discharge_ps.assign_attributes(revenue=calculate_revenue)
    discharge_ah.assign_attributes(revenue=calculate_revenue)    
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
        duration=24*HOURS,
        warm_up_period=2*HOURS,        
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
        until=365*DAYS,
        warm_up_period=30*DAYS
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
    def simulation_wrapper(arrival_rate=360, num_ctiBeds=31, 
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
        levels=[360, 300, 420],  # Minutes between arrivals
        description='Intervalo entre chegadas de pacientes (min)'
    )
    
    # factorial.add_factor(
    #     factor_name='num_physicians',
    #     parameter_path='Resource.physicians.capacity',
    #     levels=[1, 5, 10],
    #     description='Número de médicos'
    # )

    factorial.add_factor(
        factor_name='num_ctiBeds',
        parameter_path='Resource.ctiBeds.capacity',
        levels=[25, 31, 45],
        description='Número de leitos CTI'
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
        simulation_time=365*DAYS,  # 40 hours
        warm_up_period=30*DAYS,    # 7 hours
        verbose=True
    )
    
    # Analyze results
    factorial.print_summary()
    factorial.plot_correlation_matrix()
    factorial.plot_main_effects('system_time_avg')
    factorial.plot_interaction_effects('system_time_avg', 'arrival_rate', 'num_ctiBeds')
    
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
    print("Building CTI model...")

    # Create configuration
    config = SimulationConfig(
        # warm_up_period=0
        # until=20
        # duration=24*HOURS,
        # warm_up_period=2*HOURS,
        duration=365*DAYS,
        warm_up_period=30*DAYS,        
        seed=321,
        check_stability=True
    )
    config.validate()
    
    # model = build_model(event_logger)
    # model = build_model(config.duration, event_logger, verbose=True)
    model = build_model(config.duration, event_logger, verbose=False)

    
    # --- NEW: create the builder right after the model exists ---
    # ===================================================================
    builder = MasterReportBuilder(model, run_name="cti")
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
    
    
    # === ANALYSIS PHASE (using separate modules) ===
    #     
    # ========================================
    # Trace specific patient
    # ========================================    
    # print("\n" + "="*80)
    # print("FILTER: Journey of Patient_1")
    # print("="*80)    
    # pause_simulation()
    # model.trace_entity('Patient_1')        
    
    # # ========================================
    # # Replay with filters
    # # ========================================
    # print("\n" + "="*80)
    # print("FILTER: Replay - First 3 patients only")
    # print("="*80)    
    # pause_simulation()
    # model.replay_trace(entity_pattern = r'^Patient_[1-3]$')
    
    # ========================================
    # Trace specific resource
    # ========================================
    # print("\n" + "="*80)
    # print("FILTER: Replay - CTI beds interactions only")
    # print("="*80)    
    # pause_simulation()
    # model.replay_trace(resource_filter={'ctiBeds'})
    
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
    
    # 4. Plotting
    print("\nPlotting resourse use over time...")
    plotter = SimulationPlotter(model)
    
    # Plot resource utilization over time
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='nursingTech', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='nurses', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='physicians', moving_average_window=50)
    plotter.plot_resource_use_over_time(show_warm_up=True, resource='ctiBeds', moving_average_window=50)
    # plotter.plot_resource_use_over_time(show_warm_up=True, resource='eMulti', moving_average_window=50)
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
    df = event_logger.export_to_csv("results/cti_event_log.csv")
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


def run_factorial_cli():
    return factorial_analysis()


def run_visualization_cli(simulation_time=30*DAYS):
    return run_visualization(build_model, simulation_time=simulation_time)
# ===========================================