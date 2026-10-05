# =====================================================================
# FILE: analytics/report_builder.py
# =====================================================================
"""
Consolidates all terminal-printed simulation reports into a single,
timestamped .txt file. Output is captured silently — nothing is
printed to the console during report generation.

Design goals:
- Zero changes required to SimulationReporter, FinancialAnalyzer,
  StabilityAnalyzer, or WarmUpAnalyzer. They keep using plain print().
- Works by redirecting sys.stdout into an in-memory buffer during the
  report-generation calls only (not for the whole program), so it's
  safe to drop into an existing script with minimal disruption.
- Produces one clean, sectioned .txt file with headers separating each
  analysis block, plus a run timestamp and simulation identifier.
"""

import sys
import io
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Callable


@contextmanager
def _capture_to(buffer: io.StringIO):
    """Temporarily redirect sys.stdout into a buffer only (console silent)."""
    original_stdout = sys.stdout
    sys.stdout = buffer
    try:
        yield
    finally:
        sys.stdout = original_stdout


class MasterReportBuilder:
    """
    Orchestrates calls to existing analyzer/reporter objects, captures
    everything they print, and writes one consolidated .txt report at
    the end. Output is captured silently — the console stays clean
    while sections run.

    Usage:
        builder = MasterReportBuilder(model, run_name="checkout_model")
        builder.add_section("STABILITY ANALYSIS", stability_analyzer.check_system_stability)
        builder.add_section("WARM-UP ANALYSIS", warmup_analyzer.analyze_warm_up_period)
        builder.add_section("SIMULATION RESULTS", reporter.print_results)
        builder.add_section("FINANCIAL SUMMARY", financial_analyzer.print_financial_summary)
        builder.run_and_save(output_dir="results")
    """

    def __init__(self, model, run_name: Optional[str] = None):
        self.model = model
        self.run_name = run_name or "simulation"
        self._sections: List[tuple] = []  # (title, callable, args, kwargs)
        self._buffer = io.StringIO()

    def add_section(self, title: str, func: Callable, *args, **kwargs):
        """
        Register a report-producing call to include in the consolidated
        report. `func` is called exactly as you'd normally call it
        (e.g. reporter.print_results); its prints are captured in order.
        """
        self._sections.append((title, func, args, kwargs))
        return self  # allow chaining

    def _write_header(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bar = "#" * 70
        lines = [
            bar,
            "# CONSOLIDATED SIMULATION REPORT",
            f"# Run: {self.run_name}",
            f"# Generated: {timestamp}",
            bar,
        ]
        print("\n".join(lines))

    def _write_section_title(self, title: str):
        bar = "=" * 70
        print(f"\n{bar}\n>>> {title}\n{bar}")

    def run_and_save(self, output_dir: str = "results",
                      filename: Optional[str] = None) -> str:
        """
        Executes every registered section in order, capturing all output
        silently (nothing printed to console), and writes the consolidated
        text to a single .txt file. Returns the path to the written file.
        """
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.run_name}_report_{timestamp}.txt"

        output_path = os.path.join(output_dir, filename)

        with _capture_to(self._buffer):
            self._write_header()
            for title, func, args, kwargs in self._sections:
                self._write_section_title(title)
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"\n[ERROR] Section '{title}' failed: {e}")

            print("\n" + "#" * 70)
            print("# END OF REPORT")
            print("#" * 70)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._buffer.getvalue())

        # Console confirmation only (after capture has ended) — this is the
        # only line that ever reaches the terminal.
        print(f"\n✅ Consolidated report saved to: {output_path}")
        return output_path
