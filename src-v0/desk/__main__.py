# desk/__main__.py

import argparse
import importlib.util
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------
def load_model_from_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------
# DESK mode registry (SINGLE SOURCE OF TRUTH)
# ---------------------------------------------------------------------
MODE_REGISTRY = {
    "single": "run_single_replication",
    "replications": "run_replications_cli",
    "factorial": "run_factorial_cli",
    "visualization": "run_visualization_cli",
}


# ---------------------------------------------------------------------
# Mode discovery
# ---------------------------------------------------------------------
def list_modes(module, model_path: Path):
    print(f"\nAvailable DESK execution modes for {model_path.name}:\n")

    found_any = False
    for mode, func_name in MODE_REGISTRY.items():
        if hasattr(module, func_name):
            print(f"  {mode:<14} → {func_name}()")
            found_any = True

    if not found_any:
        print("  (No explicit modes found)")

    print("\nDefault execution order (no --mode):")

    if hasattr(module, "run_simulation_cli"):
        print("  run_simulation_cli()")
    elif hasattr(module, "run_single_replication"):
        print("  run_single_replication()")
    elif hasattr(module, "build_model"):
        print("  build_model() → model.run_simulation()")
    else:
        print("  ❌ No valid DESK entrypoint found")

    print("")


# ---------------------------------------------------------------------
# Execution dispatcher
# ---------------------------------------------------------------------
def dispatch(module, mode: str):
    if mode:
        func_name = MODE_REGISTRY.get(mode)
        if not func_name:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes: {list(MODE_REGISTRY)}"
            )

        if not hasattr(module, func_name):
            raise RuntimeError(
                f"Model does not define `{func_name}()` required for mode '{mode}'."
            )

        return getattr(module, func_name)()

    # Default fallback chain
    if hasattr(module, "run_simulation_cli"):
        return module.run_simulation_cli()

    if hasattr(module, "run_single_replication"):
        return module.run_single_replication()

    if hasattr(module, "build_model"):
        model = module.build_model()
        model.run_simulation()
        return model

    raise RuntimeError(
        "DESK model must define one of:\n"
        "  • run_simulation_cli()\n"
        "  • run_single_replication()\n"
        "  • build_model()"
    )


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DESK – Discrete Event Simulation Kit"
    )

    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to a DESK simulation model (.py)"
    )

    parser.add_argument(
        "--mode",
        choices=MODE_REGISTRY.keys(),
        help="Execution mode"
    )

    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List available execution modes for the model and exit"
    )

    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    module = load_model_from_file(model_path)

    if args.list_modes:
        list_modes(module, model_path)
        return

    dispatch(module, args.mode)


if __name__ == "__main__":
    main()
