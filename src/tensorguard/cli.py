"""
TensorGuard CLI
Main entry point for the TensorGuard Unified Trust Fabric.
"""

import click
import logging
import sys
import importlib.util
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get the project root directory."""
    # cli.py is at src/tensorguard/cli.py, so project root is 3 levels up
    return Path(__file__).resolve().parent.parent.parent


def _load_script_module(script_name: str):
    """
    Dynamically load a module from the scripts directory.

    Args:
        script_name: Name of the script file (without .py extension)

    Returns:
        Loaded module object
    """
    project_root = _get_project_root()
    script_path = project_root / "scripts" / f"{script_name}.py"

    if not script_path.exists():
        raise ImportError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)

    # Add src to path temporarily for script imports
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    spec.loader.exec_module(module)
    return module


@click.group()
def cli():
    """TensorGuard Trust Fabric CLI."""
    pass


# === DAEMON ===
@cli.command()
def agent():
    """Start the Unified Edge Daemon (Identity, Network, ML)."""
    from .agent.daemon import main as agent_main
    agent_main()


# === PLATFORM ===
@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to listen on")
def server(host, port):
    """Start the Control Plane Server (Platform + UI)."""
    import uvicorn
    uvicorn.run("tensorguard.platform.main:app", host=host, port=port, reload=False)


# === ROBOTICS TOOLS ===
@cli.command()
@click.argument("input_path")
@click.option("--output-dir", default="./dataset", help="Output directory")
def ingest(input_path, output_dir):
    """Ingest a rosbag2 or MCAP file."""
    # Load the script dynamically using robust path resolution
    ingest_module = _load_script_module("tgflow_ros2_ingest")

    # Set sys.argv for the script's argument parser
    sys.argv = ["tgflow_ros2_ingest.py", input_path, "--output-dir", output_dir]
    ingest_module.main()


# === SECURE PACKAGE (TGSP) ===
@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("subcommand_args", nargs=-1, type=click.UNPROCESSED)
def pkg(subcommand_args):
    """TGSP Package Management (Create, Verify, Decrypt)."""
    from .tgsp.cli import main as tgsp_main
    sys.argv = ["tgsp"] + list(subcommand_args)
    tgsp_main()


# === BENCHMARK ===
@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("subcommand_args", nargs=-1, type=click.UNPROCESSED)
def bench(subcommand_args):
    """Run TensorGuard Benchmarks."""
    from .bench.cli import main as bench_main
    sys.argv = ["bench"] + list(subcommand_args)
    bench_main()


# === TG-TINKER (Privacy-First Training API) ===
@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("subcommand_args", nargs=-1, type=click.UNPROCESSED)
def tinker(subcommand_args):
    """TG-Tinker Privacy-First Training API commands."""
    from .platform.tg_tinker_api.cli import main as tinker_main
    sys.argv = ["tinker"] + list(subcommand_args)
    tinker_main()


# === PEFT STUDIO ===
@cli.group()
def peft():
    """PEFT (Parameter-Efficient Fine-Tuning) Studio & Orchestration."""
    pass


@peft.command("ui")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to listen on")
def peft_ui(host, port):
    """Launch the PEFT Studio UI."""
    import uvicorn
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open(f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}/#peft-studio")

    Timer(1.5, open_browser).start()
    uvicorn.run("tensorguard.platform.main:app", host=host, port=port, reload=False)


@peft.command("run")
@click.argument("config_path", type=click.Path(exists=True))
def peft_run(config_path):
    """Run a PEFT workflow from a JSON/YAML configuration file."""
    import json
    import asyncio
    from .integrations.peft_hub.workflow import PeftWorkflow
    from .integrations.peft_hub.schemas import PeftRunConfig

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    config = PeftRunConfig(**config_data)
    workflow = PeftWorkflow(config)

    click.echo(f"Starting PEFT Run [Simulation={config.simulation}]...")
    
    async def _run():
        async for log in workflow.execute():
            click.echo(log)

    asyncio.run(_run())
    click.echo("PEFT Run Completed.")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
