"""
TG-Tinker CLI commands.

Provides CLI interface for the TG-Tinker training API.
"""

import json
import os
import sys
from datetime import datetime
from typing import Optional

import click


def get_client():
    """Get a ServiceClient instance."""
    try:
        from tg_tinker import ServiceClient

        return ServiceClient()
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo(
            "Set TG_TINKER_API_KEY environment variable or use 'tensorguard tinker auth'",
            err=True,
        )
        sys.exit(1)


@click.group()
def tinker():
    """TG-Tinker Privacy-First Training API commands."""
    pass


# ==============================================================================
# Auth Commands
# ==============================================================================


@tinker.command("auth")
@click.option("--api-key", prompt=True, hide_input=True, help="API key")
@click.option("--base-url", default=None, help="Base URL for TG-Tinker API")
def auth(api_key: str, base_url: Optional[str]):
    """
    Configure authentication for TG-Tinker API.

    Sets environment variables for API key and base URL.
    """
    # Validate the key by attempting a connection
    try:
        from tg_tinker import ServiceClient

        client = ServiceClient(
            api_key=api_key,
            base_url=base_url or "http://localhost:8080",
        )
        click.echo("Authentication successful!")

        # Print instructions for setting env vars
        click.echo("\nTo persist credentials, add to your shell profile:")
        click.echo(f'  export TG_TINKER_API_KEY="{api_key}"')
        if base_url:
            click.echo(f'  export TG_TINKER_BASE_URL="{base_url}"')

    except Exception as e:
        click.echo(f"Authentication failed: {e}", err=True)
        sys.exit(1)


# ==============================================================================
# Training Client Commands
# ==============================================================================


@tinker.command("create-client")
@click.option("--model", "-m", required=True, help="Model reference (e.g., meta-llama/Llama-3-8B)")
@click.option("--lora-rank", default=16, help="LoRA rank")
@click.option("--lora-alpha", default=32.0, help="LoRA alpha")
@click.option("--learning-rate", "-lr", default=1e-4, help="Learning rate")
@click.option("--dp/--no-dp", default=False, help="Enable differential privacy")
@click.option("--dp-epsilon", default=8.0, help="Target DP epsilon")
@click.option("--dp-noise", default=1.0, help="DP noise multiplier")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def create_client(
    model: str,
    lora_rank: int,
    lora_alpha: float,
    learning_rate: float,
    dp: bool,
    dp_epsilon: float,
    dp_noise: float,
    output_json: bool,
):
    """
    Create a new training client.

    Example:
        tensorguard tinker create-client -m meta-llama/Llama-3-8B --dp
    """
    from tg_tinker import (
        DPConfig,
        LoRAConfig,
        OptimizerConfig,
        ServiceClient,
        TrainingConfig,
    )

    client = get_client()

    # Build configuration
    lora_config = LoRAConfig(rank=lora_rank, alpha=lora_alpha)
    optimizer_config = OptimizerConfig(learning_rate=learning_rate)

    dp_config = None
    if dp:
        dp_config = DPConfig(
            enabled=True,
            noise_multiplier=dp_noise,
            target_epsilon=dp_epsilon,
        )

    config = TrainingConfig(
        model_ref=model,
        lora_config=lora_config,
        optimizer=optimizer_config,
        dp_config=dp_config,
    )

    try:
        tc = client.create_training_client(config)

        if output_json:
            click.echo(
                json.dumps(
                    {
                        "training_client_id": tc.id,
                        "model_ref": config.model_ref,
                        "status": tc.status.value,
                        "step": tc.step,
                        "dp_enabled": tc.dp_enabled,
                    },
                    indent=2,
                )
            )
        else:
            click.echo(f"Created training client: {tc.id}")
            click.echo(f"  Model: {config.model_ref}")
            click.echo(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}")
            click.echo(f"  Learning rate: {learning_rate}")
            click.echo(f"  DP enabled: {tc.dp_enabled}")
            if tc.dp_enabled:
                click.echo(f"    Target epsilon: {dp_epsilon}")
                click.echo(f"    Noise multiplier: {dp_noise}")

    except Exception as e:
        click.echo(f"Error creating training client: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


@tinker.command("list-clients")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def list_clients(output_json: bool):
    """List all training clients."""
    client = get_client()

    try:
        clients = client.list_training_clients()

        if output_json:
            click.echo(
                json.dumps(
                    [
                        {
                            "training_client_id": tc.training_client_id,
                            "model_ref": tc.model_ref,
                            "status": tc.status.value,
                            "step": tc.step,
                            "created_at": tc.created_at.isoformat(),
                        }
                        for tc in clients
                    ],
                    indent=2,
                )
            )
        else:
            if not clients:
                click.echo("No training clients found.")
                return

            click.echo(f"{'ID':<40} {'Model':<30} {'Status':<10} {'Step':<6}")
            click.echo("-" * 90)
            for tc in clients:
                click.echo(
                    f"{tc.training_client_id:<40} {tc.model_ref[:28]:<30} "
                    f"{tc.status.value:<10} {tc.step:<6}"
                )

    except Exception as e:
        click.echo(f"Error listing clients: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


@tinker.command("get-client")
@click.argument("client_id")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def get_client_info(client_id: str, output_json: bool):
    """Get information about a training client."""
    client = get_client()

    try:
        tc = client.get_training_client(client_id)

        if output_json:
            data = {
                "training_client_id": tc.training_client_id,
                "tenant_id": tc.tenant_id,
                "model_ref": tc.model_ref,
                "status": tc.status.value,
                "step": tc.step,
                "created_at": tc.created_at.isoformat(),
                "config": tc.config.model_dump() if tc.config else {},
            }
            if tc.dp_metrics:
                data["dp_metrics"] = tc.dp_metrics.model_dump()
            click.echo(json.dumps(data, indent=2))
        else:
            click.echo(f"Training Client: {tc.training_client_id}")
            click.echo(f"  Model: {tc.model_ref}")
            click.echo(f"  Status: {tc.status.value}")
            click.echo(f"  Step: {tc.step}")
            click.echo(f"  Created: {tc.created_at}")
            if tc.dp_metrics:
                click.echo(f"  DP Epsilon: {tc.dp_metrics.total_epsilon:.4f}")
                click.echo(f"  DP Delta: {tc.dp_metrics.delta}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


# ==============================================================================
# Futures Commands
# ==============================================================================


@tinker.group()
def futures():
    """Manage async operation futures."""
    pass


@futures.command("ls")
@click.option("--client-id", "-c", help="Filter by training client ID")
@click.option("--status", "-s", help="Filter by status")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def futures_list(
    client_id: Optional[str],
    status: Optional[str],
    limit: int,
    output_json: bool,
):
    """List futures."""
    # Note: This would require a list futures endpoint
    # For now, show a message
    click.echo("Futures listing requires the server to expose a /v1/futures endpoint.")
    click.echo("Use 'tensorguard tinker get-future <id>' to check specific futures.")


@futures.command("get")
@click.argument("future_id")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def futures_get(future_id: str, output_json: bool):
    """Get future status."""
    client = get_client()

    try:
        future = client.get_future(future_id)

        if output_json:
            click.echo(
                json.dumps(
                    {
                        "future_id": future.future_id,
                        "status": future.status.value,
                        "operation": future.operation.value,
                        "training_client_id": future.training_client_id,
                        "created_at": future.created_at.isoformat(),
                        "started_at": future.started_at.isoformat() if future.started_at else None,
                        "completed_at": future.completed_at.isoformat() if future.completed_at else None,
                    },
                    indent=2,
                )
            )
        else:
            click.echo(f"Future: {future.future_id}")
            click.echo(f"  Status: {future.status.value}")
            click.echo(f"  Operation: {future.operation.value}")
            click.echo(f"  Training Client: {future.training_client_id}")
            click.echo(f"  Created: {future.created_at}")
            if future.started_at:
                click.echo(f"  Started: {future.started_at}")
            if future.completed_at:
                click.echo(f"  Completed: {future.completed_at}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


# ==============================================================================
# Logs Commands
# ==============================================================================


@tinker.command("logs")
@click.argument("client_id")
@click.option("--operation", "-o", help="Filter by operation type")
@click.option("--limit", "-n", default=50, help="Max results")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def logs(
    client_id: str,
    operation: Optional[str],
    limit: int,
    output_json: bool,
):
    """View audit logs for a training client."""
    client = get_client()

    try:
        entries = client.get_audit_logs(
            training_client_id=client_id,
            operation=operation,
            limit=limit,
        )

        if output_json:
            click.echo(
                json.dumps(
                    [
                        {
                            "entry_id": e.entry_id,
                            "operation": e.operation.value,
                            "started_at": e.started_at.isoformat(),
                            "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                            "duration_ms": e.duration_ms,
                            "success": e.success,
                            "request_hash": e.request_hash,
                            "record_hash": e.record_hash,
                        }
                        for e in entries
                    ],
                    indent=2,
                )
            )
        else:
            if not entries:
                click.echo("No audit log entries found.")
                return

            click.echo(f"Audit logs for {client_id}:")
            click.echo(f"{'Time':<20} {'Operation':<18} {'Success':<8} {'Duration':<10}")
            click.echo("-" * 60)
            for e in entries:
                time_str = e.started_at.strftime("%Y-%m-%d %H:%M:%S")
                dur_str = f"{e.duration_ms}ms" if e.duration_ms else "-"
                success_str = "OK" if e.success else "FAIL"
                click.echo(f"{time_str:<20} {e.operation.value:<18} {success_str:<8} {dur_str:<10}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


# ==============================================================================
# Artifacts Commands
# ==============================================================================


@tinker.group()
def artifacts():
    """Manage encrypted artifacts."""
    pass


@artifacts.command("pull")
@click.argument("artifact_id")
@click.option("--output", "-o", help="Output file path")
def artifacts_pull(artifact_id: str, output: Optional[str]):
    """Download an encrypted artifact."""
    client = get_client()

    try:
        content = client.pull_artifact(artifact_id)

        output_path = output or f"{artifact_id}.enc"
        with open(output_path, "wb") as f:
            f.write(content)

        click.echo(f"Downloaded artifact to: {output_path}")
        click.echo(f"  Size: {len(content):,} bytes")
        click.echo("  Note: Content is encrypted. Use your tenant key to decrypt.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


# ==============================================================================
# Server Command
# ==============================================================================


@tinker.command("serve")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8080, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """
    Start the TG-Tinker API server.

    Example:
        tensorguard tinker serve --port 8080
    """
    import uvicorn
    from fastapi import FastAPI

    from tensorguard.platform.tg_tinker_api import router, start_worker

    # Create app
    app = FastAPI(
        title="TG-Tinker API",
        description="Privacy-First ML Training API",
        version="1.0.0",
    )
    app.include_router(router)

    click.echo(f"Starting TG-Tinker API server at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")

    # Start background worker
    start_worker()

    uvicorn.run(app, host=host, port=port, reload=reload)


def main():
    """Main entry point for tinker CLI."""
    tinker()


if __name__ == "__main__":
    main()
