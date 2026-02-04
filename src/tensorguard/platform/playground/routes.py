"""
TenSafe API Playground routes.

Serves an interactive web-based playground for testing the TenSafe API.
"""

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(tags=["playground"])

# Get the directory containing this file
PLAYGROUND_DIR = Path(__file__).parent


@router.get("/playground", response_class=HTMLResponse)
async def get_playground():
    """
    Serve the TenSafe API Playground.

    The playground provides an interactive web interface for:
    - Testing API endpoints
    - Viewing request/response examples
    - Learning the API structure

    This is similar to Swagger UI but with a more streamlined,
    purpose-built interface for TenSafe.
    """
    index_path = PLAYGROUND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    else:
        return HTMLResponse(
            content="""
            <html>
                <head><title>TenSafe Playground</title></head>
                <body>
                    <h1>Playground not found</h1>
                    <p>The playground HTML file could not be located.</p>
                    <p>Try using <a href="/docs">Swagger UI</a> instead.</p>
                </body>
            </html>
            """,
            status_code=404,
        )


@router.get("/playground/health")
async def playground_health():
    """Check if the playground is available."""
    index_path = PLAYGROUND_DIR / "index.html"
    return {
        "available": index_path.exists(),
        "path": str(index_path),
    }
