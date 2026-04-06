"""
OpenEnv-compatible server entry point for PipelineRx.

This module provides the standard ``server`` entry point expected by
``openenv validate`` and ``uv run server``.  It simply wraps the existing
FastAPI application defined in ``app.main`` and launches it with Uvicorn.
"""

from __future__ import annotations

import os


def main() -> None:
    """Start the PipelineRx FastAPI server via Uvicorn."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    main()
