"""Dashboard component builders for the sentiment trader pipeline.

Provides reusable Plotly-based dashboard components for interactive
visualisation of sentiment trends, trader performance, and correlations.
"""

from typing import Any


def build_empty_dashboard(message: str = "No data available.") -> dict[str, Any]:
    """Build a minimal placeholder dashboard structure.

    Args:
        message: Placeholder message to display (default "No data available.").

    Returns:
        A dict with ``title`` and ``message`` keys for dashboard rendering.
    """
    return {"title": "Dashboard", "message": message}
