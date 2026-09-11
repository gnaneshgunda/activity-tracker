"""Interface layer.

This package groups the routing and response interfaces that decide which task
logic should answer a user query.
"""

from analysis.router import DEFAULT_RULES, Route, RouteResult, build_router, route

try:
    from analysis.store import ActivityStore, TimelineRow
except ImportError:  # pragma: no cover
    ActivityStore = TimelineRow = None

__all__ = [
    "DEFAULT_RULES",
    "Route",
    "RouteResult",
    "ActivityStore",
    "TimelineRow",
    "build_router",
    "route",
]
