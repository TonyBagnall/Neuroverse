"""Neuroverse: benchmark archive and leaderboards for neural time series classification.

Neuroverse is the neural companion to the Multiverse archive for multivariate
time series classification (https://github.com/aeon-toolkit/multiverse) and
stands alone: it does not depend on the Multiverse package.
"""

from __future__ import annotations

__version__ = "0.0.1"

from typing import TYPE_CHECKING

from neuroverse.datasets import (
    by_type,
    metadata,
    neuroverse,
    neuroverse_90,
    neuroverse_114,
    neuroverse_open,
    neuroverse_paper,
    neuroverse_study,
)
from neuroverse.estimators import (
    BAKEOFF_ESTIMATORS,
    CANONICAL_ESTIMATORS,
    HEADLINE_ESTIMATORS,
    PAPER_ESTIMATORS,
    estimator_info,
    estimator_table,
)

if TYPE_CHECKING:
    from neuroverse.results import (
        download_results,
        interactive_download,
        list_result_options,
        load_results,
    )

_LAZY_RESULTS = {
    "load_results",
    "download_results",
    "interactive_download",
    "list_result_options",
}


def __getattr__(name: str):
    if name in _LAZY_RESULTS:
        from neuroverse import results as _results_mod

        return getattr(_results_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "neuroverse",
    "neuroverse_study",
    "neuroverse_90",
    "neuroverse_114",
    "neuroverse_open",
    "neuroverse_paper",
    "by_type",
    "metadata",
    "HEADLINE_ESTIMATORS",
    "CANONICAL_ESTIMATORS",
    "BAKEOFF_ESTIMATORS",
    "PAPER_ESTIMATORS",
    "estimator_info",
    "estimator_table",
    "load_results",
    "download_results",
    "interactive_download",
    "list_result_options",
]
