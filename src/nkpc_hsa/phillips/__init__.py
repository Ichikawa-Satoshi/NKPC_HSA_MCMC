"""Nine-cell implementation of the design in ``report/design.tex``.

The public API intentionally separates the measurement-only competition module
from the inflation modules.  This makes the primary modular cut executable and
prevents an accidental full-joint result from being reported as the headline.
"""

from .data import CELL_SPECS, DesignData, load_design_data
from .estimation import run_nine_cell_design
from .temporal import aggregation_matrix

__all__ = [
    "CELL_SPECS",
    "DesignData",
    "aggregation_matrix",
    "load_design_data",
    "run_nine_cell_design",
]
