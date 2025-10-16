"""
Astro AMASE - Automated Molecular Assignment and Source Parameter Estimation

A comprehensive package for automated molecular line identification in 
radio astronomical observations.
"""

# Register molsim compatibility for pickle files
import sys
from .utils import molsim
sys.modules['molsim'] = molsim

# Now continue with regular imports
from .version import __version__
from .main import (
    assign_observations,
    run_pipeline,
    get_linewidth,
    get_source_parameters
)
from .constants import (
    DEFAULT_VALID_ATOMS,
    ALL_VALID_ATOMS,
    ckm,
    c,
    global_thresh
)

__all__ = [
    '__version__',
    'assign_observations',
    'run_pipeline',
    'get_linewidth',
    'get_source_parameters',
    'DEFAULT_VALID_ATOMS',
    'ALL_VALID_ATOMS',
    'ckm',
    'c',
    'global_thresh'
]