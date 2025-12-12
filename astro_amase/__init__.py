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

from .utils.plotting import show_fit_in_notebook, plot_from_saved, get_individual_plots, create_interactive_vlsr_plot
from .utils.astro_utils import find_peaks_local
from .utils.molsim_utils import load_obs, find_limits, get_rms
from .output.create_output_file import molecule_summary
from .core.molecule_prediction import molecule_prediction

__all__ = [
    '__version__',
    'assign_observations',
    'run_pipeline',
    'get_linewidth',
    'get_source_parameters',
    'show_fit_in_notebook',
    'plot_from_saved',
    'get_individual_plots',
    'DEFAULT_VALID_ATOMS',
    'ALL_VALID_ATOMS',
    'ckm',
    'c',
    'global_thresh',
    'find_peaks_local',
    'get_rms',
    'load_obs',
    'find_limits',
    'molecule_prediction',
    'create_interactive_vlsr_plot',
    'molecule_summary'
]