# molsim.py - compatibility work-around for old pickle files

# Import all classes
from .molsim_classes import (
    Workspace,
    Catalog,
    Transition,
    Level,
    Molecule,
    PartitionFunction,
    Continuum,
    Simulation,
    Spectrum,
    Observation,
    Observatory,
    Source,
    Trace,
    Iplot,
    Ulim_Result
)

# Import all utility functions
from .molsim_utils import (
    _read_txt,
    get_rms,
    _find_ones,
    _trim_arr,
    find_nearest,
    find_nearest_vectorized,
    _make_gauss,
    _apply_vlsr,
    _apply_beam,
    _make_fmted_qnstr,
    _read_xy,
    _read_spectrum,
    load_obs,
    find_limits,
    find_peaks,
    load_mol,
    _make_qnstr,
    _make_level_dict,
    _load_catalog,
    _read_spcat
)

# Create a fake 'classes' submodule so pickle can find molsim.classes.Molecule
import sys
from types import ModuleType

# Create a module object for molsim.classes
classes = ModuleType('molsim.classes')
classes.Workspace = Workspace
classes.Catalog = Catalog
classes.Transition = Transition
classes.Level = Level
classes.Molecule = Molecule
classes.PartitionFunction = PartitionFunction
classes.Continuum = Continuum
classes.Simulation = Simulation
classes.Spectrum = Spectrum
classes.Observation = Observation
classes.Observatory = Observatory
classes.Source = Source
classes.Trace = Trace
classes.Iplot = Iplot
classes.Ulim_Result = Ulim_Result

# Register it in sys.modules so pickle can find it
sys.modules['molsim.classes'] = classes