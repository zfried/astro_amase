"""
Constants and configuration values for Astro AMASE.
"""

from astrochem_embedding import VICGAE
import scipy.constants

# Speed of light
ckm = 299792.458  # km/s
ccm = scipy.constants.c * 100  # cm/s
cm = scipy.constants.c  # m/s
c = scipy.constants.c #m/s

# Physical constants
h = scipy.constants.h  # Planck's constant (J·s)
k = scipy.constants.k  # Boltzmann's constant (J/K)
kcm = scipy.constants.k * (scipy.constants.h**-1) * ((scipy.constants.c * 100)**-1)  # cm^-1/K

# Algorithm parameters
maxMols = 300  # Maximum number of candidates per line
local_thresh = 0.7  # Local threshold for combined score
global_thresh = 93.5  # Global threshold for assignment

# Valid atoms
DEFAULT_VALID_ATOMS = ['C', 'O', 'H', 'N', 'S']

ALL_VALID_ATOMS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
    'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

# Molecules to ignore for isotopologue checks
ignoreIso = [
    'CO', 'C=O', 'O=C=S', 'C#N', 'C#CC#N', 'C#CC#CC#N',
    '[C-]#[O+]', '[C-]#[S+]', 'S', 'COC', 'COC=O', 'CC=O',
    'O=S', 'N', '[N]=O', 'CC#N', 'NC=O', 'N=C=O', 'CC(C)=O',
    'C=S', 'CS', '[C]1C=C1'
]

# VICGAE model for structural relevance
vicgae_model = VICGAE.from_pretrained()

# Structural relevance hyperparameters
covParam = 0.0105
span = 3.5