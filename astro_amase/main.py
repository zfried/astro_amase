"""
Main entry point for Astro AMASE package.
"""

import os
import time
from typing import Dict, Any, Optional, Union

from .config.config_handler import load_config_file, get_parameters
from .constants import ckm
from .analysis.determine_linewidth import find_linewidth, find_linewidth_standalone
from .analysis.determine_vlsr import find_vlsr
from .data.load_data import load_data_get_peaks, load_data_original
from .data.create_dataset import create_full_dataset
from .core.run_assignment import run_full_assignment
from .output.create_output_file import write_output_text, remove_molecules_and_write_output
from .analysis.best_fit_model import full_fit
from .utils.astro_utils import initial_banner, save_parameters

def assign_observations(
    config_path: Optional[str] = None,
    spectrum_path: Optional[str] = None,
    directory_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function to perform automated molecular line assignment.
    
    Parameters
    ----------
    config_path : str, optional
        Path to YAML or JSON configuration file. If provided, other parameters
        are loaded from config file.
    spectrum_path : str, optional
        Path to spectrum file (.txt format with frequency and intensity columns).
        Used if config_path not provided.
    directory_path : str, optional
        Directory containing required database files and for output storage.
        Used if config_path not provided.
    force_ignore_molecules : list of str, optional
        List of molecule names to forcibly exclude from the assignment.
        Format: ['CH3OH', 'H2CO']. Default: [] (empty list)
    force_include_molecules : list of str, optional
        List of molecule names to forcibly include in the assignment.
        Format: ['CH3OH', 'H2CO']. Default: [] (empty list)
    **kwargs : dict
        Additional parameters:
        - temperature : float
            Excitation temperature in Kelvin
        - temperature_is_exact : bool, optional
            If True, uses exact temperature. If False, determines best-fit 
            temperature within ±100K. Default: False (will be set to True if vlsr provided)
        - vlsr : float, optional
            VLSR in km/s. If not provided, will be determined automatically.
            Note: If vlsr is provided, temperature_is_exact is forced to True.
        - sigma_threshold : float, optional
            Sigma threshold for line detection. Default: 5.0
        - rms_noise : float, optional
            RMS noise level. If not provided, calculated automatically.
        - observation_type : str, optional
            '1' or 'single_dish' for single dish, '2' or 'interferometric' 
            for interferometric. Default: 'single_dish'
        - dish_diameter : float, optional
            Dish diameter in meters (for single dish observations). Default: 100.0
        - beam_major_axis : float, optional
            Beam major axis in arcseconds (for interferometric)
        - beam_minor_axis : float, optional
            Beam minor axis in arcseconds (for interferometric)
        - source_size : float, optional
            Source diameter in arcseconds. Use 1E20 if source fills beam. Default: 1E20
        - continuum_temperature : float, optional
            Continuum temperature in Kelvin. Default: 2.7
        - valid_atoms : list of str, optional
            List of atomic symbols that could be present. Default: ['C', 'O', 'H', 'N', 'S']
        - known_molecules : list of str, optional
            List of SMILES strings for molecules known to be present in the source.
            Can include duplicates to require multiple copies (e.g., ['C=O', 'C=O'] 
            ensures at least 2 copies of formaldehyde are maintained in the detected 
            list throughout iteration). This influences structural relevance calculations.
            Default: None (no known molecules)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'assigner': IterativeSpectrumAssignment object with results
        - 'statistics': Assignment statistics (assigned, unidentified counts)
        - 'vlsr': Determined or input VLSR (km/s)
        - 'temperature': Determined or input temperature (K)
        - 'linewidth': Determined linewidth (km/s)
        - 'linewidth_freq': Linewidth in frequency units (MHz)
        - 'rms': RMS noise level (K)
        - 'source_size': inputted source size (arcseconds)
        - 'execution_time': Total time taken (seconds)
        - 'removed_molecules': List of molecules removed during fitting
        - 'output_files': Paths to generated output files
    
    Examples
    --------
    Using config file:
    >>> import astro_amase
    >>> results = astro_amase.assign_observations(config_path='config.yaml')
    
    With known VLSR and exact temperature:
    >>> results = astro_amase.assign_observations(
    ...     spectrum_path='spectrum.txt',
    ...     directory_path='./data/',
    ...     temperature=150.0,
    ...     vlsr=5.5,  # Automatically sets temperature_is_exact=True
    ...     sigma_threshold=5.0,
    ...     observation_type='interferometric',
    ...     beam_major_axis=0.5,
    ...     beam_minor_axis=0.5
    ... )
    
    With temperature estimation (±100K search):
    >>> results = astro_amase.assign_observations(
    ...     spectrum_path='spectrum.txt',
    ...     directory_path='./data/',
    ...     temperature=150.0,  # Initial guess
    ...     temperature_is_exact=False,  # Will search ±100K
    ...     sigma_threshold=5.0,
    ...     observation_type='interferometric',
    ...     beam_major_axis=0.5,
    ...     beam_minor_axis=0.5
    ... )
    
    Full parameter specification:
    >>> results = astro_amase.assign_observations(
    ...     spectrum_path='spectrum.txt',
    ...     directory_path='./data/',
    ...     temperature=150.0,
    ...     sigma_threshold=5.0,
    ...     observation_type='interferometric',
    ...     beam_major_axis=0.5,
    ...     beam_minor_axis=0.5,
    ...     source_size=1.0,
    ...     continuum_temperature=2.7,
    ...     valid_atoms=['C', 'O', 'H', 'N', 'S'],
    ...     rms_noise=0.01
    ... )
    
    With known molecules (count-based):
    >>> results = astro_amase.assign_observations(
    ...     spectrum_path='spectrum.txt',
    ...     directory_path='./data/',
    ...     temperature=150.0,
    ...     sigma_threshold=5.0,
    ...     observation_type='interferometric',
    ...     beam_major_axis=0.5,
    ...     beam_minor_axis=0.5,
    ...     known_molecules=['C=O', 'C=O', 'CC(=O)O']  # 2x formaldehyde, 1x acetic acid
    ... )
    
    With known molecules (count-based):
    >>> results = astro_amase.assign_observations(
    ...     spectrum_path='spectrum.txt',
    ...     directory_path='./data/',
    ...     temperature=150.0,
    ...     vlsr=5.5,
    ...     sigma_threshold=5.0,
    ...     observation_type='interferometric',
    ...     beam_major_axis=0.5,
    ...     beam_minor_axis=0.5,
    ...     known_molecules=['C=O', 'C=O', 'CC(=O)O']  # At least 2x formaldehyde, 1x acetic acid
    ... )
    """
    initial_banner()
    
    # Load parameters
    if config_path:
        print(f"Loading configuration from: {config_path}")
        user_outputs = load_config_file(config_path)
    else:
        if spectrum_path and directory_path:
            # Build parameters from direct inputs
            user_outputs = _build_parameters_from_kwargs(
                spectrum_path, directory_path, **kwargs
            )
        else:
            # Interactive mode
            print("Starting interactive parameter collection...")
            user_outputs = get_parameters()
    
    return run_pipeline(user_outputs)


def get_linewidth(
    spectrum_path: str,
    observation_type: str = 'single_dish',
    sigma_threshold: float = 5.0,
    rms_noise: Optional[float] = None,
    dish_diameter: float = 100.0,
    beam_major_axis: Optional[float] = None,
    beam_minor_axis: Optional[float] = None
) -> Dict[str, Any]:
    """
    Determine the linewidth from a spectrum.
    
    This is a lightweight function that only performs linewidth determination
    without running the full assignment pipeline.
    
    Parameters
    ----------
    spectrum_path : str
        Path to spectrum file (.txt format with frequency and intensity columns)
    observation_type : str, optional
        '1' or 'single_dish' for single dish, '2' or 'interferometric' 
        for interferometric. Default: 'single_dish'
    sigma_threshold : float, optional
        Sigma threshold for line detection. Default: 5.0
    rms_noise : float, optional
        RMS noise level. If not provided, calculated automatically.
    dish_diameter : float, optional
        Dish diameter in meters (for single dish observations). Default: 100.0
    beam_major_axis : float, optional
        Beam major axis in arcseconds (required for interferometric)
    beam_minor_axis : float, optional
        Beam minor axis in arcseconds (required for interferometric)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'linewidth_kms': Linewidth in km/s
        - 'linewidth_mhz': Linewidth in MHz
        - 'consider_hyperfine': Boolean indicating if hyperfine structure should be considered
        - 'rms': RMS noise level (K)
        - 'resolution': Spectral resolution (MHz)
    
    Examples
    --------
    For single dish observations (default):
    >>> import astro_amase
    >>> result = astro_amase.get_linewidth(
    ...     spectrum_path='spectrum.txt',
    ...     dish_diameter=100.0,
    ...     sigma_threshold=5.0
    ... )
    >>> print(f"Linewidth: {result['linewidth_kms']:.3f} km/s")
    
    For interferometric observations:
    >>> result = astro_amase.get_linewidth(
    ...     spectrum_path='spectrum.txt',
    ...     observation_type='interferometric',
    ...     beam_major_axis=0.5,
    ...     beam_minor_axis=0.5,
    ...     sigma_threshold=5.0
    ... )
    """
    # Standardize observation type
    if observation_type in ['single_dish', '1']:
        obs_type = '1'
    elif observation_type in ['interferometric', '2']:
        obs_type = '2'
    else:
        raise ValueError("observation_type must be '1', 'single_dish', '2', or 'interferometric'")
    
    # Handle beam parameters
    if obs_type == '1':
        if dish_diameter is None:
            raise ValueError("dish_diameter is required for single dish observations")
        bmaj_or_dish = dish_diameter
        bmin = None
    else:
        if beam_major_axis is None or beam_minor_axis is None:
            raise ValueError("beam_major_axis and beam_minor_axis are required for interferometric observations")
        bmaj_or_dish = beam_major_axis
        bmin = beam_minor_axis
    
    # Load data
    data, ll0, ul0, freq_arr, int_arr, resolution, min_separation, bandwidth = load_data_original(
        spectrum_path, obs_type, bmaj_or_dish, bmin
    )
    
    # Determine linewidth
    dv_kms, dv_mhz, consider_hyperfine = find_linewidth_standalone(
        freq_arr, int_arr, resolution,
        sigma_threshold, data, rms_noise
    )
    
    # Get or calculate RMS
    if rms_noise is None:
        # Load peaks to get calculated RMS
        peak_data = load_data_get_peaks(
            spectrum_path, sigma_threshold, dv_mhz,
            obs_type, bmaj_or_dish, bmin, rms_noise
        )
        rms = peak_data['rms']
    else:
        rms = rms_noise
    
    return {
        'linewidth_kms': float(dv_kms),
        'linewidth_mhz': float(dv_mhz),
        'consider_hyperfine': consider_hyperfine,
        'rms': float(rms),
        'resolution': float(resolution)
    }


def get_source_parameters(
    spectrum_path: str,
    directory_path: str,
    observation_type: str = 'single_dish',
    sigma_threshold: float = 5.0,
    temperature: Optional[float] = None,
    vlsr: Optional[float] = None,
    rms_noise: Optional[float] = None,
    continuum_temperature: float = 2.7,
    dish_diameter: float = 100.0,
    source_size: float = 1.E20,
    beam_major_axis: Optional[float] = None,
    beam_minor_axis: Optional[float] = None
) -> Dict[str, Any]:
    """
    Determine source parameters (linewidth, VLSR, temperature) without full assignment.
    
    This function runs the parameter determination steps of the pipeline without
    performing molecular line assignment. Useful for quickly characterizing a source.
    
    Parameters
    ----------
    spectrum_path : str
        Path to spectrum file (.txt format with frequency and intensity columns)
    directory_path : str
        Directory containing required database files
    observation_type : str, optional
        '1' or 'single_dish' for single dish, '2' or 'interferometric' 
        for interferometric. Default: 'single_dish'
    sigma_threshold : float, optional
        Sigma threshold for line detection. Default: 5.0
    temperature : float, optional
        Initial temperature guess (K). If not provided, defaults to 150 K.
        If vlsr is provided, this must be the exact temperature.
        If vlsr is not provided, this is used as initial guess for optimization.
    vlsr : float, optional
        VLSR in km/s. If not provided, will be determined automatically.
    rms_noise : float, optional
        RMS noise level. If not provided, calculated automatically.
    continuum_temperature : float, optional
        Continuum temperature in Kelvin. Default: 2.7
    dish_diameter : float, optional
        Dish diameter in meters (for single dish observations). Default: 100.0
    beam_major_axis : float, optional
        Beam major axis in arcseconds (required for interferometric)
    beam_minor_axis : float, optional
        Beam minor axis in arcseconds (required for interferometric)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'linewidth_kms': Linewidth in km/s
        - 'linewidth_mhz': Linewidth in MHz
        - 'vlsr': Determined or input VLSR (km/s)
        - 'temperature': Determined or input temperature (K)
        - 'rms': RMS noise level (K)
        - 'resolution': Spectral resolution (MHz)
    
    Examples
    --------
    Determine all parameters automatically (single dish, default):
    >>> import astro_amase
    >>> params = astro_amase.get_source_parameters(
    ...     spectrum_path='spectrum.txt',
    ...     directory_path='./data/',
    ...     dish_diameter=100.0
    ... )
    >>> print(f"VLSR: {params['vlsr']:.2f} km/s")
    >>> print(f"Temperature: {params['temperature']:.1f} K")
    >>> print(f"Linewidth: {params['linewidth_kms']:.3f} km/s")
    
    For interferometric observations:
    >>> params = astro_amase.get_source_parameters(
    ...     spectrum_path='spectrum.txt',
    ...     directory_path='./data/',
    ...     observation_type='interferometric',
    ...     beam_major_axis=0.5,
    ...     beam_minor_axis=0.5
    ... )
    
    With known VLSR, determine temperature:
    >>> params = astro_amase.get_source_parameters(
    ...     spectrum_path='spectrum.txt',
    ...     directory_path='./data/',
    ...     vlsr=5.5,
    ...     temperature=150.0,
    ...     dish_diameter=100.0
    ... )
    """
    # Standardize observation type
    if observation_type in ['single_dish', '1']:
        obs_type = '1'
    elif observation_type in ['interferometric', '2']:
        obs_type = '2'
    else:
        raise ValueError("observation_type must be '1', 'single_dish', '2', or 'interferometric'")
    
    # Handle beam parameters
    if obs_type == '1':
        if dish_diameter is None:
            raise ValueError("dish_diameter is required for single dish observations")
        bmaj_or_dish = dish_diameter
        bmin = None
    else:
        if beam_major_axis is None or beam_minor_axis is None:
            raise ValueError("beam_major_axis and beam_minor_axis are required for interferometric observations")
        bmaj_or_dish = beam_major_axis
        bmin = beam_minor_axis
    
    # Handle temperature default
    if temperature is None:
        temperature = 150.0
        temperature_is_exact = False
    else:
        # If VLSR is known, temperature must be exact
        temperature_is_exact = vlsr is not None
    
    # Ensure directory path ends with /
    if not directory_path.endswith('/'):
        directory_path += '/'
    
    print("\n=== Determining Source Parameters ===")
    
    # Load data
    print("Loading spectrum...")
    data, ll0, ul0, freq_arr, int_arr, resolution, min_separation, bandwidth = load_data_original(
        spectrum_path, obs_type, bmaj_or_dish, bmin
    )
    
    # Determine linewidth
    print("Determining linewidth...")
    dv_kms, dv_mhz, consider_hyperfine = find_linewidth(
        freq_arr, int_arr, resolution,
        sigma_threshold, data, rms_noise
    )
    print(f"Linewidth: {dv_kms:.3f} km/s ({dv_mhz:.3f} MHz)")
    
    # Get or calculate RMS
    if rms_noise is None:
        peak_data = load_data_get_peaks(
            spectrum_path, sigma_threshold, dv_mhz,
            obs_type, bmaj_or_dish, bmin, rms_noise
        )
        rms = peak_data['rms']
    else:
        rms = rms_noise
    print(f"RMS noise: {rms:.2e} K")
    
    # Determine VLSR and temperature
    vlsr_known = vlsr is not None
    
    if not vlsr_known:
        print("\nDetermining VLSR and temperature...")
        print(f"Initial temperature guess: {temperature} K")
        
        best_vlsr, best_temp, best_columns, vlsr_mols = find_vlsr(
            vlsr_known, vlsr, temperature_is_exact, temperature,
            directory_path, freq_arr, int_arr, resolution,
            dv_mhz, data, consider_hyperfine, min_separation,
            dv_kms, ll0, ul0, continuum_temperature, rms_noise, bandwidth, source_size
        )
        
        print(f"\nDetermined VLSR: {best_vlsr:.2f} km/s")
        print(f"Best-fit temperature: {best_temp:.1f} K")
    else:
        best_vlsr = vlsr
        best_temp = temperature
        best_columns = None
        print(f"\nUsing input VLSR: {best_vlsr:.2f} km/s")
        print(f"Using input temperature: {best_temp:.1f} K")
    
    results = {
        'linewidth_kms': float(dv_kms),
        'linewidth_mhz': float(dv_mhz),
        'vlsr': float(best_vlsr),
        'temperature': float(best_temp),
        'rms': float(rms),
        'resolution': float(resolution),
        'mols_used_in_analysis': vlsr_mols
    }
    
    print("\n=== Parameter Determination Complete ===")
    
    return results




def run_pipeline(user_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the full AMASE pipeline with given parameters.
    
    This function orchestrates the complete molecular line assignment workflow:
    1. Load observational data
    2. Determine linewidth from spectral features
    3. Determine VLSR and temperature (if not provided)
    4. Detect spectral peaks above threshold
    5. Create molecular candidate dataset from catalogs
    6. Assign molecular lines to observed peaks
    7. Perform best-fit model optimization
    8. Generate output files and reports
    
    Parameters
    ----------
    user_outputs : dict
        Dictionary of all required parameters containing:
        - spectrum_path : str
            Path to input spectrum file
        - directory_path : str
            Directory for database files and outputs
        - observation_type : str
            '1' for single dish, '2' for interferometric
        - bmaj_or_dish : float
            Dish diameter (m) or beam major axis (arcsec)
        - bmin : float or None
            Beam minor axis (arcsec) for interferometric, None for single dish
        - sigma_threshold : float
            Detection threshold in units of RMS
        - temperature : float
            Excitation temperature (K)
        - temperature_choice : bool
            True if temperature is exact, False to optimize ±100K
        - vlsr_known : bool
            True if VLSR is provided, False to determine
        - vlsr_input : float or None
            VLSR value (km/s) if known
        - source_size : float
            Source diameter (arcsec)
        - continuum_temperature : float
            Continuum temperature (K)
        - valid_atoms : list of str
            Atomic symbols to consider
        - rms_noise : float or None
            Manual RMS noise (K), None for automatic calculation
        - column_density_range: list
            Column density bounds for fitting. 
        - peak_df: str
            Path to csv file that contains peak information (optional)
        - peak_df_3sigma: str, optional
            Path to csv file that contains 3 sigma peaks in data (optional)
    Returns
    -------
    results : dict
        Complete results dictionary containing:
        - assigner : object
            IterativeSpectrumAssignment object with all assignments
        - peak_assignment_results: pandas DF
            dataframe that stores the assignment status of each line
        - column_density_results: pandas DF
            dataframe that stores the SMILES string and determined column density of each assigned molecule
        - statistics: dictionary
            dictionary containing summary statistics of the line assignment
        - vlsr : float
            Determined or input VLSR (km/s)
        - temperature : float
            Determined or input temperature (K)
        - linewidth : float
            Determined linewidth (km/s)
        - linewidth_freq : float
            Linewidth in frequency units (MHz)
        - rms : float
            RMS noise level (K)
        - source_size: float
            Inputted source size (arcseconds)
        - execution_time : float
            Total time taken (seconds)
        - _internal_data : dict
            Internal data needed for recreating plots in notebooks
        - output_files : dict
            Paths to all generated output files
    
    Notes
    -----
    The function prints progress updates throughout execution and reports
    timing for each major step. All output files are saved to the directory
    specified in user_outputs['directory_path'].
    """
    
    initial_time = time.perf_counter()
    
    # Load observational data
    print("\n=== Loading Observational Data ===")
    #specPath, observation_type, bmaj, bmin, rmsInp
    data, ll0, ul0, freq_arr, int_arr, resolution, min_separation, bandwidth, rms_original = load_data_original(
        user_outputs['spectrum_path'],
        user_outputs['observation_type'],
        user_outputs['bmaj_or_dish'],
        user_outputs['bmin'],
        user_outputs['rms_noise'],
    )

    print("\n=== User-Inputted Parameters ===")
    print(f"Spectrum path: {user_outputs['spectrum_path']}")
    print(f"Directory path: {user_outputs['directory_path']}")
    print(f"Observation type: {'Single Dish' if user_outputs['observation_type'] == '1' else 'Interferometric'}")

    if user_outputs['observation_type'] == '1':
        print(f"Dish diameter: {user_outputs['bmaj_or_dish']} m")
    else:
        print(f"Beam major axis: {user_outputs['bmaj_or_dish']} arcsec")
        print(f"Beam minor axis: {user_outputs['bmin']} arcsec")

    print(f"Sigma threshold: {user_outputs['sigma_threshold']}")
    if user_outputs['temperature_choice']:
        print(f"Inputted exact temperature: {user_outputs['temperature']} K")
        print("Algorithm will use this temperature for analysis")
    else:
        print(f"Inputted approximate temperature: {user_outputs['temperature']} K")
        print('Code will approximate temperature within 100 K of inputted value')
    
    
    print(f"VLSR known: {user_outputs['vlsr_known']}")

    if user_outputs['vlsr_known']:
        print(f"VLSR input: {user_outputs['vlsr_input']} km/s")
    else:
        print('Code will determine best-fit vlsr')

    print(f"Source size: {user_outputs['source_size']} arcsec")
    print(f"Continuum temperature: {user_outputs['continuum_temperature']} K")
    print(f"Valid atoms: {', '.join(user_outputs['valid_atoms'])}")

    if user_outputs['rms_noise'] is not None:
        print(f"Manual RMS noise: {user_outputs['rms_noise']} K")
    else:
        print("RMS noise: Auto-calculate")

    print("=" * 40)
    
    # Determine linewidth
    print("\n=== Determining Linewidth ===")
    dv_value, dv_value_freq, consider_hyperfine = find_linewidth(
        freq_arr, int_arr, resolution,
        user_outputs['sigma_threshold'],
        data, user_outputs['rms_noise']
    )
    
    # Determine VLSR and temperature
    print("\n=== Determining VLSR and Temperature ===")
    if not user_outputs['vlsr_known']:
        best_vlsr, best_temp, best_columns, vlsr_mols = find_vlsr(
            user_outputs['vlsr_known'],
            user_outputs['vlsr_input'],
            user_outputs['temperature_choice'],
            user_outputs['temperature'],
            user_outputs['directory_path'],
            freq_arr, int_arr, resolution,
            dv_value_freq, data,
            consider_hyperfine, min_separation,
            dv_value, ll0, ul0,
            user_outputs['continuum_temperature'],
            #user_outputs['rms_noise'],
            rms_original,
            bandwidth,
            user_outputs['source_size']

        )
    else:
        best_vlsr = user_outputs['vlsr_input']
        best_temp = user_outputs['temperature']
        best_columns = None
        print(f'Input VLSR: {round(best_vlsr, 2)} km/s')
        print(f'Input Temperature: {best_temp} K')

    

    param_time = time.perf_counter()
    print(f'\nTime for linewidth/VLSR determination: {round((param_time - initial_time) / 60, 2)} minutes')
    
    # Find peaks
    print("\n=== Detecting Spectral Peaks ===")
    peak_data = load_data_get_peaks(
        user_outputs['spectrum_path'],
        user_outputs['sigma_threshold'],
        dv_value_freq,
        user_outputs['observation_type'],
        user_outputs['bmaj_or_dish'],
        user_outputs['bmin'],
        user_outputs['rms_noise'],
        user_outputs['peak_df'],
        user_outputs['peak_df_3sigma']
    )
    print(f"RMS noise: {peak_data['rms']:.2g} K")
    
    # Create dataset
    print("\n=== Creating Molecular Candidate Dataset ===")
    print("Scraping catalogs and creating dataset...")
    all_loaded, noCanFreq, noCanInts, splatDict, cont_obj = create_full_dataset(
        user_outputs['directory_path'],
        peak_data['spectrum_freqs'],
        peak_data['spectrum_ints'],
        dv_value, peak_data['data'],
        best_vlsr, dv_value_freq,
        consider_hyperfine, best_temp,
        user_outputs['source_size'],
        ll0, ul0, freq_arr, resolution,
        user_outputs['continuum_temperature'],
        user_outputs['force_ignore_molecules']
    )
    dataset_time = time.perf_counter()
    print(f'Dataset creation time: {round((dataset_time - param_time) / 60, 2)} minutes')
    
    # Assign lines
    print("\n=== Assigning Molecular Lines ===")
    assigner, stats = run_full_assignment(
        best_temp,
        user_outputs['directory_path'],
        splatDict,
        user_outputs['valid_atoms'],
        dv_value_freq,
        peak_data['rms'],
        peak_data['peak_freqs_full'],
        known_molecules=user_outputs.get('known_molecules', None)
    )
    assign_time = time.perf_counter()
    print(f'Line assignment time: {round((assign_time - dataset_time) / 60, 2)} minutes')
    
    # Perform best-fit model
    print("\n=== Fitting Best-Fit Model ===")
    delMols, column_density_df, assignment_df, internal_data = full_fit(
        user_outputs['directory_path'],
        assigner, peak_data['data'],
        best_temp, dv_value, dv_value_freq,
        ll0, ul0, best_vlsr,
        peak_data['spectrum_freqs'],
        peak_data['spectrum_ints'],
        peak_data['rms'], cont_obj, user_outputs['force_include_molecules'], user_outputs['source_size'], user_outputs['column_density_range']
    )

    fit_time = time.perf_counter()
    print(f'Best-fit model time: {round((fit_time - assign_time) / 60, 2)} minutes')
    
    # Generate final output
    assignMols, stat_dict = remove_molecules_and_write_output(
        assigner, delMols,
        user_outputs['directory_path'],
        best_temp, dv_value, best_vlsr
    )
    
    final_time = time.perf_counter()
    total_time = final_time - initial_time
    
    print(f"\n=== Analysis Complete ===")
    print(f'Total execution time: {round(total_time / 60, 2)} minutes')
    print(f'\nResults saved to: {user_outputs["directory_path"]}')
    print('  - fit_spectrum.html: Interactive plot of fitted spectra')
    print('  - final_peak_results.csv: Molecular assignments for each line')
    print('  - output_report.txt: Detailed assignment report')
    print('  - column_density_results.csv: Best-fit column densities')
    
    # Return results
    results = {
        'assigner': assigner,
        'peak_assignment_results': assignment_df,
        'column_density_results': column_density_df,
        'statistics': stat_dict,
        'vlsr': float(best_vlsr),
        'temperature': float(best_temp),
        'linewidth': float(dv_value),
        'linewidth_freq': float(dv_value_freq),
        'rms': float(peak_data['rms']),
        'resolution': float(resolution),
        'source_size': float(user_outputs['source_size']),
        '_internal_data': internal_data,
        'execution_time': total_time,
        'output_files': {
            'interactive_plot': os.path.join(user_outputs['directory_path'], 'fit_spectrum.html'),
            'peak_results': os.path.join(user_outputs['directory_path'], 'final_peak_results.csv'),
            'detailed_report': os.path.join(user_outputs['directory_path'], 'output_report.txt'),
            'column_densities': os.path.join(user_outputs['directory_path'], 'column_density_results.csv')
        }
    }

    print('saving parameters')
    print(user_outputs['directory_path'])
    save_parameters(user_outputs, results, user_outputs['directory_path'])
    

    return results


def _build_parameters_from_kwargs(spectrum_path: str, directory_path: str, **kwargs) -> Dict[str, Any]:
    """
    Build parameter dictionary from function arguments.
    
    This function validates and processes keyword arguments into the standardized
    parameter dictionary format required by the AMASE pipeline. It handles default
    values, parameter validation, and the relationship between VLSR and temperature
    settings.
    
    Parameters
    ----------
    spectrum_path : str
        Path to spectrum file
    directory_path : str
        Directory for outputs and database files
    **kwargs : dict
        Keyword arguments containing:
        - temperature : float, required
            Excitation temperature (K)
        - temperature_is_exact : bool, optional
            Whether temperature is exact (True) or should be optimized (False)
            Default: False, but forced to True if vlsr is provided
        - vlsr : float, optional
            VLSR in km/s. If provided, temperature_is_exact is forced to True
        - sigma_threshold : float, optional
            Detection threshold. Default: 5.0
        - observation_type : str, optional
            '1'/'single_dish' or '2'/'interferometric'. Default: '2'
        - dish_diameter : float, optional
            Dish diameter (m) for single dish observations
        - beam_major_axis : float, optional
            Beam major axis (arcsec) for interferometric
        - beam_minor_axis : float, optional
            Beam minor axis (arcsec) for interferometric
        - source_size : float, optional
            Source diameter (arcsec). Default: 1E20
        - continuum_temperature : float, optional
            Continuum temperature (K). Default: 2.7
        - valid_atoms : list of str, optional
            Atomic symbols. Default: ['C', 'O', 'H', 'N', 'S']
        - rms_noise : float, optional
            Manual RMS noise (K). Default: None (auto-calculate)
        - known_molecules : list of str, optional
            SMILES strings for known molecules. Can include duplicates to 
            require multiple copies. Default: None (no known molecules)
        - known_molecules : list of str, optional
            List of SMILES strings for known molecules. Can include duplicates
            for count-based requirements. Default: None
        - column_density_range: list or floats, optional
            Minimum and maximum column density bounds for fitting.
            Format is [min, max]. Default: [1.e10, 1.e20]
        - peak_df: str, optional
            Path to csv file that contains peak information 
            Used if the user wants to input peak frequencies/intensities 
            instead of having code determine them.
            Must have a column titled 'frequency' and a column titled 'intensity'
            Default: None
        - peak_df_3sigma: str, optional
            Path to csv file that contains 3 sigma peaks in data
            Used for intensity analysis. Recommended to input if inputting optional
            peak_df parameter as well. 
            Must have a column titled 'frequency' and a column titled 'intensity'
            Default: None
    Returns
    -------
    params : dict
        Validated parameter dictionary ready for pipeline execution
    
    Raises
    ------
    ValueError
        If temperature is not provided
        If observation_type is invalid
        If required beam parameters are missing for the observation type
    
    Notes
    -----
    Temperature and VLSR relationship:
    - If vlsr is provided: temperature_is_exact is automatically set to True
      (user must provide exact temperature when VLSR is known)
    - If vlsr is not provided and temperature_is_exact is False:
      code will optimize temperature within ±100K of provided value
    - If vlsr is not provided and temperature_is_exact is True:
      code will use exact temperature to determine VLSR
    """
    # Check if temperature is provided
    if 'temperature' not in kwargs:
        raise ValueError("temperature parameter is required")
    
    # Handle VLSR
    vlsr_input = kwargs.get('vlsr', None)
    vlsr_known = vlsr_input is not None
    
    # Handle temperature_is_exact logic
    # If vlsr is known, temperature MUST be exact (user can't estimate when vlsr is provided)
    if vlsr_known:
        temperature_choice = True
        if 'temperature_is_exact' in kwargs and not kwargs['temperature_is_exact']:
            print("Warning: temperature_is_exact set to True because vlsr is provided")
    else:
        # User can specify whether temperature is exact or estimate
        temperature_choice = kwargs.get('temperature_is_exact', False)
    
    # Handle observation type
    obs_type = kwargs.get('observation_type', 'single_dish')
    if obs_type in ['single_dish', '1']:
        obs_type = '1'
    elif obs_type in ['interferometric', '2']:
        obs_type = '2'
    else:
        raise ValueError("observation_type must be '1', 'single_dish', '2', or 'interferometric'")
    
    # Set defaults
    params = {
        'spectrum_path': spectrum_path,
        'directory_path': directory_path if directory_path.endswith('/') else directory_path + '/',
        'sigma_threshold': kwargs.get('sigma_threshold', 5.0),
        'temperature': float(kwargs['temperature']),
        'observation_type': obs_type,
        'source_size': kwargs.get('source_size', 1E20),
        'continuum_temperature': kwargs.get('continuum_temperature', 2.7),
        'valid_atoms': kwargs.get('valid_atoms', ['C', 'O', 'H', 'N', 'S']),
        'rms_noise': kwargs.get('rms_noise', None),
        'rms_manual': kwargs.get('rms_noise') is not None,
        'vlsr_input': vlsr_input,
        'vlsr_known': vlsr_known,
        'temperature_choice': temperature_choice,
        'force_ignore_molecules': kwargs.get('force_ignore_molecules', []),
        'force_include_molecules': kwargs.get('force_include_molecules', []),
        'known_molecules': kwargs.get('known_molecules', None),
        'column_density_range': kwargs.get('column_density_range', [1.e10, 1.e20]),
        'peak_df':kwargs.get('peak_df', None),
        'peak_df_3sigma':kwargs.get('peak_df_3sigma',None),
    }
    
    # Handle beam parameters based on observation type
    if params['observation_type'] == '1':
        # Single dish - need dish_diameter
        if 'dish_diameter' not in kwargs:
            # Use default
            params['bmaj_or_dish'] = 100.0
        else:
            params['bmaj_or_dish'] = float(kwargs['dish_diameter'])
        params['bmin'] = None
    else:
        # Interferometric - need beam_major_axis and beam_minor_axis
        if 'beam_major_axis' not in kwargs or 'beam_minor_axis' not in kwargs:
            raise ValueError("beam_major_axis and beam_minor_axis are required for interferometric observations")
        params['bmaj_or_dish'] = float(kwargs['beam_major_axis'])
        params['bmin'] = float(kwargs['beam_minor_axis'])
    
    return params