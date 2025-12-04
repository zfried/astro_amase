"""
If the VLSR is not known, this block estimates it from the spectrum.

Process:
1. Load a compressed database of transition frequencies, line strengths (logints), and associated
molecular metadata from CDMS/JPL catalogs.
2. Find spectral peaks in the observed spectrum by varying the sigma threshold until at least 150
peaks are detected (to ensure robust sampling).
3. For each detected peak:
- Identify candidate molecular transitions within a ±250 km/s frequency window from the database.
- Load the corresponding molecular models (and parents if applicable).
- Apply filtering rules: only keep strong lines (logint > -4.7), exclude certain molecules
    (e.g., CH3OCHO, problematic SO transitions), and exclude hyperfine lines unless specified.
- For each candidate, simulate its spectrum with the appropriate VLSR needed to align it to the
    observed peak frequency.
- Scale simulated intensities to match observed peak intensities, discard candidates that
    produce unrealistically strong lines, and keep those that align in frequency within 0.5*dv.
4. Collect the VLSRs from all valid candidates and compute the densest windows of the VLSR
distribution (via top_k_densest_windows).
5. Select the top densest window as the initial guess for VLSR.
- Extract molecules contributing to that window.
- Initialize their column densities, VLSR (center of the densest window), and temperature.
6. Fit the observed spectrum with these initial parameters using non-linear least squares
(scipy.optimize.least_squares), adjusting column densities, temperature, and VLSR.
7. Store the best-fit vlsr (best_vlsr), temperature, and column densities
for further analysis.

Outputs:
- best_vlsr : estimated VLSR
- best_temp : estimated excitation temperature
- best_columns : fitted column densities for the contributing molecules
"""

import os
import gzip
import pickle
import numpy as np
import sys
from scipy.optimize import least_squares
from ..utils.molsim_classes import Source, Simulation, Continuum
from ..utils.molsim_utils import find_peaks
from ..utils.astro_utils import find_peaks_local
from ..constants import ckm

def top_k_densest_windows(data, window_radius, top_k):
    """
    Finds the top_k windows (intervals) of length 2*window_radius
    that contain the most data points in a 1D dataset.
    
    Args:
        data: iterable of numeric values
        window_radius: half-width of each window
        top_k: number of densest windows to return

    Returns:
        List of tuples (count, lower_bound, upper_bound) for the top_k densest windows
    """
    data = np.sort(np.array(data))
    seen_ranges = set()
    results = []

    for i in range(len(data)):
        center = data[i]
        lower = center - window_radius
        upper = center + window_radius

        # Round for deduplication stability (avoid near-identical floats)
        key = (round(lower, 6), round(upper, 6))
        if key in seen_ranges:
            continue
        seen_ranges.add(key)

        count = np.searchsorted(data, upper, side='right') - np.searchsorted(data, lower, side='left')
        results.append((count, lower, upper))

    # Sort by count descending
    results.sort(reverse=True)

    return results[:top_k]

def check_5sigma(spectrum_frequencies, spectrum_intensities, 
                      target_frequencies, tolerance, min_intensity):
    """
    Check if spectrum contains intensities above threshold within tolerance of target frequencies.
    
    Parameters:
    -----------
    spectrum_frequencies : array-like
        Frequencies from your spectrum
    spectrum_intensities : array-like
        Corresponding intensities from your spectrum
    target_frequencies : array-like
        Frequencies you want to check for
    tolerance : float
        Plus/minus range around target frequency (e.g., 0.5 for ±0.5)
    min_intensity : float
        Minimum intensity threshold
    
    Returns:
    --------
     count : int
        Number of peaks with a nearby three sigma feature in spectrum.
    count/len(target_frequencies) : float
        Percentage of target frequencies with nearby 3 sigma feature
    """

    count = 0
    for target_freq in target_frequencies:
        # Find indices where spectrum frequencies are within tolerance
        freq_mask = np.abs(spectrum_frequencies - target_freq) <= tolerance
        
        # Find indices where intensities are above threshold
        intensity_mask = spectrum_intensities >= min_intensity
        
        # Combine both conditions
        match_mask = freq_mask & intensity_mask
        if np.any(match_mask):  # Check if ANY element is True
            count += 1

        
   
    return count, count/len(target_frequencies)



def top_k_densest_windows_next(data, window_radius, top_k):
    """
    Finds the top_k windows (intervals) of length 2*window_radius
    that contain the most data points in a 1D dataset.
    
    Args:
        data: iterable of numeric values
        window_radius: half-width of each window
        top_k: number of densest windows to return

    Returns:
        List of tuples (count, lower_bound, upper_bound) for the top_k densest windows
    """
    data = np.sort(np.array(data))
    seen_ranges = set()
    results = []

    for i in range(len(data)):
        center = data[i]
        lower = center - window_radius
        upper = center + window_radius

        # Round for deduplication stability (avoid near-identical floats)
        key = (round(lower, 6), round(upper, 6))
        if key in seen_ranges:
            continue
        seen_ranges.add(key)

        count = np.searchsorted(data, upper, side='right') - np.searchsorted(data, lower, side='left')
        results.append((count, lower, upper))

    # Sort by count descending
    results.sort(reverse=True)

    return results[:top_k]

def calc_needed_vlsr(v_measured, v_rest):
    """
    Calculate the required VLSR (Local Standard of Rest velocity) to align an observed (measured) frequency
    with a rest frequency.

    Parameters
    ----------
    v_measured : float
        The measured (observed) frequency of the spectral line (in the same units as v_rest, e.g., MHz).
    v_rest : float
        The rest frequency of the spectral line (in the same units as v_measured, e.g., MHz).

    Returns
    -------
    vlsr : float
        The velocity (in km/s) that needs to be applied to shift v_measured to v_rest (VLSR correction).

    Notes
    -----
    Uses the radio definition of Doppler shift:
        vlsr = c * (v_rest - v_measured) / v_rest
    where c is the speed of light in km/s (ckm).
    """
    vlsr = ckm * (v_rest - v_measured) / v_rest
    return vlsr

def simulate_sum(params, mol_list, dv_value, ll0, ul0, data, cont_obj, source_size):
    """
    Simulates and sums spectral profiles for a list of molecules using provided parameters.
    Parameters
    ----------
    params : list or array-like
        List of parameters where:
            - params[:-2] are column densities for each molecule in `mol_list`
            - params[-2] is the velocity (VLSR) value
            - params[-1] is the excitation temperature
    mol_list : list
        List of molecule objects to simulate.
    Returns
    -------
    np.ndarray
        The summed simulated spectral profile for all molecules.
    Notes
    -----
    - Uses a fixed line width (`dv_value`) for all simulations.
    - Assumes global variables `ll0`, `ul0`, `data`, and `molsim` are defined elsewhere.
    - Each molecule's spectrum is simulated with a Gaussian line profile and summed together.
    """

    columns = params[:-2]              # All but last two are column densities
    vlsr_value_fit = params[-2]        # Second to last is VLSR
    temp_fit = params[-1]              # Last parameter is temperature

    total_sim = None
    for mol, col in zip(mol_list, columns):
        src = Source(
            Tex=temp_fit,
            column=col,
            size=source_size,
            dV=dv_value,                # Keep dv_value fixed
            velocity=vlsr_value_fit,
            continuum=cont_obj
        )
        sim = Simulation(
            mol=mol,
            ll=ll0,
            ul=ul0,
            source=src,
            line_profile='Gaussian',
            use_obs=True,
            observation=data
        )
        spec = np.array(sim.spectrum.int_profile)
        total_sim = spec if total_sim is None else total_sim + spec

    return total_sim

def residuals(params, mol_list, y_exp, dv_value, ll0, ul0, data, cont_obj, source_size):
    """
    Calculate the residuals between simulated and experimental data for a given set of parameters and molecular list.
    Args:
        params (array-like): Parameters to be used in the simulation.
        mol_list (list): List of molecules to include in the simulation.
    Returns:
        numpy.ndarray: The difference between the simulated data and the experimental data (y_sim - y_exp).
    """
    y_sim = simulate_sum(params, mol_list, dv_value, ll0, ul0, data, cont_obj, source_size)
    return y_sim - y_exp

def simulate_sum_knowTemp(params, mol_list, dv_value, ll0, ul0, data, tempInput, cont_obj):
    """
    Simulates and sums spectral profiles for a list of molecules using provided parameters.
    Parameters
    ----------
    params : list or array-like
        List of parameters where:
            - params[:-2] are column densities for each molecule in `mol_list`
            - params[-2] is the velocity (VLSR) value
            - params[-1] is the excitation temperature
    mol_list : list
        List of molecule objects to simulate.
    Returns
    -------
    np.ndarray
        The summed simulated spectral profile for all molecules.
    Notes
    -----
    - Uses a fixed line width (`dv_value`) for all simulations.
    - Assumes global variables `ll0`, `ul0`, `data`, and `molsim` are defined elsewhere.
    - Each molecule's spectrum is simulated with a Gaussian line profile and summed together.
    """

    columns = params[:-1]              # All but last is column densities
    vlsr_value_fit = params[-1]        # Second to last is VLSR
    #temp_fit = params[-1]              # Last parameter is temperature

    total_sim = None
    for mol, col in zip(mol_list, columns):
        src = Source(
            Tex=tempInput,
            column=col,
            size=1.E20,
            dV=dv_value,                # Keep dv_value fixed
            velocity=vlsr_value_fit,
            continuum=cont_obj
        )
        sim = Simulation(
            mol=mol,
            ll=ll0,
            ul=ul0,
            source=src,
            line_profile='Gaussian',
            use_obs=True,
            observation=data
        )
        spec = np.array(sim.spectrum.int_profile)
        total_sim = spec if total_sim is None else total_sim + spec

    return total_sim

def residuals_knowTemp(params, mol_list, y_exp, dv_value, ll0, ul0, data, tempInput, cont_obj):
    """
    Calculate the residuals between simulated and experimental data for a given set of parameters and molecular list.
    Args:
        params (array-like): Parameters to be used in the simulation.
        mol_list (list): List of molecules to include in the simulation.
    Returns:
        numpy.ndarray: The difference between the simulated data and the experimental data (y_sim - y_exp).
    """
    y_sim = simulate_sum_knowTemp(params, mol_list, dv_value, ll0, ul0, data, tempInput, cont_obj)
    return y_sim - y_exp

def find_vlsr(vlsr_choice, vlsrInput, temp_choice, tempInput, direc, freq_arr, int_arr, resolution, dv_value_freq, data, consider_hyperfine, min_separation, dv_value,ll0,ul0, cont_temp, rms_original, bandwidth, source_size, vlsr_range, vlsr_mols):
    """
    Determine source velocity (VLSR) and excitation temperature through spectral fitting of common molecules.
    
    When VLSR is unknown (vlsr_choice=False), this function implements a algorithm that:
    1. Identifies spectral peaks with adaptive sigma threshold (targets ≥150 peaks)
    2. Searches transition database for candidates within ±250 km/s of each peak
    3. Simulates each candidate at the VLSR needed to align with observed peak
    4. Filters candidates by intensity scaling and frequency agreement
    5. Finds densest VLSR clustering using sliding window analysis
    6. Performs least-squares fit to optimize VLSR, temperature, and column densities
    
    The algorithm exploits the fact that correct VLSR brings multiple molecular transitions
    into agreement simultaneously, creating a clustering in VLSR space that can be identified
    statistically even in complex spectra.
    
    Parameters
    ----------
    vlsr_choice : bool
        If True, uses vlsrInput directly without determination.
        If False, determines VLSR from spectrum via fitting algorithm.
    vlsrInput : float
        User-provided VLSR value (km/s) used when vlsr_choice=True.
        Ignored when vlsr_choice=False.
    temp_choice : bool
        If True, fixes temperature to tempInput during VLSR optimization.
        If False, fits temperature simultaneously with VLSR.
    tempInput : float
        Excitation temperature (K). Used as initial guess if temp_choice=False,
        or fixed value if temp_choice=True.
    direc : str
        Base directory containing:
        - 'vlsr_database_logint.pkl.gz': compressed transition database with line strengths
        - 'cdms_pkl/': pickled CDMS molecule objects
    freq_arr : array_like
        Observed frequency array (MHz) from spectrum.
    int_arr : array_like
        Observed intensity array corresponding to freq_arr.
    resolution : float
        Spectral resolution (MHz) for peak detection and simulations.
    dv_value_freq : float
        Line width in frequency units (MHz) for candidate matching tolerance.
    data : Observation
        molsim Observation object containing spectrum attributes including data.spectrum.Tb.
    consider_hyperfine : bool
        If True, includes hyperfine catalog (tag > 200000).
        If False, excludes hyperfine catalogs.
    min_separation : float
        Minimum frequency separation (MHz) between peaks in simulated spectra.
    dv_value : float
        Line width (km/s) for Gaussian line profiles and VLSR window calculation.
    ll0 : float
        Lower frequency bound array (MHz) for spectral simulations.
    ul0 : float
        Upper frequency bound array (MHz) for spectral simulations.
    cont_temp : float
        Continuum temperature (K) for background radiation field.
    rmsInp : float
        RMS noise level for peak detection significance threshold.
    source_size: float
        Inputted source size (arcseconds)
    
    Returns
    -------
    best_vlsr : float
        Determined or input VLSR value (km/s). If vlsr_choice=True, returns vlsrInput.
        If vlsr_choice=False, returns optimized VLSR from fitting.
    best_temp : float
        Determined or input temperature (K). If temp_choice=True, returns tempInput.
        If temp_choice=False, returns optimized temperature from fitting.
    best_columns : array_like
        Best-fit column densities (cm^-2) for molecules contributing to densest
        VLSR window. Empty if vlsr_choice=True.
    
    Algorithm Details
    -----------------
    **Peak Detection:**
    - Adaptively searches sigma values [100, 75, 50, 40, 30, 25, 20, 15, 10, 5]
    - Selects first sigma yielding ≥150 peaks for robust VLSR sampling
    
    **Candidate Selection:**
    - Searches ±250 km/s window: Δν = 250 × (ν_center/300000 MHz)
    - Loads molecules from CDMS database (with isotopologue parent handling)
    - Filters by line strength: log(intensity) > -4.7 (strong transitions only)
    - Handles isotopologues via parentDict mapping to parent molecules
    
    **Candidate Validation:**
    - For each candidate, calculates required VLSR: v = c(ν_rest - ν_obs)/ν_rest
    - Simulates spectrum at column density 1e11 cm^-2 with calculated VLSR
    - Finds closest simulated peak to observed peak
    - Scales simulated intensities to match observed peak
    - Rejects if: scaled peak > 15× max observed intensity (poor intensity match)
    - Rejects if: closest peak > 0.5×dV_freq from observed (poor frequency match)
    
    **VLSR Determination:**
    - Collects all valid candidate VLSRs into distribution
    - Finds top 10 densest windows of radius ±dV using sliding window
    - Selects window with maximum candidate count as initial VLSR estimate
    - Identifies all molecules contributing to densest window
    
    **Optimization:**
    If temp_choice=False (fit temperature):
    - Initial parameters: [columns=1e14 cm^-2, VLSR=window_center, T=tempInput]
    - Column bounds: [1e10, 1e25] cm^-2
    - VLSR bounds: densest window edges
    - Temperature bounds: [max(0, T-100), min(500, T+100)] K
    - Minimizes residuals between observed and summed simulated spectra
    
    If temp_choice=True (fixed temperature):
    - Optimizes only column densities and VLSR
    - Temperature held constant at tempInput
    
    Notes
    -----
    - The 250 km/s search window accommodates most galactic sources
    - Line strength threshold empirically determined to avoid weak lines
    - Temperature bounds prevent unphysical values while allowing flexibility
    - Works best with spectra containing several strong common molecules (CO, CS, etc.)
    """
    
    

    if vlsr_choice == False:
        #print('BANDWIDTH!!!',bandwidth)
        #dictionary that links isotoplogues to their parent molecules
        parentDict = {'13CS, v = 0, 1': [('CS, v = 0 - 4', 44501)], 'CH318OH, vt le 2': [('CH3OH, vt = 0 - 2', 32504)], 'HDCO': [('H2CO', 30501)], 'H2C18O': [('H2CO', 30501)], 'N18O': [('NO, v = 0', 30517)], 'H213CO': [('H2CO', 30501)], 'H2C17O': [('H2CO', 30501)], '13CH3OH, vt = 0, 1': [('CH3OH, vt = 0 - 2', 32504)], 'CH2DCN': [('CH3CN, v = 0', 41505)], 'HDS': [('H2S', 34502)], '15NH3': [('NH3, v = 0', 17506)], 'DC3N, v = 0': [('HC3N, (0,0,0,0)', 51501)], 'H13CCCN, v = 0': [('HC3N, (0,0,0,0)', 51501)], 'HC13CCN, v = 0': [('HC3N, (0,0,0,0)', 51501)], 'HCC13CN, v = 0': [('HC3N, (0,0,0,0)', 51501)], 'S18O': [('SO, v = 0', 48501)], '18OCS': [('OCS, v = 0', 60503)], 'OC34S': [('OCS, v = 0', 60503)], 'O13CS': [('OCS, v = 0', 60503)], '34SO': [('SO, v = 0', 48501)], 'c-13CC2H2': [('c-C3H2', 38508)], 'c-CC13CH2': [('c-C3H2', 38508)], '13CH3CN, v = 0': [('CH3CN, v = 0', 41505)], 'CH313CN, v = 0': [('CH3CN, v = 0', 41505)], 'H234S': [('H2S', 34502)], 'C34S, v = 0, 1': [('CS, v = 0 - 4', 44501)], 'DCO+': [('HCO+, v = 0', 29507)], 'DCN, v = 0': [('HCN, v = 0', 27501)], 'H13CN, v = 0': [('HCN, v = 0', 27501)], 'H13CO+': [('HCO+, v = 0', 29507)], 'D2S': [('H2S', 34502), ('HDS', 35502)], 'C18O': [('CO, v = 0', 28503)], '13CO': [('CO, v = 0', 28503)]}
        igMols = ['SO, a 1Î\x94, v = 0, 1', 'c-C3H2', 'c-CC13CH2','c-13CC2H2', 'c-C3HD'] #list of molecules to filter out due to quality control, 'c-CC13CH2','c-13CC2H2', 'c-C3H2', 'c-C3HD'
        #if bandwidth <= 4000: #include methyl formate if broadband
        #    igMols.append('CH3OCHO')

        #print('IGNORE MOLS')
        #print(igMols)
        print('Determining VLSR now!')
        #mask_threshold = 5*rmsInp
        mask_threshold = 5*rms_original
        #uploading database of all transitions of molecules that are considered for VLSR determiation
        with gzip.open(os.path.join(direc,"vlsr_database_logint.pkl.gz"), "rb") as f: 
            database_freqs, database_errs, database_tags, database_lists, database_smiles, database_names, database_isos, database_vibs, database_forms, database_logints = pickle.load(f)


        center_freq = np.median(freq_arr)
        #min_freq_threshold = min(vlsr_range)*(max(freq_arr)/300000) #calculating approximately the maximum vlsr threshold in frequency
        #freq_threshold = 250*(center_freq/300000) #frequency threshold to allow for up to 250 km/s vlsr
        #max_freq_threshold = max(vlsr_range)*(max(freq_arr)/300000) #calculating approximately the maximum vlsr threshold in frequency
        min_freq_threshold = min(vlsr_range)*(center_freq/300000) #calculating approximately the maximum vlsr threshold in frequency
        max_freq_threshold = max(vlsr_range)*(center_freq/300000) #calculating approximately the maximum vlsr threshold in frequency
        #print('Frequency Threshold (MHz):', freq_threshold)
        molDict = {}
        allCans = []
        numPeakVals = []
        foundSig = False
        sigList = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
        sigList.reverse()
        for sig in sigList:
            #print(sig)
            if foundSig == False:
                peak_indices = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr), 2*dv_value_freq), sigma=sig, local_rms=False, rms=rms_original)
                if len(peak_indices) >= 150: #find sigma value that has at least 150 peaks
                    foundSig = True


        if len(peak_indices) == 0:
            raise ValueError("Error: No peaks found at 5 sigma or stronger. You need to adjust the rms noise level.")

        sorted_peak_freqs = data.spectrum.frequency[peak_indices]
        sorted_peak_ints = abs(data.spectrum.Tb[peak_indices])
        #print('Sorted Peak Frequencies:')
        #print(sorted_peak_freqs)

        if vlsr_mols != 'all':
            vlsr_mols = set(vlsr_mols)


        if vlsr_mols == 'all':
            for i in range(len(sorted_peak_freqs)): #loop through all determined peaks
                line_mols = []
                start_idx = np.searchsorted(database_freqs, sorted_peak_freqs[i] + min_freq_threshold, side="left") #start index in uploaded database for candidates within specified vlsr range
                end_idx = np.searchsorted(database_freqs, sorted_peak_freqs[i] + max_freq_threshold, side="right") #end index in uploaded database for candidates within specified vlsr range
                for match_idx in range(start_idx, end_idx):
                    match_tu = (
                    database_names[match_idx], database_forms[match_idx], database_smiles[match_idx], database_freqs[match_idx],
                    database_errs[match_idx], database_isos[match_idx], database_tags[match_idx], database_lists[match_idx],
                    database_vibs[match_idx],database_logints[match_idx]) #tuple of matched molecule data
                    if (database_names[match_idx], database_lists[match_idx]) not in molDict:
                        if database_lists[match_idx] == 'CDMS': #upload CDMS molecule and load into molsim object
                            molPath = os.path.join(direc, 'cdms_pkl', f"{database_tags[match_idx]:06d}.pkl")
                            with open(molPath, 'rb') as md:
                                mol = pickle.load(md)
                        molDict[(database_names[match_idx], database_lists[match_idx])] = mol #add molecule to molDict
                        if database_names[match_idx] in parentDict: #if the molecule is an isotopologue, add the parent molecule to molDict too
                            for pa in parentDict[database_names[match_idx]]:
                                if (pa[0],'CDMS') not in molDict:
                                    molPath = os.path.join(direc, 'cdms_pkl', f"{pa[1]:06d}.pkl")
                                    with open(molPath, 'rb') as md:
                                        mol = pickle.load(md)
                                    molDict[(pa[0],'CDMS')] = mol   
                                    #allNames.append(pa)
                    
                    #consider_hyperfine = False
                    #only consider strong lines with logint in .cat file > -5.1 (determined through some testing)
                    #print(match_tu)
                    if consider_hyperfine == True:
                        if match_tu[-1] > -5.1:
                            line_mols.append(match_tu)
                    else:
                        if match_tu[6] < 200000:  # Exclude hyperfine lines
                            #if 'SO, a 1Î\x94, v = 0, 1' not in match_tu[0] and 'c-C3H2' not in match_tu[0] and 'CH3OCHO' not in match_tu[0] and 'c-CC13CH2' not in match_tu[0]: #'CH3OCHO' not in match_tu[0] and 
                            rule_out_var = False
                            for igm in igMols: #excluding problematic molecules for quality control
                                if igm in match_tu[0]:
                                    rule_out_var = True
                            if rule_out_var == False:
                                if match_tu[-1] > -5.1 and match_tu not in line_mols:
                                    line_mols.append(match_tu)

                allCans.append(line_mols) #store all candidates for each peak frequency within 250 km/s


        else:
            for i in range(len(sorted_peak_freqs)): #loop through all determined peaks
                line_mols = []
                start_idx = np.searchsorted(database_freqs, sorted_peak_freqs[i] + min_freq_threshold, side="left") #start index in uploaded database for candidates within specified vlsr range
                end_idx = np.searchsorted(database_freqs, sorted_peak_freqs[i] + max_freq_threshold, side="right") #end index in uploaded database for candidates within specified vlsr range
                for match_idx in range(start_idx, end_idx):
                    if database_names[match_idx] in vlsr_mols:
                        match_tu = (
                        database_names[match_idx], database_forms[match_idx], database_smiles[match_idx], database_freqs[match_idx],
                        database_errs[match_idx], database_isos[match_idx], database_tags[match_idx], database_lists[match_idx],
                        database_vibs[match_idx],database_logints[match_idx]) #tuple of matched molecule data
                        if (database_names[match_idx], database_lists[match_idx]) not in molDict:
                            if database_lists[match_idx] == 'CDMS': #upload CDMS molecule and load into molsim object
                                molPath = os.path.join(direc, 'cdms_pkl', f"{database_tags[match_idx]:06d}.pkl")
                                with open(molPath, 'rb') as md:
                                    mol = pickle.load(md)
                            molDict[(database_names[match_idx], database_lists[match_idx])] = mol #add molecule to molDict
                            if database_names[match_idx] in parentDict: #if the molecule is an isotopologue, add the parent molecule to molDict too
                                for pa in parentDict[database_names[match_idx]]:
                                    if (pa[0],'CDMS') not in molDict:
                                        molPath = os.path.join(direc, 'cdms_pkl', f"{pa[1]:06d}.pkl")
                                        with open(molPath, 'rb') as md:
                                            mol = pickle.load(md)
                                        molDict[(pa[0],'CDMS')] = mol   
                                        #allNames.append(pa)
                        
                        #consider_hyperfine = False
                        #only consider strong lines with logint in .cat file > -5.1 (determined through some testing)
                        #print(match_tu)
                        if consider_hyperfine == True:
                            if match_tu[-1] > -5.1:
                                line_mols.append(match_tu)
                        else:
                            if match_tu[6] < 200000:  # Exclude hyperfine lines
                                #if 'SO, a 1Î\x94, v = 0, 1' not in match_tu[0] and 'c-C3H2' not in match_tu[0] and 'CH3OCHO' not in match_tu[0] and 'c-CC13CH2' not in match_tu[0]: #'CH3OCHO' not in match_tu[0] and 
                                rule_out_var = False
                                for igm in igMols: #excluding problematic molecules for quality control
                                    if igm in match_tu[0]:
                                        rule_out_var = True
                                if rule_out_var == False:
                                    if match_tu[-1] > -5.1 and match_tu not in line_mols:
                                        line_mols.append(match_tu)

                allCans.append(line_mols) #store all candidates for each peak frequency within 250 km/s

        #print('all cans')
        #for g in allCans:
        #    print(g)
        maxPeakInt = max(sorted_peak_ints) #storing the maximum peak intensity of the spectrum
        allCansFiltered = []
        for i in range(len(allCans)): #loop through all candidates for each peak frequency
            indivCans = []
            indivVlsrs = []
            already_indiv = []
            for can in allCans[i]:
                indivFreq = can[3]
                vlsr_needed = calc_needed_vlsr(sorted_peak_freqs[i], indivFreq) #calculate the needed vlsr for the candidate to match the peak frequency
                #print(vlsr_needed)
                mol = molDict[(can[0], can[-3])]
                cont_obj = Continuum(type='thermal', params=cont_temp)
                src = Source(size=source_size, dV=dv_value, velocity=vlsr_needed, Tex=tempInput, column=1.E11, continuum=cont_obj)
                #simulate the spectrum for the candidate molecule at the needed vlsr
                sim = Simulation(
                    mol=mol,
                    ll=ll0,
                    ul=ul0,
                    observation=data,
                    source=src,
                    line_profile='Gaussian',
                    use_obs=True)
                hasCan = False
                if len(sim.spectrum.freq_profile) > 0:
                    #finding the peaks in the simulated spectrum
                    sim_peak_indices = find_peaks(
                        sim.spectrum.freq_profile, sim.spectrum.int_profile,
                        res=resolution,
                        min_sep=min_separation,
                        is_sim=True
                    )

                    #finding if there are strong peaks of the candidate in the frequency range and storing them               
                    if len(sim_peak_indices) > 0:
                        sim_peak_freqs = sim.spectrum.freq_profile[sim_peak_indices]
                        sim_peak_ints = np.abs(sim.spectrum.int_profile[sim_peak_indices])
                        hasCan = True
                    #else:
                    #    print('No peaks found for ' + str(can[0]) + ' at ' + str(can[3]))
                #else:
                #     print('No spectrum found for ' + str(can[0]) + ' at ' + str(can[3]))

                
                if hasCan:
                    #print(sim_peak_freqs)
                    closestIdx = np.argmin(np.abs(sim_peak_freqs - sorted_peak_freqs[i])) #find the closest peak in simulated spectrum to the observed peak frequency
                    closestFreq = sim_peak_freqs[closestIdx]
                    closestInt = sim_peak_ints[closestIdx]
                    scaleVal = sorted_peak_ints[i]/closestInt #scaling factor for the simulated peak intensity to match the observed peak intensity
                    scaled_peak_ints = sim_peak_ints * scaleVal #scaling all simulated peak intensities
                    hasTooStrong = False
                    if max(scaled_peak_ints) > 10*max(sorted_peak_ints): #checking if any scaled peak intensity is too strong (> 15 times the maximum peak intensity of the observed spectrum)
                        hasTooStrong = True
                    if abs(closestFreq-sorted_peak_freqs[i]) >= 0.5*dv_value_freq:
                        hasTooStrong = True
                    #if the closest peak frequency is within 0.5*dv_value_freq of the observed peak frequency and no scaled peak intensity is too strong
                    # add the candidate to the filtered candidates list
                    if bandwidth >= 15000 and abs(closestFreq-sorted_peak_freqs[i]) <= 0.5*dv_value_freq and hasTooStrong == False:
                        allCansFiltered.append((can,vlsr_needed,closestFreq))
                    elif bandwidth < 15000 and hasTooStrong == False:
                        #for low bandwidth cases, be more careful and check whether the 5 sigma lines of the molecule are present.
                        mask = scaled_peak_ints > mask_threshold
                        sim_peak_freqs_filtered = sim_peak_freqs[mask]
                        sim_peak_ints_filtered = scaled_peak_ints[mask]
                        mask = sim_peak_freqs_filtered != closestFreq
                        sim_peak_freqs_final = sim_peak_freqs_filtered[mask]
                        sim_peak_ints_final = sim_peak_ints_filtered[mask]
                        if len(sim_peak_freqs_final):
                            numMatch, percentMatch = check_5sigma(freq_arr, int_arr, sim_peak_freqs_final, 0.5*dv_value_freq, 3*rms_original)
                            #if abs(vlsr_needed-0) < 4:
                                #print(can, percentMatch)
                            if percentMatch >= 0.25:
                                rule_out_iso = False
                                if can[0] in parentDict:
                                    #print('parent found for ' + str(can[0]))
                                    for pa in parentDict[can[0]]:
                                        #print(pa)
                                        par_mol = molDict[(pa[0], 'CDMS')]
                                        par_src = Source(size=source_size, dV=dv_value, velocity=vlsr_needed, Tex=tempInput, column=scaleVal*1.E11, continuum=cont_obj)
                                        sim = Simulation(
                                            mol=par_mol,
                                            ll=ll0,
                                            ul=ul0,
                                            observation=data,
                                            source=par_src,
                                            line_profile='Gaussian',
                                            use_obs=True)
                                        if len(sim.spectrum.freq_profile) > 0:
                                            sim_peak_indices = find_peaks(
                                                sim.spectrum.freq_profile, sim.spectrum.int_profile,
                                                res=resolution,
                                                min_sep=min_separation,
                                                is_sim=True
                                            )

                                            #finding if there are strong peaks of the main isotopologue in the frequency range                
                                            if len(sim_peak_indices) > 0:
                                                sim_peak_freqs = sim.spectrum.freq_profile[sim_peak_indices]
                                                sim_peak_ints = np.abs(sim.spectrum.int_profile[sim_peak_indices])
                                                mask = sim_peak_ints > mask_threshold
                                                sim_peak_freqs_filtered = sim_peak_freqs[mask]
                                                sim_peak_ints_filtered = sim_peak_ints[mask]
                                                #tolVal = 0.5*dv_value_freq

                                                if len(sim_peak_freqs_filtered):
                                                    numMatch, percentMatch = check_5sigma(freq_arr, int_arr, sim_peak_freqs_filtered, 0.5*dv_value_freq, 3*rms_original)
                                                    #print(percentMatch)
                                                    if percentMatch < 0.3:
                                                        rule_out_iso = True
                                                    #print(rule_out_iso)


                                if rule_out_iso == False:
                                    allCansFiltered.append((can,vlsr_needed,closestFreq))


                                #allCansFiltered.append((can,vlsr_needed,closestFreq))
                        else:
                            allCansFiltered.append((can,vlsr_needed,closestFreq))






        

        #print('DV VALUE')
        #print(dv_value)
        total_vlsrs = []
        #storing the vlsrs of all filtered candidates
        for i in allCansFiltered:
            total_vlsrs.append(i[1])
            #if abs(i[1]) < 2:
            #    for zi in i:
            #        print(zi)

            #s    print('')

        y_exp = data.spectrum.Tb

        #determining the top densest windows of +/- dV in the vlsr distribution
        top_bins = (top_k_densest_windows(total_vlsrs, window_radius=1*dv_value, top_k=50))
        #print(top_bins)
        top_scored_bins = []
        for tb in top_bins:
            if tb[0] == top_bins[0][0]:
                top_scored_bins.append(tb)

        if len(top_scored_bins) > 1:
        #print(total_vlsrs)
        #preferentially choose the lower vlsrs
            sorted_top_bins = sorted(top_scored_bins, key=lambda x: (x[0], abs((x[1] + x[2]) / 2)))
            #print(sorted_top_bins)
            top_bin = sorted_top_bins[0]
        else:
            top_bin = top_scored_bins[0]

        mol_list = []
        labels = [] 
        indivBin = []
        #print(top_bin)
        #top_bin = sorted_top_bins[0]
        '''
        #print('top bin')
        for i in allCansFiltered:
            if i[1] > top_bin[1] and i[1] < top_bin[2]:
                for zi in i:
                    print(zi)
                print('')

        print('0!!!')
        for i in allCansFiltered:
            if i[1] > -2 and i[1] < 2:
                for zi in i:
                    print(zi)
                print('')
        '''
    

        #storing the molecules that have a transition contributing to the top densest window
        for z in allCansFiltered:
            if z[1] <= top_bin[2] and z[1] >= top_bin[1]:
                if z[0][0] not in labels:
                    labels.append(z[0][0])
                    mol_list.append(molDict[z[0][0],'CDMS'])

        if temp_choice == False:
            #setting initial column densities for fit
            initial_columns = [1e14] * len(mol_list)
            #setting initial vlsr to be the center of the top densest window
            initial_vlsr = (top_bin[1] + top_bin[2]) / 2
            #setting initial temperature to the user input temperature
            initial_temp = tempInput  
            initial_params = initial_columns + [initial_vlsr, initial_temp]

            column_bounds = ([1e10] * len(mol_list), [1e25] * len(mol_list))
            vlsr_bounds = (top_bin[1], top_bin[2])
            minTempBound = max(0, tempInput - 100) # Ensure temperature is not negative 
            maxTempBound = min(500, tempInput + 100) #don't let temperature exceed 500 or 100 plus the input temperature
            temp_bounds = (minTempBound, maxTempBound)
            lower_bounds = column_bounds[0] + [vlsr_bounds[0], temp_bounds[0]]
            upper_bounds = column_bounds[1] + [vlsr_bounds[1], temp_bounds[1]]
            # Use scipy's least_squares to fit the model to the data to determine the best column densities, vlsr, and temperature
            result = least_squares(
                residuals,
                x0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                args=(mol_list, y_exp, dv_value, ll0, ul0, data, cont_obj, source_size),
                method='trf',
                verbose=0)

            # Extract the best-fit parameters
            best_temp = round(result.x[-1],1)
            tempInput = best_temp
            best_vlsr = result.x[-2]
            best_columns = result.x[:-2]  # All but last two are column densities


        else:
            #setting initial column densities for fit
            initial_columns = [1e14] * len(mol_list)
            #setting initial vlsr to be the center of the top densest window
            initial_vlsr = (top_bin[1] + top_bin[2]) / 2
            #setting initial temperature to the user input temperature
            initial_temp = tempInput  
            initial_params = initial_columns + [initial_vlsr]
            column_bounds = ([1e10] * len(mol_list), [1e25] * len(mol_list))
            vlsr_bounds = (top_bin[1], top_bin[2])
            lower_bounds = column_bounds[0] + [vlsr_bounds[0]]
            upper_bounds = column_bounds[1] + [vlsr_bounds[1]]
            # Use scipy's least_squares to fit the model to the data to determine the best column densities, vlsr, and temperature
            result = least_squares(
                residuals_knowTemp,
                x0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                args=(mol_list, y_exp, dv_value, ll0, ul0, data, tempInput, cont_obj),
                method='trf',
                verbose=0)

            # Extract the best-fit parameters
            best_temp = tempInput
            best_vlsr = result.x[-1]
            best_columns = result.x[:-1]  # All but last are column densities



            
    print('Determined VLSR:', round(best_vlsr,2), 'km/s')
    print('Determined (or inputted) Temperature:', best_temp, 'K')
    return best_vlsr, best_temp, best_columns, labels
