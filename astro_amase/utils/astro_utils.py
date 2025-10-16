"""
Astro AMASE Utility Functions

A collection of utility functions for processing and analyzing astronomical
spectral data, including:
- Peak detection and lineshape analysis in radio spectra
- Molecular formula parsing and isotopologue identification
- Vibrational state detection from molecular line catalogs
- Array manipulation and search utilities
- Progress tracking and user interface elements

These functions support both single-dish and interferometric observations.
"""

import numpy as np
import math
import shutil
from scipy.signal import correlate
from scipy.stats import multivariate_normal
from scipy import signal
from datetime import datetime
from .molsim_utils import get_rms
from ..constants import ckm


def near_whole(number):
    """
    Check if a number is within 0.05 of a whole number.
    
    Useful for identifying frequency coincidences or testing if
    spectral features align with integer channel numbers.
    
    Args:
        number: Float value to test
        
    Returns:
        True if within 0.05 of nearest integer, False otherwise
    """
    return abs(number - round(number)) <= 0.05




def sortTupleArray(tup):
    """
    Sort an array of tuples by the second element.
    
    Commonly used to sort (frequency, intensity) pairs by intensity.
    Modifies the input list in-place.
    
    Args:
        tup: List of tuples to sort
        
    Returns:
        Sorted list (same reference as input)
    """
    tup.sort(key=lambda x: x[1])
    return tup


def find_nearest(arr, val):
    """
    Find the index of the array element nearest to a given value.
    
    Uses binary search for efficiency with sorted arrays.
    
    Args:
        arr: Sorted numpy array
        val: Target value to find
        
    Returns:
        Index of nearest element in arr
    """
    idx = np.searchsorted(arr, val, side="left")
    if idx > 0 and (idx == len(arr) or math.fabs(val - arr[idx-1]) 
                    < math.fabs(val - arr[idx])):
        return idx-1
    else:
        return idx


def deleteDuplicates(list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Modifies the input list in-place. More memory-efficient than
    converting to a set for large lists.
    
    Args:
        list: List to remove duplicates from (modified in-place)
    """
    seen = {}
    pos = 0
    for item in list:
        if item not in seen:
            seen[item] = True
            list[pos] = item
            pos += 1
    del list[pos:]


def printProgressBar(iteration, total, prefix='', suffix='', decimals=2, 
                     length=None, fill='█', printEnd="\r"):
    """
    Display a dynamic terminal progress bar that auto-adjusts to terminal width.
    
    Call this function repeatedly in a loop to update the progress bar.
    The bar automatically clears and redraws to provide smooth updates.
    
    Args:
        iteration: Current iteration count (0 to total)
        total: Total number of iterations
        prefix: Text to display before the progress bar
        suffix: Text to display after the progress bar
        decimals: Number of decimal places in percentage
        length: Fixed bar length (if None, auto-calculates from terminal width)
        fill: Character to use for the filled portion of the bar
        printEnd: Line ending character (default: carriage return for in-place update)
        
    Example:
        for i in range(100):
            # Do work here
            printProgressBar(i + 1, 100, prefix='Processing:', suffix='Complete')
    """
    # Get terminal width if length not specified
    if length is None:
        try:
            terminal_width = shutil.get_terminal_size().columns
        except (AttributeError, OSError):
            terminal_width = 80
        
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        overhead = len(prefix) + len(suffix) + len(percent) + 6
        length = max(10, terminal_width - overhead)
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    
    # Clear the entire line first, then print the progress bar
    terminal_width = shutil.get_terminal_size().columns
    print('\r' + ' ' * terminal_width, end='')
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    
    if iteration == total:
        print()


def softmax(x):
    """
    Compute softmax (normalized exponential).
    
    Subtracts the maximum value before exponentiation to prevent overflow.
    Used in for local score calculation in this algorithm.
    
    Args:
        x: Array of numerical values
        
    Returns:
        Array of same shape with values normalized to sum to 1
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def closest(lst, K):
    """
    Find the value and index in a list closest to a target value.
    
    Args:
        lst: Numpy array or list of values
        K: Target value to find
        
    Returns:
        Tuple of (index, value) for the closest element
    """
    idx = (np.abs(lst - K)).argmin()
    return idx, lst[idx]


def hasIso(mol):
    """
    Count the number of rare isotopologue substitutions in a molecular formula.
    
    Identifies isotopic substitutions (e.g., 13C, 15N, 18O, D) in molecular
    formulas following standard spectroscopic notation conventions.
    
    Args:
        mol: Molecular formula string (e.g., 'H213CO', 'CH3(13)CN')
        
    Returns:
        Integer count of isotopologue substitutions in the formula
        
    """
    isotopologue = 0
    isoList = ['17O', '(17)O', '18O', '(18)O', 'O18', '37Cl', '(37)Cl', 'Cl37', 
               '15N', '(15)N', 'N15', 'D', '(13)C', '13C', 'C13', '(50)Ti', '50Ti', 
               'Ti50', '33S', '(33)S', 'S33', '34S', '(34)S', 'S34', '36S', '(36)S', 
               'S36', '(29)S', '29S', '30S', 'S30', '(30)S']
    
    for iso in isoList:
        if iso in mol:
            isotopologue += 1

    # Handle special cases
    if "C13C" in mol:
        isotopologue = isotopologue - 1

    if 'D2' in mol:
        isotopologue += 1

    if 'D3' in mol:
        isotopologue += 1

    return isotopologue


def find_lineshape(frequency, intensity, template, threshold=0.8):
    """
    Identify spectral features matching a template lineshape using cross-correlation.
    
    Useful for finding repeated patterns in spectra, such as self absorption lines.
    
    Args:
        frequency: Array of frequency values
        intensity: Array of intensity values
        template: Template lineshape to search for
        threshold: Correlation coefficient threshold (0-1) for accepting matches
        
    Returns:
        Array of indices where template matches exceed the threshold
    """
    # Normalize for correlation
    template = (template - np.mean(template)) / np.std(template)
    intensity = (intensity - np.mean(intensity)) / np.std(intensity)

    correlation = correlate(intensity, template, mode='valid')
    matches = np.where(correlation >= threshold * np.max(correlation))[0]

    return matches


def find(s, ch):
    """
    Find all positions of a character in a string.
    
    Args:
        s: String to search
        ch: Character to find
        
    Returns:
        List of integer indices where character appears
    """
    return [i for i, ltr in enumerate(s) if ltr == ch]


def compute_pdf(mean, cov, allVectors, scale):
    """
    Compute scaled multivariate Gaussian probability density.
    
    Used in the calculation of structural relevance. 

    
    Args:
        mean: Mean vector of the Gaussian distribution
        cov: Covariance matrix
        allVectors: Points at which to evaluate the PDF
        scale: Scaling factor for the PDF values
        
    Returns:
        Array of scaled probability density values
    """
    gaussian_pdf = multivariate_normal(mean=mean, cov=cov)
    return scale * gaussian_pdf.pdf(allVectors)


def find_indices_within_threshold(values, target, threshold=5):
    """
    Find all values within a specified distance of a target value.
    
    Commonly used to find spectral lines within a frequency tolerance
    of a frequency.
    
    Args:
        values: List or array of values to search
        target: Target value to compare against
        threshold: Maximum distance from target to include
        
    Returns:
        List of indices where |value - target| <= threshold
    """
    values_array = np.array(values)
    indices = np.where(np.abs(values_array - target) <= threshold)[0]
    return indices.tolist()


def find_closest(array, value):
    """
    Find the index of the array element closest to a given value.
    
    Simple nearest-value search using absolute differences.
    
    Args:
        array: Numpy array to search
        value: Target value to find
        
    Returns:
        Index of the closest element
    """
    differences = np.abs(array - value)
    index_of_min = np.argmin(differences)
    return index_of_min


def find_peaks_local(freq_arr, int_arr, res, min_sep, is_sim=False, sigma=3, 
                     kms=True, local_rms=True, local_span=500, rms=None):
    """
    Detect spectral line peaks with configurable significance thresholding.
    
    This is a primary peak-finding routine for spectral analysis. It can
    operate in velocity or frequency space, use local or global noise estimation,
    and apply user-defined significance thresholds.
    
    Args:
        freq_arr: Array of frequency values (MHz or GHz)
        int_arr: Array of intensity values (Kelvin or Jy)
        res: Spectral resolution (frequency units)
        min_sep: Minimum separation between peaks (km/s or frequency units)
        is_sim: If True, returns all peaks without noise filtering (for simulations)
        sigma: Significance threshold (multiples of RMS noise)
        kms: If True, operates in velocity space; if False, frequency space
        local_rms: If True, computes RMS locally around each peak
        local_span: Number of channels for local RMS calculation (each side of peak)
        
    Returns:
        Array of indices corresponding to detected peaks in freq_arr/int_arr
        
    Note:
        Operating in velocity space (kms=True) ensures uniform sensitivity across
        the band and prevents bias toward higher frequencies. Local RMS helps
        in spectra with varying baseline noise.
    """
    #print('find peaks local rms', rms)
    if kms is True:
        max_f = np.amax(freq_arr)
        min_f = np.amin(freq_arr)
        cfreq = (max_f + min_f) / 2
        v_res = res * ckm / max_f
        v_span = (max_f - min_f) * ckm / (cfreq)
        v_samp = np.arange(-v_span / 2, v_span / 2 + v_res, v_res)
        freq_new = v_samp * cfreq / ckm + cfreq
        int_new = np.interp(freq_new, freq_arr, int_arr, left=0., right=0.)
        chan_sep = min_sep / v_res
    else:
        freq_new = freq_arr
        int_new = int_arr
        chan_sep = min_sep / res

    indices = signal.find_peaks(int_new, distance=chan_sep)

    if kms is True:
        indices = [find_nearest(freq_arr, freq_new[x]) for x in indices[0]]

    if is_sim is True:
        return np.asarray(indices)

    if local_rms is True:
        tmp = []
        for x in indices:
            if x - local_span > 0:
                l_idx = x - local_span
            else:
                l_idx = 0
            u_idx = x + local_span
            rms = get_rms(int_arr[l_idx:u_idx])
            if int_arr[x] > sigma * rms:
                tmp.append(x)
        return np.asarray(tmp)
    else:
        if rms == None:
            rms = get_rms(int_arr)
            indices = [x for x in indices if int_arr[x] > sigma * rms]
        else:
            indices = [x for x in indices if int_arr[x] > sigma * rms]
            
        return np.asarray(indices)


def hasVib(form):
    """
    Determine if a molecular line entry represents a vibrationally excited state.
    
    Parses molecular line catalog notation to identify vibrational excitation.
    This is important for filtering out vibrationally excited lines in cold
    astronomical environments where they are not expected.
    
    Args:
        form: Molecular name from catalog
        
    Returns:
        True if vibrationally excited, False if ground vibrational state
        
    Note:
        - Looks for 'v' in the string and analyzes vibrational quantum numbers
        - Special handling for CH3SH due to non-standard naming conventions
        - Checks for patterns like 'v=1', 'v>0', or vibrational quantum numbers != 0
        - Returns False for 'v=0' or 'v<=0' patterns (ground state)
    """
    vibEx = False
    if 'v' in form:
        if 'CH3SH' in form:
            return False
        
        idx = form.index('v')
        
        if '0' not in form:
            return True
        
        if 'le' in form:
            return False
        
        if '> 0' in form:
            return True

        greater = False
        allIdx = find(form, '0')
        for q in allIdx:
            if q > idx:
                greater = True

        if greater == False:
            return True
        else:
            return False
        
    return vibEx


def initial_banner():
    """
    Display the Astro AMASE welcome banner with ASCII art.
    
    Prints a formatted startup message including the software version,
    current timestamp, and contact information. The banner includes
    decorative star ASCII art centered above the main banner.
    
    Called once at the start of the identification pipeline to provide
    user feedback and version information.
    """
    banner = [
        "╔══════════════════════════════════════════════════════════╗",
        "║                      Astro AMASE                         ║",
        "║                     Version 1.0.0                        ║",
        "╠══════════════════════════════════════════════════════════╣",
        "║                                                          ║",
        "║                        Greetings!                        ║",
        "║         This code will help you automatically            ║",
        "║  identify molecules in radio astronomical observations   ║",
        "║                                                          ║",
        "║     If you have any questions, issues, or feedback       ║",
        "║            please email zfried@mit.edu                   ║",
        "║                                                          ║",
       f"║             Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                 ║",
        "╚══════════════════════════════════════════════════════════╝",
    ]

    star = [
        "             .        *        .",
        "   *                 .             *",
        "        .    *    ✦    *    .",
        "   *                 .             *",
        "             .        *        .",
    ]

    banner_width = len(banner[0])

    # Print stars centered on banner
    for line in star:
        padding = max((banner_width - len(line)) // 2, 0)
        print(" " * padding + line)

    print("\n")

    # Print banner
    for line in banner:
        print(line)


def checkAllLines(form, rms, sorted_freqs, sorted_ints, dv_value_freq, closestFreq, peak_freqs_full):
    """
    Evaluate whether a molecular candidate should be ruled out based on the presence 
    of its predicted strong spectral lines in the observed spectrum.

    The function examines the top predicted line intensities (20σ, 10σ, and 4σ levels)
    and checks how many of these transitions are matched by observed spectral peaks 
    within a given frequency tolerance (±0.5 × dV). If too few of these strong lines 
    are present in the observed spectrum, the molecule is flagged for exclusion 
    (i.e., `rule_out = True`).

    Parameters
    ----------
    form : str
        Identifier or formula of the molecular candidate (not directly used in scoring).
    rms : float
        Root mean square noise level of the observed spectrum.
    sorted_freqs : np.ndarray
        Array of predicted transition frequencies (sorted by intensity).
    sorted_ints : np.ndarray
        Array of predicted line intensities corresponding to `sorted_freqs`.
    dv_value_freq : float
        Frequency equivalent of the velocity width (Δv) used to define tolerance ranges.
    closestFreq : float
        Frequency of the candidate line currently under consideration; excluded from checks.
    peak_freqs_full : np.ndarray
        Array of observed 3σ spectral peak frequencies in the dataset.

    Returns
    -------
    rule_out : bool
        `True` if the candidate should be ruled out based on insufficient presence of
        strong predicted lines in the observed spectrum, otherwise `False`.

    Notes
    -----
    The function applies three intensity thresholds sequentially:
        1. **≥ 20σ lines:** Must have at least 80% matched within ±0.5 × dV.
        2. **≥ 10σ lines:** Must have at least 50% matched within ±0.5 × dV.
        3. **≥ 4σ lines (fallback):** Used only if ≤1 line above 10σ; must have at least 30% matched.

    If any of these conditions fail, the molecule is marked as ruled out.
    """

    rule_out = False #variable to determine whether to rule out the molecule based on these intensity checks

    #check how many of the 20 sigma lines are within 0.5*dV of a peak in the observed spectrum
    intensity_mask = sorted_ints >= 20 * rms
    freq_mask = sorted_freqs != closestFreq
    combined_mask = intensity_mask & freq_mask
    filtered_freqs = sorted_freqs[combined_mask]
    correct = 0

    tolerance = 0.5 * dv_value_freq

    if len(filtered_freqs) > 0:
        # Create all tolerance ranges at once
        lower_bounds = filtered_freqs - tolerance
        upper_bounds = filtered_freqs + tolerance
        
        # Check which 20 sigma lines are nearby peaks in the spectrum
        matches = np.any(
            (peak_freqs_full[:, None] >= lower_bounds) & 
            (peak_freqs_full[:, None] <= upper_bounds), 
            axis=0
        )
        
        #counts the number of lines with matches
        correct = np.sum(matches)
                    
        #if less than 80% of the 20 sigma lines are present, rule out the molecule
        if correct / len(filtered_freqs) < 0.8: 
            return True
        
            
    #check how many of the 10 sigma lines are within 0.5*dV of a peak in the observed spectrum
    intensity_mask = sorted_ints >= 10 * rms
    freq_mask = sorted_freqs != closestFreq
    combined_mask = intensity_mask & freq_mask
    filtered_freqs = sorted_freqs[combined_mask]
    correct = 0

    
    if len(filtered_freqs) > 1:
        # Create all tolerance ranges at once
        lower_bounds = filtered_freqs - tolerance
        upper_bounds = filtered_freqs + tolerance
        
        # Check which 10 sigma lines are nearby peaks in the spectrum
        matches = np.any(
            (peak_freqs_full[:, None] >= lower_bounds) & 
            (peak_freqs_full[:, None] <= upper_bounds), 
            axis=0
        )

        correct = np.sum(matches)
        if correct / len(filtered_freqs) < 0.5:
            return True

    else: #if there is only one (or less) 10 sigma line(s), check the 4 sigma lines
        intensity_mask = sorted_ints >= 4 * rms
        freq_mask = sorted_freqs != closestFreq
        combined_mask = intensity_mask & freq_mask
        filtered_freqs = sorted_freqs[combined_mask]
        correct = 0
        if len(filtered_freqs) > 0:
            # Create all tolerance ranges at once
            lower_bounds = filtered_freqs - tolerance
            upper_bounds = filtered_freqs + tolerance
            
            # Check which 5 sigma lines are nearby peaks in the spectrum
            matches = np.any(
                (peak_freqs_full[:, None] >= lower_bounds) & 
                (peak_freqs_full[:, None] <= upper_bounds), 
                axis=0
            )
            
            correct = np.sum(matches)
            if correct / len(filtered_freqs) < 0.3:  
                rule_out = True

    return rule_out
