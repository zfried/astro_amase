"""
Data Loading and Peak Detection Module

This module provides utilities for loading astronomical spectral data, 
processing observational parameters, and identifying spectral line peaks.
It handles both single-dish and interferometric observations, calculating
appropriate resolution parameters and noise characteristics.

Main functionality:
- Load and validate spectral data from text files
- Configure observatory parameters (single-dish or array)
- Detect and sort spectral peaks above significance thresholds
- Calculate RMS noise and spectral resolution
"""

import numpy as np
import pandas as pd
import sys
import warnings
from ..constants import ckm
from ..utils.astro_utils import find_peaks_local
from ..utils.molsim_utils import load_obs, find_limits, get_rms
from ..utils.molsim_classes import Observatory

def load_data_original(specPath, observation_type, bmaj, bmin, rmsInp):
    """
    Load observational spectrum and configure basic observatory parameters.
    
    This function initializes the data structure with telescope configuration
    and determines the frequency range and resolution of the spectrum.
    
    Args:
        specPath: Path to the spectrum text file
        observation_type: Either 'single_dish' or interferometric array
        bmaj: Major axis beam size or dish diameter (arcsec or meters)
        bmin: Minor axis beam size for interferometric observations (arcsec)
    
    Returns:
        Tuple containing processed data object, frequency limits, arrays,
        and resolution parameters needed for peak finding
    """
    data = load_obs(specPath, type = 'txt') #load spectrum
    ll0, ul0 = find_limits(data.spectrum.frequency) #determine upper and lower limits of the spectrum
    freq_arr = data.spectrum.frequency #frequency array of spectrum
    int_arr = data.spectrum.Tb #intensity array of spectrum
    # Calculate differences between consecutive points
    freq_differences = np.diff(freq_arr)
    if freq_arr[1] - freq_arr[0] > 0:
        resolution = freq_arr[1] - freq_arr[0]
    else:
        resolution = np.median(freq_differences)

    if freq_arr[1] - freq_arr[0] < 0:
        raise ValueError('Spectrum must be in increasing frequency order, but appears to be in decreasing order.')
    if freq_arr[1] == freq_arr[0]:
        warnings.warn('First two frequency values in spectrum are identical.', UserWarning)



    min_separation = resolution * ckm / np.amax(freq_arr) #minimum separation between peaks in km/s

    #setting telescope parameters
    #print(observation_type)
    if observation_type == '1':
        observatory1 = Observatory(sd=True, dish=bmaj)
    else:
        observatory1 = Observatory(sd=False, array=True,synth_beam = [bmaj,bmin])

    data.observatory = observatory1
    
    bandwidth = ul0[-1] - ll0[0]
    
    if rmsInp is None:
        rms = get_rms(int_arr)
    else:
        #if type(rmsInp) == dict:
        if isinstance(rmsInp, dict):
            rms_dict_original = map_rms_to_spectrum(freq_arr, rmsInp)
            rms = np.median(rms_dict_original)

        else:  
            rms = rmsInp



    return data, ll0, ul0, freq_arr, int_arr, resolution, min_separation, bandwidth, rms


def load_data_get_peaks(specPath, sigOG, dv_value_freq, observation_type, bmaj, bmin, rmsInp, peak_df, peak_df_3sigma):

    """
    Load spectrum and identify significant spectral line peaks.
    
    Performs comprehensive peak detection at a specified significance level,
    calculates noise characteristics, and returns peaks sorted by intensity.
    Also stores a complete catalog of 3-sigma peaks for intensity check purposes.
    
    Args:
        specPath: Path to the spectrum text file
        sigOG: Sigma threshold for primary peak detection
        dv_value_freq: Velocity width parameter in frequency units
        observation_type: 'single_dish' or interferometric configuration
        bmaj: Major axis beam size or dish diameter
        bmin: Minor axis beam size for arrays
    
    Returns:
        Dictionary containing:
            - Processed data object with observatory configuration
            - Peak frequencies, intensities, and indices (sorted by intensity)
            - Full 3-sigma peak catalog for validation
            - RMS noise level and spectral resolution parameters
    """



    if peak_df is None and peak_df_3sigma is None: #did not input list of lines, will automatically find
        data = load_obs(specPath, type = 'txt') #load spectrum
        ll0, ul0 = find_limits(data.spectrum.frequency) #determine upper and lower limits of the spectrum
        freq_arr = data.spectrum.frequency #frequency array of spectrum
        int_arr = data.spectrum.Tb #intensity array of spectrum
        if rmsInp is None:
            rms = get_rms(int_arr)
            rms_full_arr = np.full_like(freq_arr, rms) #setting all values of rms array at rms value, maybe should update this

        else:
            #if type(rmsInp) == dict:
            if isinstance(rmsInp, dict):
                rms_full_arr = map_rms_to_spectrum(freq_arr, rmsInp) #making rms array with inputted dictionary values
            else:
                try:
                    rms_full_arr = np.full_like(freq_arr, rmsInp) #setting all values of rms array at rms value
                except:
                    raise ValueError(f"Your rms input was incorrect. Should either be a dictionary or a number.")

            rms = rmsInp
        freq_differences = np.diff(freq_arr)
        if freq_arr[1] - freq_arr[0] > 0:
            resolution = freq_arr[1] - freq_arr[0]
        else:
            resolution = np.median(freq_differences)
        #finding all peak frequencies and intensities in the spectrum at the inputted sigma level
        if rmsInp is None:
            peak_indices = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr), 2*dv_value_freq), sigma=sigOG, local_rms=True, rms=rmsInp) 
            if len(peak_indices) == 0:
                raise ValueError(f"Error: No peaks found at {sigOG} sigma or stronger. You may need to adjust the rms noise level using the rms_noise input parameter.")
            peak_freqs = data.spectrum.frequency[peak_indices]
            peak_ints = abs(data.spectrum.Tb[peak_indices])
        else:
            #if type(rmsInp) == dict: #checking if rms input was a dictionary
            if isinstance(rmsInp, dict):
                peak_indices = find_peaks_by_chunks(freq_arr,int_arr,rms_full_arr, res=resolution,dv_value_freq=dv_value_freq, sigma = sigOG)
                #print('len peak indices')
                #print(len(peak_indices))

                if len(peak_indices) == 0:
                    raise ValueError(f"Error: No peaks found at {sigOG} sigma or stronger. You may need to adjust the rms noise level using the rms_noise input parameter.")
                peak_freqs = data.spectrum.frequency[peak_indices]
                peak_ints = abs(data.spectrum.Tb[peak_indices])

            else:
                peak_indices = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr), 2*dv_value_freq), sigma=sigOG, local_rms=False, rms=rmsInp) 
                #print('len peak indices')
                #print(len(peak_indices))
                if len(peak_indices) == 0:
                    raise ValueError(f"Error: No peaks found at {sigOG} sigma or stronger. You may need to adjust the rms noise level using the rms_noise input parameter.")
                peak_freqs = data.spectrum.frequency[peak_indices]
                peak_ints = abs(data.spectrum.Tb[peak_indices])

        # Get indices that would sort peak_ints from high to low
        sort_indices = np.argsort(peak_ints)[::-1]  # [::-1] reverses for descending order

        # Sort both arrays using these indices
        spectrum_freqs = peak_freqs[sort_indices]
        spectrum_ints = peak_ints[sort_indices]

        # The sort_indices themselves are your mapping from original to sorted positions
        spectrum_indices = sort_indices

        print('')
        print('Number of peaks at ' + str(sigOG) + ' sigma significance in the spectrum: ' + str(len(peak_freqs)))
        print('')

        #storing all 3 sigma lines. Needed for future intensity checks
        if rmsInp is None:
            peak_indices_full = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_value_freq), sigma=3.0, local_rms = False, rms=rmsInp)
            if len(peak_indices_full) == 0:
                raise ValueError(f"Error: Error in peak finding.  Please manually adjust the rms noise level using the rms_noise input parameter.")
            peak_freqs_full = data.spectrum.frequency[peak_indices_full]
            peak_ints_full = abs(data.spectrum.Tb[peak_indices_full])
        else:
            #if type(rmsInp) == dict:
            if isinstance(rmsInp, dict):
                peak_indices_full = find_peaks_by_chunks(freq_arr,int_arr,rms_full_arr, res=resolution,dv_value_freq=dv_value_freq, sigma = 3.0)
                if len(peak_indices_full) == 0:
                    raise ValueError(f"Error: Error in peak finding. Please manually adjust the rms noise level using the rms_noise input parameter.")
                peak_freqs_full = data.spectrum.frequency[peak_indices_full]
                peak_ints_full = abs(data.spectrum.Tb[peak_indices_full])
                #print('len peak_indices_full')
                #print(len(peak_indices_full))

            else:
                peak_indices_full = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_value_freq), sigma=3.0, local_rms = False, rms=rmsInp)
                if len(peak_indices_full) == 0:
                    raise ValueError(f"Error: Error in peak finding. Please manually adjust the rms noise level using the rms_noise input parameter.")
                peak_freqs_full = data.spectrum.frequency[peak_indices_full]
                peak_ints_full = abs(data.spectrum.Tb[peak_indices_full])
                #print('len peak_indices_full')
                #print(len(peak_indices_full))
    
    elif peak_df is not None and peak_df_3sigma is None: #inputted list of lines but not list of 3 sigma peaks
        data = load_obs(specPath, type = 'txt') #load spectrum
        ll0, ul0 = find_limits(data.spectrum.frequency) #determine upper and lower limits of the spectrum
        freq_arr = data.spectrum.frequency #frequency array of spectrum
        int_arr = data.spectrum.Tb #intensity array of spectrum
        if freq_arr[1] - freq_arr[0] > 0:
            resolution = freq_arr[1] - freq_arr[0]
        else:
            resolution = np.median(freq_differences)
        peak_df = pd.read_csv(peak_df)
        peak_freqs = np.array(peak_df['frequency']) #loading inputted peak data
        peak_ints = np.array(peak_df['intensity'])
        # Get indices that would sort peak_ints from high to low
        sort_indices = np.argsort(peak_ints)[::-1]  # [::-1] reverses for descending order
        # Sort both arrays using these indices
        spectrum_freqs = peak_freqs[sort_indices]
        spectrum_ints = peak_ints[sort_indices]
        spectrum_indices = []
        peak_indices = []
        rms = rmsInp
        peak_indices_full = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_value_freq), sigma=3.0, local_rms = False, rms=rmsInp)
        if len(peak_indices_full) == 0:
                raise ValueError(f"Error: Error in peak finding. Please manually adjust the rms noise level using the rms_noise input parameter.")
        peak_freqs_full = data.spectrum.frequency[peak_indices_full]
        peak_ints_full = abs(data.spectrum.Tb[peak_indices_full])

        #print('')
        #print('Number of inputted peaks',len(peak_freqs))
        #print('')


    
    else: #inputted list of lines and list of 3 sigma peaks
        data = load_obs(specPath, type = 'txt') #load spectrum
        ll0, ul0 = find_limits(data.spectrum.frequency) #determine upper and lower limits of the spectrum
        freq_arr = data.spectrum.frequency #frequency array of spectrum
        int_arr = data.spectrum.Tb #intensity array of spectrum
        if freq_arr[1] - freq_arr[0] > 0:
            resolution = freq_arr[1] - freq_arr[0]
        else:
            resolution = np.median(freq_differences)
        peak_df = pd.read_csv(peak_df)
        peak_freqs = np.array(peak_df['frequency']) #loading inputted peak data
        peak_ints = np.array(peak_df['intensity'])
        # Get indices that would sort peak_ints from high to low
        sort_indices = np.argsort(peak_ints)[::-1]  # [::-1] reverses for descending order
        # Sort both arrays using these indices
        spectrum_freqs = peak_freqs[sort_indices]
        spectrum_ints = peak_ints[sort_indices]
        spectrum_indices = []
        peak_indices_full = []
        peak_indices = []
        peak_df_3sigma = pd.read_csv(peak_df_3sigma)
        peak_freqs_full = np.array(peak_df_3sigma['frequency'])
        peak_ints_full = np.array(peak_df_3sigma['intensity'])
        rms = rmsInp

        #print('')
        #print('Number of inputted peaks',len(peak_freqs))
        #print('')



    numVals = int(50/resolution) #number of resolution units to make 50 MHz

    #set the telescope parameters.
    if observation_type == '1':
        observatory1 = Observatory(sd=True, dish=bmaj)
    else:
        observatory1 = Observatory(sd=False, array=True,synth_beam = [bmaj,bmin])
    
    data.observatory = observatory1


    bandwidth = ul0[-1] - ll0[0]


    
    peak_data = {
        'data': data,
        'peak_indices': peak_indices,
        'peak_freqs': peak_freqs,
        'peak_ints': peak_ints,
        'peak_indices_full': peak_indices_full,
        'peak_freqs_full': peak_freqs_full,
        'peak_ints_full': peak_ints_full,
        'spectrum_freqs': spectrum_freqs,
        'spectrum_ints': spectrum_ints,
        'spectrum_indices': spectrum_indices,
        'numVals': numVals,
        'rms': rms,
        'freq_arr': freq_arr,
        'int_arr': int_arr,
        'rms_full_arr': rms_full_arr
    }

    return peak_data


def map_rms_to_spectrum(freq_arr, rms_dict):
    """
    Map RMS noise levels from frequency ranges to a spectrum.
    Handles gaps by using the higher RMS of surrounding chunks.
    Extends first/last RMS values to cover spectrum edges.
    
    Args:
        freq_arr: Array of frequencies
        rms_dict: Dictionary with (freq_min, freq_max) tuples as keys, 
                  RMS values as values
    
    Returns:
        Array of RMS values corresponding to each frequency
    """
    rms_arr = np.full_like(freq_arr, np.nan)
    
    # First pass: fill all defined ranges
    for (freq_min, freq_max), rms in rms_dict.items():
        mask = (freq_arr >= freq_min) & (freq_arr <= freq_max)
        rms_arr[mask] = rms
    
    # Find contiguous NaN regions
    nan_mask = np.isnan(rms_arr)
    
    if np.any(nan_mask):
        # Find where NaN regions start and stop
        nan_diff = np.diff(np.concatenate(([False], nan_mask, [False])).astype(int))
        gap_starts = np.where(nan_diff == 1)[0]
        gap_ends = np.where(nan_diff == -1)[0]
        
        # Process each gap
        for gap_start, gap_end in zip(gap_starts, gap_ends):
            # Find RMS of chunk immediately before gap
            if gap_start > 0:
                left_rms = rms_arr[gap_start - 1]
            else:
                left_rms = None
            
            # Find RMS of chunk immediately after gap
            if gap_end < len(rms_arr):
                right_rms = rms_arr[gap_end]
            else:
                right_rms = None
            
            # Fill gap based on what's available
            if left_rms is not None and right_rms is not None:
                # Gap in middle - use max of adjacent chunks
                fill_value = max(left_rms, right_rms)
            elif left_rms is not None:
                # Gap at end - extend left
                fill_value = left_rms
            elif right_rms is not None:
                # Gap at beginning - extend right
                fill_value = right_rms
            else:
                # Should never happen if rms_dict has any entries
                fill_value = 0.0
            
            rms_arr[gap_start:gap_end] = fill_value
    
    return rms_arr

def find_peaks_by_chunks(freq_arr, int_arr, rms_arr, res, dv_value_freq, is_sim=False, sigma=3, kms=True):
    """
    Find peaks by running find_peaks on each RMS chunk separately.
    
    This uses the RMS array to identify regions with constant RMS,
    then runs find_peaks on each region with its specific RMS value.
    
    Args:
        freq_arr: Array of frequency values (MHz or GHz)
        int_arr: Array of intensity values (Kelvin or Jy)
        rms_arr: Array of RMS noise values (same length as freq_arr/int_arr)
        res: Spectral resolution (frequency units)
        min_sep: Minimum separation between peaks (km/s or frequency units)
        is_sim: If True, returns all peaks without noise filtering
        sigma: Significance threshold (multiples of RMS noise)
        kms: If True, operates in velocity space; if False, frequency space
        
    Returns:
        Array of indices corresponding to detected peaks across all chunks
    """
    from ..utils.astro_utils import find_peaks_local
    
    all_peak_indices = []
    
    # Find where RMS changes (chunk boundaries)
    rms_changes = np.concatenate(([True], np.diff(rms_arr) != 0, [True]))
    chunk_starts = np.where(rms_changes)[0][:-1]
    chunk_ends = np.where(rms_changes)[0][1:]
    
    # Process each chunk
    for start_idx, end_idx in zip(chunk_starts, chunk_ends):
        chunk_freq = freq_arr[start_idx:end_idx]
        chunk_int = int_arr[start_idx:end_idx]
        chunk_rms = rms_arr[start_idx]  # RMS is constant within chunk

        #print('chunk_freq')
        #print(chunk_freq[0])
        #print(chunk_freq[-1])
        #print('chunk_rms')
        #print(chunk_rms)
        #print('')
        
        # Run find_peaks on this chunk with its specific RMS
        #chunk_peaks = find_peaks_local(chunk_freq, chunk_int, res, min_sep, 
        #                         is_sim=is_sim, sigma=sigma, kms=kms, rms=chunk_rms)

        chunk_peaks = find_peaks_local(chunk_freq, chunk_int, res=res, min_sep=max(res * ckm / np.amax(chunk_freq), 2*dv_value_freq), sigma=sigma, local_rms=False, rms=chunk_rms)
        
        # Map chunk-relative indices back to original array indices
        if len(chunk_peaks) > 0:
            original_indices = start_idx + chunk_peaks
            all_peak_indices.extend(original_indices)

            #print(chunk_freq)
            #print(chunk_peaks)
    
    return np.asarray(sorted(all_peak_indices))