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
    min_separation = resolution * ckm / np.amax(freq_arr) #minimum separation between peaks in km/s

    #setting telescope parameters
    #print(observation_type)
    if observation_type == '1':
        observatory1 = Observatory(sd=True, dish=bmaj)
    else:
        observatory1 = Observatory(sd=False, array=True,synth_beam = [bmaj,bmin])

    data.observatory = observatory1
    
    bandwidth = ul0[-1] - ll0[0]
    
    if rmsInp == None:
        rms = get_rms(int_arr)
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
        if rmsInp == None:
            rms = get_rms(int_arr)
        else:
            rms = rmsInp
        freq_differences = np.diff(freq_arr)
        if freq_arr[1] - freq_arr[0] > 0:
            resolution = freq_arr[1] - freq_arr[0]
        else:
            resolution = np.median(freq_differences)
        #finding all peak frequencies and intensities in the spectrum at the inputted sigma level
        if rmsInp == None:
            peak_indices = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr), 2*dv_value_freq), sigma=sigOG, local_rms=True, rms=rmsInp) 
            peak_freqs = data.spectrum.frequency[peak_indices]
            peak_ints = abs(data.spectrum.Tb[peak_indices])
        else:
            peak_indices = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr), 2*dv_value_freq), sigma=sigOG, local_rms=False, rms=rmsInp) 
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
        if rmsInp == None:
            peak_indices_full = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_value_freq), sigma=3.0, local_rms = False, rms=rmsInp)
            peak_freqs_full = data.spectrum.frequency[peak_indices_full]
            peak_ints_full = abs(data.spectrum.Tb[peak_indices_full])
        else:
            peak_indices_full = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_value_freq), sigma=3.0, local_rms = False, rms=rmsInp)
            peak_freqs_full = data.spectrum.frequency[peak_indices_full]
            peak_ints_full = abs(data.spectrum.Tb[peak_indices_full])
    
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
        peak_freqs_full = data.spectrum.frequency[peak_indices_full]
        peak_ints_full = abs(data.spectrum.Tb[peak_indices_full])

        print('')
        print('Number of inputted peaks',len(peak_freqs))
        print('')


    
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

        print('')
        print('Number of inputted peaks',len(peak_freqs))
        print('')



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
    }

    return peak_data