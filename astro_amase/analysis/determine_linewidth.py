"""
Identify spectral peaks and calculate the median linewidth.

This routine performs the following steps:

1. **Peak Detection**: Peaks are identified in the input frequency (`freq_arr`) 
and intensity (`int_arr`) arrays using the `find_peaks_local` function. The detection
considers the specified resolution, minimum separation between peaks, and a 
Gaussian smoothing parameter (`sigma`).

2. **Peak Filtering**: Any peaks that are within a certain proximity to lines 
matched with a template are removed to prevent duplication or interference.

3. **Peak Sorting**: The remaining peaks are sorted by intensity from strongest 
to weakest, with arrays of peak frequencies, intensities, and indices 
prepared for further analysis.

4. **Initial Gaussian Fitting (First Pass)**: For each peak, a local chunk of 
data around the peak is extracted, and a Gaussian function is fit to the 
intensity profile. The fit yields amplitude (`a`), center frequency (`mu`), 
and width (`sigma`). Peaks too close to previously fitted peaks are skipped 
to avoid double-counting. The FWHM for each peak is calculated as `2.355 * sigma`.

5. **Median FWHM Estimation**: The median FWHM from the first fitting pass is 
computed and used as an initial guess for the second pass.

6. **Refined Gaussian Fitting (Second Pass)**: The fitting procedure is repeated 
with the median FWHM as an initial estimate, yielding refined peak positions 
and linewidths. The FWHM is converted to velocity widths (`velFWHM`) using the 
Doppler relation.

7. **Output**: The result is a set of robust peak frequencies, intensities, FWHM, 
and velocity widths, suitable for further spectral analysis and line 
characterization.
"""

from scipy.optimize import curve_fit
import statistics
import numpy as np
from ..constants import ckm
from ..utils.astro_utils import find_peaks_local, sortTupleArray



def gaussian(x, a, mu, sigma):
    """
    Calculates the value of a Gaussian function.
    Parameters:
        x (float or np.ndarray): The input value(s) at which to evaluate the Gaussian.
        a (float): The amplitude of the Gaussian.
        mu (float): The mean (center) of the Gaussian.
        sigma (float): The standard deviation (width) of the Gaussian.
    Returns:
        float or np.ndarray: The value(s) of the Gaussian function at x.
    """

    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def find_linewidth(freq_arr, int_arr, resolution, sigOG, data, rmsInp):  
    """
    Determine median spectral linewidth through two-pass Gaussian fitting of detected peaks.
    
    Implements a robust linewidth estimation algorithm that:
    1. Identifies all spectral peaks above noise threshold
    2. Sorts peaks by intensity (strongest first)
    3. Fits Gaussian profiles to each peak using local data chunks
    4. Calculates median FWHM from first-pass fits
    5. Refines fits using median FWHM as initial guess
    6. Converts frequency FWHM to velocity width via Doppler relation
    7. Determines whether hyperfine structure should be considered
    
    
    Parameters
    ----------
    freq_arr : array_like
        Full frequency array (MHz) from the observation.
    int_arr : array_like
        Full intensity array (brightness temperature) corresponding
        to freq_arr.
    resolution : float
        Spectral resolution (MHz) of the observation, used for peak detection
        and determining fit window size.
    sigOG : float
        Sigma cutoff of considered lines.
    data : Observation
        molsim Observation object containing spectrum attributes:
        - data.spectrum.frequency: frequency array
        - data.spectrum.Tb: brightness temperature array
    rmsInp : float
        Root-mean-square noise level of the spectrum for peak significance
        thresholding in the local peak finder.
    
    Returns
    -------
    dv_value : float
        Median velocity linewidth (km/s) determined from Gaussian fits.
        Represents the typical FWHM of spectral lines in velocity units.
    dv_value_freq : float
        Median frequency linewidth (MHz), with minimum threshold of 0.15 MHz
        to prevent excessively narrow search windows. Used for candidate
        molecule matching and frequency analysis.
    consider_hyperfine : bool
        Flag indicating whether hyperfine catalogs should be
        included in analysis. Set to True if dv_value_freq < 1 MHz, as
        hyperfine splitting becomes significant at narrow linewidths.
    
    Algorithm Details
    -----------------
    **Peak Detection:**
    - Uses find_peaks_local
    
    **First-Pass Fitting:**
    - Extracts 50 MHz window around each peak (±numVals resolution elements)
    - Initial guess: [peak_intensity, peak_frequency, 3 MHz]
    - Fits Gaussian using scipy.optimize.curve_fit
    - Rejects fits with center within 0.1 MHz of previous fits (avoids duplicates)
    - Calculates FWHM = 2.355 x σ from fitted sigma parameter
    - Computes median FWHM for use in second pass
    
    **Second-Pass Fitting:**
    - Repeats fitting procedure with refined initial guess
    - Initial sigma set to median FWHM/2.355 from first pass
    - Converts FWHM to velocity: v_FWHM = (Δν_FWHM / ν) × c
    - Uses speed of light c = 299792.458 km/s
    - Computes final median velocity and frequency linewidths
    
    **Quality Control:**
    - Skips peaks where Gaussian fitting fails to converge
    - Prevents duplicate measurements from closely spaced peaks
    - Applies 0.15 MHz minimum to frequency linewidth.
    
    Notes
    -----
    - The 0.15 MHz minimum frequency ensures that line width is not too narrow
    and uncertain catalogs are not considered.
    
    """
    print('finding linewidth!')

    #finding the peaks in the spectrum
    peak_indices = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),1), sigma=sigOG, local_rms=True, rms=rmsInp)
    peak_freqs = data.spectrum.frequency[peak_indices]
    peak_ints = abs(data.spectrum.Tb[peak_indices])
    
    delFreqs = []

    peak_freqs2 = [peak_freqs[i] for i in range(len(peak_freqs)) if peak_freqs[i] not in delFreqs]
    peak_ints2 = [peak_ints[i] for i in range(len(peak_freqs)) if peak_freqs[i] not in delFreqs]
    peak_indices2 = [peak_indices[i] for i in range(len(peak_freqs)) if peak_freqs[i] not in delFreqs]
    #store all peaks by from strongest-to-weakest intensity
    combPeaks = [(peak_freqs2[i], peak_ints2[i], peak_indices2[i]) for i in range(len(peak_freqs2))]
    scp = sortTupleArray(combPeaks)
    scp.reverse()
    #storing peak frequencies and intensities in separate arrays
    peak_freqs = np.array([i[0] for i in scp])
    peak_ints = np.array([i[1] for i in scp])
    peak_indices = np.array([i[2] for i in scp])
    numVals = int(50 / resolution) #calculate number of resolution units to make 50 MHz
    allMu = []
    alreadyMu = []
    fwhmList = []

    for i in range(len(peak_indices)): #loop through all peaks
        #get a chunk of the frequency and intensity arrays around the peak
        freqChunk = freq_arr[peak_indices[i] - numVals:peak_indices[i] + numVals]
        intChunk = int_arr[peak_indices[i] - numVals:peak_indices[i] + numVals]

        #initial guess for the Gaussian fit parameters
        initial_guess = [peak_ints[i], peak_freqs[i], 3]
        try:
            #fit gaussian function to the chunk of data
            params, covariance = curve_fit(gaussian, freqChunk, intChunk, p0=initial_guess)

            # Extract the parameters
            a, mu, sigma = params

            nearOld = False
            # Check if the mu value is close to any previously found mu values
            for n in alreadyMu:
                if abs(mu - n) < 0.1:
                    nearOld = True
            if nearOld == False:
                # if abs(mu - peak_freqs[i]) <= 2:
                # fwhm = 2 * math.sqrt(2 * math.log(2)) * abs(sigma)
                fwhm = 2.355 * sigma #convert Gaussian sigma to FWHM
                fwhmList.append(fwhm)
                allMu.append(mu)
                alreadyMu.append(mu)
        except:
            # print('gaussian failed to converge for line at ' + str(peak_freqs[i]) + ' MHz')
            continue


    fwhmAttempt1 = statistics.median(fwhmList)


    #Now we will do a second attempt at finding the linewidth, using the median FWHM from the first attempt, just to further refine the linewidth estimate.
    alreadyMu = []
    allMu = []
    fwhmList = []
    velList = []
    for i in range(len(peak_indices)):
        #get a chunk of the frequency and intensity arrays around the peak
        freqChunk = freq_arr[peak_indices[i] - numVals:peak_indices[i] + numVals]
        intChunk = int_arr[peak_indices[i] - numVals:peak_indices[i] + numVals]
        #initial guess, including the median FWHM from the first attempt
        initial_guess = [peak_ints[i], peak_freqs[i], fwhmAttempt1/2.355]


        try:
            # Fit the Gaussian function to the data
            params, covariance = curve_fit(gaussian, freqChunk, intChunk, p0=initial_guess)

            # Extract the parameters
            a, mu, sigma = params
            nearOld = False
            for n in alreadyMu:
                if abs(mu - n) < 0.1:
                    nearOld = True
            if nearOld == False:
                fwhm = 2.355 * sigma #convert Gaussian sigma to FWHM
                velFWHM = (fwhm / mu) * 299792.458 # convert FWHM to velocity in km/s
                velList.append(velFWHM)
                fwhmList.append(fwhm)
                allMu.append(mu)
                alreadyMu.append(mu)
        except:
            # print('gaussian failed to converge for line at ' + str(peak_freqs[i]) + ' MHz')
            continue


    fwhmAttempt2 = statistics.median(fwhmList)
    print('Velocity Linewidth (km/s)')
    velMed = statistics.median(velList)
    print(round(velMed, 2))

    dv_value = velMed #storing the velocity linewidth as dv_value

    if fwhmAttempt2 < 0.15: #if the frequency linewidth is less than 0.15 MHz, we set the dv_value_freq to 0.15 so that its not too narrow
        dv_value_freq = 0.15
    else:
        dv_value_freq = fwhmAttempt2

    print('Frequency Linewidth (MHz)')
    print(round(dv_value_freq,2))
    consider_hyperfine = False
    #only consider hyperfine splitting if the linewidth is less than 1 MHz
    if dv_value_freq < 1:
        consider_hyperfine = True

    return dv_value, dv_value_freq, consider_hyperfine


def find_linewidth_standalone(freq_arr, int_arr, resolution, sigOG, data, rmsInp):  
    """
    Determine median spectral linewidth through two-pass Gaussian fitting of detected peaks.
    Used for the get_linewidth function within main.py. Has minor changes from the main find_linewidth function.
    
    Implements a robust linewidth estimation algorithm that:
    1. Identifies all spectral peaks above noise threshold
    2. Sorts peaks by intensity (strongest first)
    3. Fits Gaussian profiles to each peak using local data chunks
    4. Calculates median FWHM from first-pass fits
    5. Refines fits using median FWHM as initial guess
    6. Converts frequency FWHM to velocity width via Doppler relation
    7. Determines whether hyperfine structure should be considered
    
    
    Parameters
    ----------
    freq_arr : array_like
        Full frequency array (MHz) from the observation.
    int_arr : array_like
        Full intensity array (brightness temperature) corresponding
        to freq_arr.
    resolution : float
        Spectral resolution (MHz) of the observation, used for peak detection
        and determining fit window size.
    sigOG : float
        Sigma cutoff of considered lines.
    data : Observation
        molsim Observation object containing spectrum attributes:
        - data.spectrum.frequency: frequency array
        - data.spectrum.Tb: brightness temperature array
    rmsInp : float
        Root-mean-square noise level of the spectrum for peak significance
        thresholding in the local peak finder.
    
    Returns
    -------
    dv_value : float
        Median velocity linewidth (km/s) determined from Gaussian fits.
        Represents the typical FWHM of spectral lines in velocity units.
    dv_value_freq : float
        Median frequency linewidth (MHz), with minimum threshold of 0.15 MHz
        to prevent excessively narrow search windows. Used for candidate
        molecule matching and frequency analysis.
    consider_hyperfine : bool
        Flag indicating whether hyperfine catalogs should be
        included in analysis. Set to True if dv_value_freq < 1 MHz, as
        hyperfine splitting becomes significant at narrow linewidths.
    
    Algorithm Details
    -----------------
    **Peak Detection:**
    - Uses find_peaks_local
    
    **First-Pass Fitting:**
    - Extracts 50 MHz window around each peak (±numVals resolution elements)
    - Initial guess: [peak_intensity, peak_frequency, 3 MHz]
    - Fits Gaussian using scipy.optimize.curve_fit
    - Rejects fits with center within 0.1 MHz of previous fits (avoids duplicates)
    - Calculates FWHM = 2.355 x σ from fitted sigma parameter
    - Computes median FWHM for use in second pass
    
    **Second-Pass Fitting:**
    - Repeats fitting procedure with refined initial guess
    - Initial sigma set to median FWHM/2.355 from first pass
    - Converts FWHM to velocity: v_FWHM = (Δν_FWHM / ν) × c
    - Uses speed of light c = 299792.458 km/s
    - Computes final median velocity and frequency linewidths
    
    **Quality Control:**
    - Skips peaks where Gaussian fitting fails to converge
    - Prevents duplicate measurements from closely spaced peaks
    - Applies 0.15 MHz minimum to frequency linewidth.
    
    Notes
    -----
    - The 0.15 MHz minimum frequency ensures that line width is not too narrow
    and uncertain catalogs are not considered.
    
    """
    print('finding linewidth!')

    #finding the peaks in the spectrum
    peak_indices = find_peaks_local(freq_arr, int_arr, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),1), sigma=sigOG, local_rms=True, rms=rmsInp)
    peak_freqs = data.spectrum.frequency[peak_indices]
    peak_ints = abs(data.spectrum.Tb[peak_indices])
    
    delFreqs = []

    peak_freqs2 = [peak_freqs[i] for i in range(len(peak_freqs)) if peak_freqs[i] not in delFreqs]
    peak_ints2 = [peak_ints[i] for i in range(len(peak_freqs)) if peak_freqs[i] not in delFreqs]
    peak_indices2 = [peak_indices[i] for i in range(len(peak_freqs)) if peak_freqs[i] not in delFreqs]
    #store all peaks by from strongest-to-weakest intensity
    combPeaks = [(peak_freqs2[i], peak_ints2[i], peak_indices2[i]) for i in range(len(peak_freqs2))]
    scp = sortTupleArray(combPeaks)
    scp.reverse()
    #storing peak frequencies and intensities in separate arrays
    peak_freqs = np.array([i[0] for i in scp])
    peak_ints = np.array([i[1] for i in scp])
    peak_indices = np.array([i[2] for i in scp])
    numVals = int(50 / resolution) #calculate number of resolution units to make 50 MHz
    allMu = []
    alreadyMu = []
    fwhmList = []

    for i in range(len(peak_indices)): #loop through all peaks
        #get a chunk of the frequency and intensity arrays around the peak
        freqChunk = freq_arr[peak_indices[i] - numVals:peak_indices[i] + numVals]
        intChunk = int_arr[peak_indices[i] - numVals:peak_indices[i] + numVals]

        #initial guess for the Gaussian fit parameters
        initial_guess = [peak_ints[i], peak_freqs[i], 3]
        try:
            #fit gaussian function to the chunk of data
            params, covariance = curve_fit(gaussian, freqChunk, intChunk, p0=initial_guess)

            # Extract the parameters
            a, mu, sigma = params

            nearOld = False
            # Check if the mu value is close to any previously found mu values
            for n in alreadyMu:
                if abs(mu - n) < 0.1:
                    nearOld = True
            if nearOld == False:
                # if abs(mu - peak_freqs[i]) <= 2:
                # fwhm = 2 * math.sqrt(2 * math.log(2)) * abs(sigma)
                fwhm = 2.355 * sigma #convert Gaussian sigma to FWHM
                fwhmList.append(fwhm)
                allMu.append(mu)
                alreadyMu.append(mu)
        except:
            # print('gaussian failed to converge for line at ' + str(peak_freqs[i]) + ' MHz')
            continue


    fwhmAttempt1 = statistics.median(fwhmList)


    #Now we will do a second attempt at finding the linewidth, using the median FWHM from the first attempt, just to further refine the linewidth estimate.
    alreadyMu = []
    allMu = []
    fwhmList = []
    velList = []
    for i in range(len(peak_indices)):
        #get a chunk of the frequency and intensity arrays around the peak
        freqChunk = freq_arr[peak_indices[i] - numVals:peak_indices[i] + numVals]
        intChunk = int_arr[peak_indices[i] - numVals:peak_indices[i] + numVals]
        #initial guess, including the median FWHM from the first attempt
        initial_guess = [peak_ints[i], peak_freqs[i], fwhmAttempt1/2.355]


        try:
            # Fit the Gaussian function to the data
            params, covariance = curve_fit(gaussian, freqChunk, intChunk, p0=initial_guess)

            # Extract the parameters
            a, mu, sigma = params
            nearOld = False
            for n in alreadyMu:
                if abs(mu - n) < 0.1:
                    nearOld = True
            if nearOld == False:
                fwhm = 2.355 * sigma #convert Gaussian sigma to FWHM
                velFWHM = (fwhm / mu) * 299792.458 # convert FWHM to velocity in km/s
                velList.append(velFWHM)
                fwhmList.append(fwhm)
                allMu.append(mu)
                alreadyMu.append(mu)
        except:
            # print('gaussian failed to converge for line at ' + str(peak_freqs[i]) + ' MHz')
            continue


    fwhmAttempt2 = statistics.median(fwhmList)
    print('Velocity Linewidth (km/s)')
    velMed = statistics.median(velList)
    print(round(velMed, 2))

    dv_value = velMed #storing the velocity linewidth as dv_value

    dv_value_freq = fwhmAttempt2

    print('Frequency Linewidth (MHz)')
    print(round(dv_value_freq,2))
    consider_hyperfine = False
    #only consider hyperfine splitting if the linewidth is less than 1 MHz
    if dv_value_freq < 1:
        consider_hyperfine = True

    return dv_value, dv_value_freq, consider_hyperfine