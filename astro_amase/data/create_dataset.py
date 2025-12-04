"""
Functions to create dataset of molecular candidates for each line.
For each line in the spectrum, it determies all possible molecular candidates with
known transitions sufficiently nearby and saves this as a .csv file.

It finds all transitions in CDMS/JPL (stored in the transitions_database.pkl.gz file)
that are within 0.5*dV of each line in the spectrum. It also simulates the spectra of these
molecules at the determined observational parameters using molsim and stores these spectra.
"""


import os
import shutil
import pandas as pd
import pickle
import gzip
import numpy as np
import csv
import time
from ..constants import maxMols,ckm, vicgae_model
from ..utils.molsim_classes import Continuum, Source, Simulation
from ..utils.molsim_utils import find_peaks
#import warnings
#warnings.filterwarnings('ignore')


def apply_vlsr_shift(frequency, vlsr):
    """
    Apply Doppler shift correction to frequency array based on source velocity.
    
    Converts rest frame frequencies to observer frame by applying the velocity
    correction: f_obs = f_rest * (1 - v_lsr/c), where c is the speed of light.
    
    Parameters
    ----------
    frequency : array_like
        Rest frame frequencies in MHz.
    vlsr : float
        Line-of-sight velocity (km/s) of the source relative to the local
        standard of rest (LSR).
    
    Returns
    -------
    shifted_frequency : ndarray
        Observer frame frequencies in MHz after Doppler correction.
    
    Notes
    -----
    Uses the speed of light constant `ckm` from astro_constants module.
    """

    return frequency - vlsr * frequency / ckm

def create_folders_load_data(direc, tempInput):
    """
    Initialize directory structure and load molecular catalog metadata for dataset creation.
    
    Creates working directories for storing catalog data and
    simulated spectra. Loads CDMS and JPL molecular database information including
    molecular formulas, names, catalog tags, SMILES identifiers, and file paths.
    Removes any existing directories to ensure clean state.
    
    Directory Structure Created:
    - `splatalogue_catalogues_{temp}/`: Stores simulated spectra for all candidates
    - `splatalogue_catalogues_{temp}/catalogues/`: Catalog metadata
    - `local_catalogs_{temp}/`: Local catalog files
    - `added_catalogs_{temp}/`: Additional user-provided catalogs
    
    Parameters
    ----------
    direc : str
        Base directory containing molecular database files and subdirectories.
        Expected files:
        - 'all_cdms_final_official.csv': CDMS database metadata
        - 'all_jpl_final_official.csv': JPL database metadata
        Expected subdirectories:
        - 'cdms_pkl/': Pickled CDMS molecule objects
        - 'jpl_pkl/': Pickled JPL molecule objects
    tempInput : float
        Excitation temperature (K) used for simulations. Directory names are
        suffixed with integer value of this temperature.
    
    Returns
    -------
    allLoaded : dict
        Nested dictionary containing all loaded catalog data and path information:
        
        **'cdms' sub-dictionary:**
        - 'directory': path to CDMS pickle files
        - 'dataframe': full CDMS DataFrame
        - 'forms': list of molecular formulas
        - 'names': list of Splatalogue names
        - 'tags': list of catalog tag numbers
        - 'smiles': list of SMILES strings
        - 'mols': list of molecule names
        
        **'jpl' sub-dictionary:**
        - 'directory': path to JPL pickle files
        - 'dataframe': full JPL DataFrame
        - 'forms': list of molecular formulas
        - 'names': list of Splatalogue names
        - 'tags': list of catalog tag numbers
        - 'smiles': list of SMILES strings
        - 'mols': list of molecule names
        
        **'paths' sub-dictionary:**
        - 'splat': temperature-specific Splatalogue directory
        - 'splat_catalogues': catalog subdirectory
        - 'extra': added catalogs directory
        - 'local': local catalogs directory
        - 'splat_og': original Splatalogue directory
        - 'splat_cat_og': original catalog subdirectory
        - 'extra_og': original added catalogs directory
        - 'local_og': original local catalogs directory
    
    Notes
    -----
    - Existing temperature-specific directories are removed before creation to
      prevent data contamination from previous runs
    - All molecular catalog data is loaded into memory for fast access during
      dataset generation
    """

    pathSplatOG = os.path.join(direc, 'splatalogue_catalogues')
    pathExtraOG = os.path.join(direc, 'added_catalogs')
    pathLocalOG = os.path.join(direc, 'local_catalogs')
    pathSplatCatOG = os.path.join(pathSplatOG, 'catalogues')

    t = tempInput
    pathLocal = os.path.join(direc, 'local_catalogs_' + str(int(t)))
    pathSplat = os.path.join(direc, 'splatalogue_catalogues_' + str(int(t)))
    pathExtra = os.path.join(direc, 'added_catalogs_' + str(int(t)))

    if os.path.isdir(pathSplat):
        shutil.rmtree(pathSplat)

    if os.path.isdir(pathLocal):
        shutil.rmtree(pathLocal)

    if os.path.isdir(pathExtra):
        shutil.rmtree(pathExtra)


    os.mkdir(pathSplat)
    os.mkdir(pathLocal)
    pathSplatCat = os.path.join(pathSplat, 'catalogues')
    os.mkdir(pathSplatCat)
    os.mkdir(pathExtra)

    #loading CDMS and JPL dataframes for dataset generation
    cdmsDirec = os.path.join(direc, 'cdms_pkl/')
    cdmsFullDF = pd.read_csv(os.path.join(direc, 'all_cdms_final_official.csv'))
    cdmsForms = list(cdmsFullDF['splat form'])
    cdmsNames = list(cdmsFullDF['splat name'])
    cdmsTags = list(cdmsFullDF['tag'])
    cdmsSmiles = list(cdmsFullDF['smiles'])
    cdmsMols = list(cdmsFullDF['mol'])
    #cdmsTags = [t[1:-4] for t in cdmsTags]

    jplDirec = os.path.join(direc, 'jpl_pkl/')
    jplFullDF = pd.read_csv(os.path.join(direc, 'all_jpl_final_official.csv'))
    jplForms = list(jplFullDF['splat form'])
    jplNames = list(jplFullDF['splat name'])
    jplTags = list(jplFullDF['save tag'])
    jplSmiles = list(jplFullDF['smiles'])
    jplMols = list(jplFullDF['name'])


    allLoaded = {

        'cdms': {
            'directory': cdmsDirec,
            'dataframe': cdmsFullDF,
            'forms': cdmsForms,
            'names': cdmsNames,
            'tags': cdmsTags,
            'smiles': cdmsSmiles,
            'mols': cdmsMols
        },
        'jpl': {
            'directory': jplDirec,
            'dataframe': jplFullDF,
            'forms': jplForms,
            'names': jplNames,
            'tags': jplTags,
            'smiles': jplSmiles,
            'mols': jplMols
        },
        'paths': {
            'splat': pathSplat,
            'splat_catalogues': pathSplatCat,
            'extra': pathExtra,
            'local': pathLocal,
            'splat_og': pathSplatOG,
            'splat_cat_og': pathSplatCatOG,
            'extra_og': pathExtraOG,
            'local_og': pathLocalOG
        }
    }

    return allLoaded



def create_full_dataset(direc, spectrum_freqs, spectrum_ints,dv_value, dataScrape,vlsr_value, dv_value_freq, consider_hyperfine, user_temp,sourceSize,ll0,ul0, freq_arr,resolution, cont_temp, ignore_mol_list):
    """
    Generate comprehensive molecular candidate dataset for all spectral lines with simulated spectra.
    
    This is the main dataset creation function that:
    1. Queries CDMS/JPL transition databases for candidates within ±0.5 x dV (in MHz) of each observed line
    2. Filters duplicates and applies quality control (hyperfine handling, CDMS/JPL priorities)
    3. Simulates spectra for all candidate molecules at determined observational parameters
    4. Creates CSV dataset with candidate information and simulated peak frequencies/intensities
    5. Exports catalog metadata and molecule-SMILES mapping
    
    Candidate Selection Criteria:
    - Transition frequency within ±0.5 x dV_freq of observed line (VLSR-corrected)
    - Valid SMILES identifier (excludes molecules needing SMILES data)
    - Not a duplicate between CDMS and JPL (CDMS takes priority)
    - Hyperfine catalogs excluded unless consider_hyperfine=True (based on linewdith)
    - Must produce detectable peaks in simulated spectrum at given parameters
    
    Parameters
    ----------
    direc : str
        Base directory containing 'transitions_database.pkl.gz' and output location.
    spectrum_freqs : array_like
        Observed frequencies (MHz) of identified spectral lines.
    spectrum_ints : array_like
        Observed intensities corresponding to spectrum_freqs.
    dv_value : float
        Line width (km/s) determined from Gaussian fitting.
    dataScrape : Observation
        molsim Observation object containing full spectrum frequency grid and data.
    vlsr_value : float
        Source velocity (km/s) relative to LSR for Doppler correction.
    dv_value_freq : float
        Line width in frequency units (MHz) for search window calculation.
    consider_hyperfine : bool
        If True, include hyperfine catalogs (tag > 200000).
        If False, exclude hyperfine catalogs.
    user_temp : float
        Excitation temperature (K) for spectral simulations.
    sourceSize : float
        Source diameter for spectral simulations.
    ll0 : float
        Lower frequency bound array (MHz) for simulations.
    ul0 : float
        Upper frequency bound array (MHz) for simulations.
    freq_arr : array_like
        Full frequency array for resolution calculations.
    resolution : float
        Spectral resolution for peak finding in simulations.
    cont_temp : float
        Continuum temperature (K) for background radiation field.
    
    Returns
    -------
    all_loaded : dict
        Dictionary of loaded catalog data and paths (from create_folders_load_data).
    noCanFreq : list of float
        Frequencies of lines with no molecular candidates found.
    noCanInts : list of float
        Intensities of lines with no molecular candidates found.
    splatDict : dict
        Dictionary mapping (molecule_name, database) tuples to (frequencies, intensities)
        tuples from simulated spectra. Used for rapid candidate lookup during line assignment.
    cont_obj : Continuum
        molsim Continuum object used for simulations, returned for reuse.
    
    File Outputs
    ------------
    Creates multiple output files in the specified directory:
    
    **'dataset.csv':**
    Initial dataset with all candidate information before final quality control.
    
    **'dataset_final.csv':**
    Final quality-controlled dataset. Each row represents one observed line with columns:
    - obs frequency, obs intensity (first 2 columns)
    - For each of maxMols candidate slots (9 columns per candidate):
      * mol name: molecule identifier
      * mol form: molecular formula
      * smiles: SMILES structural notation
      * frequency: catalog transition frequency (MHz)
      * uncertainty: frequency uncertainty (MHz)
      * isotope count: isotopologue identifier
      * quantum number: quantum state information 
      * catalog tag: unique catalog identifier
      * linelist: source database ('CDMS' or 'JPL')
    - 'NA' fills empty candidate slots
    
    **'mol_smiles.csv':**
    Mapping of all unique molecules to their SMILES identifiers.
    
    **'splatalogue_catalogues_{temp}/{index}.csv':**
    Individual CSV files for each candidate molecule containing simulated
    peak frequencies and intensities.
    
    **'splatalogue_catalogues_{temp}/catalogues/catalog_list.csv':**
    Metadata linking catalog indices to molecular formulas, tags, and databases.
    
    Notes
    -----
    - Uses transitions_database.pkl.gz containing pre-compiled CDMS/JPL transitions
    - Frequency matching uses binary search (np.searchsorted) for efficiency
    - Duplicate removal prioritizes: CDMS over JPL, base transitions over hyperfine
    - Quality control removes candidates that fail to produce peaks in simulation
    - Maximum number of candidates per line set by maxMols constant (from astro_constants)
    - Simulations use Gaussian line profiles with parameters from observational analysis
    - Lines without any valid candidates are reported and excluded from dataset
    
    """
    tickScrape = time.perf_counter()
    all_loaded = create_folders_load_data(direc, user_temp)
    
    with gzip.open(os.path.join(direc, "transitions_database.pkl.gz"), "rb") as f: # Load the transitions database
        database_freqs, database_errs, database_tags, database_lists, database_smiles, database_names, database_isos, database_vibs, database_forms = pickle.load(f)

    splatDict = {}
    ignoreMolDict = {}
    for i in ignore_mol_list:
        ignoreMolDict[i] = False
    ignore_mol_list = set(ignore_mol_list)    
    #lists to store frequencies and intensities of lines that dont have any molecular candidates
    noCanFreq =[] 
    noCanInts = []

    firstLine = ['obs frequency', 'obs intensity']

    # Making the dataset file.
    for i in range(maxMols):
        idx = i + 1
        #creating the header for the dataset
        firstLine.append('mol name ' + str(idx))
        firstLine.append('mol form ' + str(idx))
        firstLine.append('smiles ' + str(idx))
        firstLine.append('frequency ' + str(idx))
        firstLine.append('uncertainty ' + str(idx))
        firstLine.append('isotope count ' + str(idx))
        firstLine.append('quantum number ' + str(idx))
        firstLine.append('catalog tag ' + str(idx))
        firstLine.append('linelist ' + str(idx))
        matrix = []

    matrix.append(firstLine)

    for i in range(len(spectrum_freqs)):
        if np.sum(spectrum_freqs == spectrum_freqs[i]) == 1:
            matrix.append([spectrum_freqs[i], spectrum_ints[i]])

    del matrix[0]
    newMatrix = matrix

    #print('querying JPL/CDMS')
    print('')
    savedForms = []
    savedList = []
    savedTags = []
    savedCatIndices = []
    catCount = 0
    no_line_mols = []

    '''
    The following loop combines queries of Splatalogue, CDMS, and JPL to get all candidate molecules
    for all of the lines in the spectrum along with the required information. For all candidates,
    the spectrum is simulated at the observational parameters.
    '''

    # Apply VLSR shift to database frequencies

    fullMatrix = []
    fullMatrix.append(newMatrix[0])
    database_freqs = apply_vlsr_shift(database_freqs, vlsr_value)
    #frequency threshold for matching database frequencies to observed frequencies (i.e. will consider all lines within +/- 0.5* frequency linewidth as candidates)
    freq_threshold = 0.5* dv_value_freq 
    allScrapedMols = []

    tested_smiles = {}

    for row in newMatrix: #loop through all observed lines in the spectrum
        sf = float(row[0])
        line_mols = []
        start_idx = np.searchsorted(database_freqs, sf - freq_threshold, side="left") #start index in uploaded database for candidates within frequency threshold
        end_idx = np.searchsorted(database_freqs, sf + freq_threshold, side="right") #end index in uploaded database for candidates within frequency threshold
        for match_idx in range(start_idx, end_idx):
            match_tu = (
            database_names[match_idx], database_forms[match_idx], database_smiles[match_idx], database_freqs[match_idx],
            database_errs[match_idx], database_isos[match_idx], database_tags[match_idx], database_lists[match_idx],
            database_vibs[match_idx])

            smi_to_test = database_smiles[match_idx]
            if smi_to_test not in tested_smiles: #ensuring that the molecule can be properly embedded by VICGAE, otherwise ignoring
                try: 
                    vicgae_model.embed_smiles(database_smiles[match_idx])
                    tested_smiles[smi_to_test] = False #no issues with embedding
                except:
                    tested_smiles[smi_to_test] = True #issues with embedding
                    #print('failed embedding', database_smiles[match_idx])

            if database_names[match_idx] not in ignore_mol_list and tested_smiles[smi_to_test] == False:
                if consider_hyperfine == True:
                    line_mols.append(match_tu)
                else:
                    if match_tu[6] < 200000:  # Exclude hyperfine lines
                        line_mols.append(match_tu)
            else:
                ignoreMolDict[database_names[match_idx]] = True


        '''
        Next few lines remove duplicate molecules between CDMS and JPL, and also remove non-hyperfine lines if hyperfine lines are present
        '''

        
        cdms_keys = {(tup[2], tup[5], tup[8]) for tup in line_mols if tup[7] == 'CDMS' and tup[2] != 'NEEDS SMILES'}
        jpl_tuples = [tup for tup in line_mols if tup[7] == 'JPL' and tup[2] != 'NEEDS SMILES']
        jpl_with_matching_cdms = [tup for tup in jpl_tuples if (tup[2], tup[5], tup[8]) in cdms_keys]


        tag_to_tuple = {tup[6]: tup for tup in line_mols}
        non_hyperfine_match = []

        for ta in tag_to_tuple:
            if ta > 200000:
                base_tag = ta - 200000
                if base_tag in tag_to_tuple:
                    non_hyperfine_match.append(tag_to_tuple[base_tag])

        local_molecule_count = (len(row) - 2) // 9
        #print(local_molecule_count)
        local_smi_iso = [
            (row[2 + 9 * i + 2], row[2 + 9 * i + 5])
            for i in range(local_molecule_count)
        ]

        matching_mols = [
            mol for mol in line_mols
            if [mol[2], mol[5]] in [list(pair) for pair in local_smi_iso]
        ]
        line_mols_final = []
        for lmf in line_mols:
            if lmf not in jpl_with_matching_cdms and lmf not in matching_mols and lmf not in non_hyperfine_match:
                line_mols_final.append(lmf)

        #adding molecule information to the dataset row
        for lmf in line_mols_final:
            outsider = False
            if '.' not in lmf[2] and 'NEED' not in lmf[2]:
                #if outsider == False:
                row.append(lmf[0])
                row.append(lmf[0])
                row.append(lmf[2])
                row.append(lmf[3])
                row.append(lmf[4])
                row.append(lmf[5])
                row.append(lmf[-2])
                row.append(lmf[-3])
                row.append(lmf[-2])

                if (lmf[0],lmf[-3],lmf[-2],lmf[2]) not in allScrapedMols:
                    allScrapedMols.append((lmf[0],lmf[-3],lmf[-2],lmf[2]))

        numMols = int((len(row) - 2) / 9)
        if len(row) > 2:
            rem = maxMols - numMols
            for v in range(rem):
                for g in range(9):
                    row.append('NA')
            fullMatrix.append(row)
        else:
            print('Line at ' + str(row[0]) + ' has no molecular candidates, it is being ignored')
            print('')
            noCanFreq.append(float(row[0]))
            noCanInts.append(float(row[1]))

    cont_obj = Continuum(type='thermal', params=cont_temp)
    src = Source(Tex=user_temp, column=1.E10, size=sourceSize, dV=dv_value, velocity = vlsr_value, continuum=cont_obj)
    
    #for all molecules that were found as candidates for any of the lines in the spectrum, simulate their spectra at the inputted and determined parameters
    for asm in allScrapedMols:
        dfFreq = pd.DataFrame()
        if asm[2] == 'CDMS':
            tagString = f"{asm[1]:06d}"
            molDirec = all_loaded['cdms']['directory'] + tagString + '.pkl'
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)

            #simulate the spectrum for the molecule
            sim = Simulation(mol = mol,
                    ll = ll0,
                    ul = ul0,
                    observation = dataScrape,
                    source = src,
                    line_profile = 'Gaussian',
                    use_obs = True) 
                
            if len(sim.spectrum.freq_profile) > 0:
                        #find peaks in simulated spectrum.
                        peak_indicesIndiv = find_peaks(sim.spectrum.freq_profile, sim.spectrum.int_profile, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_value_freq), is_sim=True)
                        if len(peak_indicesIndiv) > 0:
                            peak_freqs2 = sim.spectrum.freq_profile[peak_indicesIndiv]
                            peak_ints2 = abs(sim.spectrum.int_profile[peak_indicesIndiv])
                            if peak_ints2 is not None:
                                freqs = peak_freqs2
                                ints = peak_ints2
                                dfFreq['frequencies'] = freqs
                                dfFreq['intensities'] = ints
                                splatDict[(asm[0],'CDMS')] = (freqs,ints) #store the frequencies and intensities in a dictionary
                                saveName = os.path.join(all_loaded['paths']['splat'], str(catCount) + '.csv')
                                dfFreq.to_csv(saveName)
                                savedCatIndices.append(catCount)
                                savedForms.append(asm[0])
                                savedTags.append(asm[1])
                                savedList.append('CDMS')
                            else:
                                no_line_mols.append(asm)
                        else: 
                            no_line_mols.append(asm)
            else:
                no_line_mols.append(asm)
                    
        elif asm[2] == 'JPL':
            tagString = str(asm[1])
            molDirec = all_loaded['jpl']['directory'] + tagString + '.pkl'
            with open(molDirec, 'rb') as md: #upload JPL molecule
                mol = pickle.load(md)
            #simulate the spectrum for the molecule
            sim = Simulation(mol = mol,
                    ll = ll0,
                    ul = ul0,
                    observation = dataScrape,
                    source = src,
                    line_profile = 'Gaussian',
                    use_obs = True) 
                
            if len(sim.spectrum.freq_profile) > 0:
                        #find peaks in simulated spectrum.
                        peak_indicesIndiv = find_peaks(sim.spectrum.freq_profile, sim.spectrum.int_profile, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_value_freq), is_sim=True)
                        if len(peak_indicesIndiv) > 0:
                            peak_freqs2 = sim.spectrum.freq_profile[peak_indicesIndiv]
                            peak_ints2 = abs(sim.spectrum.int_profile[peak_indicesIndiv])
                            if peak_ints2 is not None:
                                freqs = peak_freqs2
                                ints = peak_ints2
                                dfFreq['frequencies'] = freqs
                                dfFreq['intensities'] = ints
                                splatDict[(asm[0],'JPL')] = (freqs,ints) #store the frequencies and intensities in a dictionary
                                saveName = os.path.join(all_loaded['paths']['splat'], str(catCount) + '.csv')
                                dfFreq.to_csv(saveName)

                                savedCatIndices.append(catCount)
                                savedForms.append(asm[0])
                                savedTags.append(asm[1])
                                savedList.append('JPL')
                            else:
                                no_line_mols.append(asm)
                        else:
                            no_line_mols.append(asm)
            else:
                no_line_mols.append(asm)

        catCount += 1

    newMatrixNext = []
    rowCounter = 0
    '''
    Quality control step 
    '''
    for row_check in newMatrix:
        keptChunks = []
        newRow = [row_check[0],row_check[1]]
        short_list = row_check[2:]
        chunks = [short_list[o:o + 9] for o in range(0, len(short_list), 9)]
        for ch in chunks:
            if ch[0]!= 'NA':
                foundWrong = False
                for g in no_line_mols:
                    if g[1] == ch[-2] and g[2] == ch[-1]:
                        foundWrong = True

                if foundWrong == False:
                    keptChunks.append(ch)
        
        for kc in keptChunks:
            for n in kc:
                newRow.append(n)
        rem = maxMols - len(keptChunks)
        for re in range(9*rem):
            newRow.append('NA')
        if len(keptChunks) > 0:
            newMatrixNext.append(newRow)

        rowCounter+=1 

    newMatrix = newMatrixNext   


    pathDataset = os.path.join(direc, 'dataset.csv')
    file = open(pathDataset, 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(newMatrix)


    dfTags = pd.DataFrame()
    dfTags['idx'] = savedCatIndices
    dfTags['formula'] = savedForms
    dfTags['tags'] = savedTags
    dfTags['linelist'] = savedList

    pathSplatCat = all_loaded['paths']['splat_og'] + '_' + str(int(user_temp)) + '/catalogues/'
    dfTags.to_csv(os.path.join(pathSplatCat, 'catalog_list.csv'))

    fullMatrix = []

    for row in newMatrix:
        numMols = int((len(row) - 2) / 9)
        if len(row) > 2:
            rem = maxMols - numMols
            for v in range(rem):
                for g in range(9):
                    row.append('NA')
            fullMatrix.append(row)
        else:
            print('Line at ' + str(row[0]) + ' has no molecular candidates, it is being ignored')
            print('')
            noCanFreq.append(float(row[0]))
            noCanInts.append(float(row[1]))


    qualCount = 0
    # quality control step
    updatedFull = []
    updatedFull.append(firstLine)
    #del fullMatrix[0]
    for row in fullMatrix:
        if len(row) != 9 * maxMols + 2:
            inIndices = []
            for p in range(len(row)):
                if row[p] == None or row[p] == '':
                    if row[p + 1] == None or row[p + 1] == '':
                        inIndices.append(p)

            blankAlready = 0
            for blankIdx in inIndices:
                sI = blankIdx - 5 * blankAlready
                eI = blankIdx - 5 * blankAlready + 5
                del row[sI:eI]
                blankAlready += 1

            updatedFull.append(row)
        else:
            updatedFull.append(row)

        qualCount += 1

    fullMatrix = updatedFull
    finalMatrix2 = []
    #finalMatrix2.append(firstLine)

    for row in fullMatrix:
        if len(row) < 2 + 9 * maxMols:
            rem = 2 + 9 * maxMols - len(row)
            for c in range(rem):
                row.append('NA')

        finalMatrix2.append(row)

    fullMatrix = finalMatrix2
    pathDatasetInt = os.path.join(direc, 'dataset_final.csv')

    file = open(pathDatasetInt, 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(fullMatrix)

    tockScrape = time.perf_counter()


    scrapeMins = (tockScrape-tickScrape)/60
    scrapeMins2 = "{{:.{}f}}".format(2).format(scrapeMins)
    #print('Catalog scraping took ' + str(scrapeMins2) + ' minutes.')

    mol_smileMols = []
    mol_smileSmiles = []

    fullMatrix2 = fullMatrix
    del fullMatrix2[0]
    for row in fullMatrix2:
        for z in range(maxMols):
            molIdx = 9 * z + 3
            smileIdx = 9 * z + 4
            if row[molIdx] not in mol_smileMols:
                mol_smileMols.append(row[molIdx])
                mol_smileSmiles.append(row[smileIdx])

    dfMolSmiles = pd.DataFrame()
    dfMolSmiles['molecules'] = mol_smileMols
    dfMolSmiles['smiles'] = mol_smileSmiles
    dfMolSmiles.to_csv(os.path.join(direc, 'mol_smiles.csv'), index=False)


    for p in ignoreMolDict:
        if ignoreMolDict[p] == False:
            print('Algorithm was forced to ignore',p,'but never came across a line for which this was a potential candidate.')
    return all_loaded, noCanFreq, noCanInts, splatDict, cont_obj

