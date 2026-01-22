"""
run_assignment.py

Script to perform iterative spectral line assignment on radioastronomical observational data 
using the core functionality defined in spectrum_assignment.py.

Workflow Overview:
1. **Data Loading**:
   - Reads observed spectral line frequencies and intensities from a CSV dataset.
   - Loads candidate molecular information including SMILES, formulas, isotopes, catalog tags,
     linelists, and reference frequencies.

2. **Candidate Score Creation**:
   - Generates `CandidateScore` objects for each candidate molecule per observed line.
   - Scales catalog intensities to match observed values and sorts lines by intensity.
   - Prepares all necessary data structures for static and dynamic scoring.

3. **Scoring Context Setup**:
   - Initializes `ScoringContext` with temperature, RMS noise, valid atoms, and thresholds.
   - Tracks detected molecules, structural relevance, and frequency/intensity checks.

4. **Iterative Assignment Execution**:
   - Initializes `IterativeSpectrumAssignment` and adds all candidate scores.
   - Performs one-time static checks (invalid atoms, vibrational excitation, missing strong lines, etc.).
   - Iteratively assigns lines with dynamic rescoring:
       - Structural relevance is updated using VICGAE embeddings.
       - Softmax and combined scores determine priority candidates.
       - Assignment status is updated for each line.
       - Overrides handle outlier molecules based on structural relevance.
   - Final convergence ensures stable assignments.

5. **Results Generation and Saving**:
   - Saves per-line combined scores and full candidate score information to pickle files.

Functions:
- `run_assignment`: Loads data, generates candidates, runs iterative assignment, and returns stats.
- `save_results`: Saves assignment outputs to files.
- `run_full_assignment`: Wrapper that executes assignment and automatically saves results.
"""


import csv
import os
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple
import time

# Import the core classes and functions
from .spectrum_assignment import (
    CandidateScore,
    SpectralLine,
    ScoringContext,
    IterativeSpectrumAssignment,
    AssignmentStatus
)

# Import required constants and utilities
from ..constants import (
    ckm, maxMols, local_thresh, global_thresh, 
    vicgae_model, covParam, span, ignoreIso
)


def create_candidate_score(line_idx: int, cand_idx: int, 
                           allSmiles: List, molForms: List, allIso: List,
                           allFrequencies: List, allQn: List, molTags: List,
                           molLinelist: List, actualFrequencies: List,
                           intensities: List, splatDict: Dict,
                           dv_value_freq: float) -> CandidateScore:
    """
    Create a CandidateScore object with all required fields from the dataset.
    """
    formula = molForms[line_idx][cand_idx]
    linelist = molLinelist[line_idx][cand_idx]
    catalog_freq = allFrequencies[line_idx][cand_idx]
    observed_freq = actualFrequencies[line_idx]
    observed_int = intensities[line_idx]
    
    # Get catalog data from splatDict
    freqs = splatDict[(formula, linelist)][0]  # Catalog frequencies
    peak_ints = splatDict[(formula, linelist)][1]  # Catalog intensities
    
    # Find closest catalog frequency
    closest_idx = np.argmin(np.abs(freqs - catalog_freq))
    closest_freq = freqs[closest_idx]
    
    # Handle multiple entries at same frequency (take highest intensity)
    mask = freqs == closest_freq
    if np.sum(mask) > 1:
        indices = np.where(mask)[0]
        int_specs = peak_ints[indices]
        int_idx = np.argmax(int_specs)
        closest_idx = indices[int_idx]
    
    # Calculate intensity scale factor
    int_value = peak_ints[closest_idx]
    if int_value == 0 or np.isnan(int_value):
        scale_value = 1.E20
    else:
        scale_value = observed_int / int_value
    
    # Scale all intensities and sort
    peak_ints_scaled = peak_ints * scale_value
    sorted_indices = np.argsort(peak_ints_scaled)[::-1]
    sorted_freqs = freqs[sorted_indices]
    sorted_ints = peak_ints_scaled[sorted_indices]
    
    max_int = sorted_ints[0] if len(sorted_ints) > 0 else 0
    
    # Find molecule rank
    rank_indices = np.where(sorted_freqs == closest_freq)[0]
    if len(rank_indices) > 0:
        mol_rank = rank_indices[0]
    else:
        mol_rank = 1000
    
    # Handle catalog tag
    tag = molTags[line_idx][cand_idx]
    if tag == 'NA' or tag == '<NA>':
        catalog_tag = -1
    else:
        catalog_tag = int(tag)
    
    # Create the CandidateScore object
    return CandidateScore(
        # Molecular identity
        smiles=allSmiles[line_idx][cand_idx],
        formula=formula,
        isotope=int(allIso[line_idx][cand_idx]),
        quantum_number=allQn[line_idx][cand_idx],
        catalog_tag=catalog_tag,
        linelist=linelist,
        
        # Raw data
        #catalog_frequency=catalog_freq,
        catalog_frequency=closest_freq,
        observed_frequency=observed_freq,
        observed_intensity=observed_int,
        simulated_intensity=int_value,
        max_simulated_intensity=max_int,
        intensity_scale_factor=scale_value,
        molecule_rank=mol_rank,
        
        # Cached arrays for intensity checks
        sorted_frequencies=sorted_freqs,
        sorted_intensities=sorted_ints,
        
        # Quality checks (will be set by perform_static_checks)
        has_invalid_atoms=False,
        is_vibrationally_excited=False,
        rule_out_strong_lines=False,
        frequency_match_score=0.0,
        
        # Dynamic scores (will be calculated later)
        structural_score=0.0,
        global_score=0.0,
        softmax_score=0.0,
        combined_score=0.0,
        
        # Flags
        penalties=[],
        is_isotopologue_too_strong=False,
        has_relative_intensity_mismatch=False,
        has_unreasonably_strong_lines=False
    )


def load_dataset(direc: str, numMols: int = None) -> Tuple:
    """
    Load the dataset from CSV file.
    
    Returns:
        Tuple containing all the loaded data arrays
    """
    if numMols is None:
        numMols = maxMols
    
    actualFrequencies = []
    allSmiles = []
    allIso = []
    allFrequencies = []
    intensities = []
    molForms = []
    molTags = []
    molLinelist = []
    allQn = []
    totalSmiles = []
    totalForms = []
    
    # Load the dataset CSV
    file = open(os.path.join(direc, 'dataset_final.csv'), 'r')
    matrix = list(csv.reader(file, delimiter=","))
    del matrix[0]  # Remove header
    
    # Process each row
    for row in matrix:
        intensities.append(float(row[1]))
        actualFrequencies.append(float(row[0]))
        
        rowSmiles = []
        rowIso = []
        rowFreq = []
        rowForms = []
        rowTags = []
        rowLinelist = []
        rowQN = []
        
        for i in range(numMols):
            idx = i * 9 + 4
            if row[idx] != 'NA':
                rowSmiles.append(row[idx])
                if row[idx] not in totalSmiles:
                    totalSmiles.append(row[idx])
                if row[idx-1] not in totalForms:
                    totalForms.append(row[idx-1])
                
                rowIso.append(int(row[idx + 3]))
                rowFreq.append(float(row[idx + 1]))
                rowForms.append(row[idx - 1])
                
                if row[idx + 5] != '<NA>' and row[idx + 5] != 'local':
                    rowTags.append(int(row[idx + 5]))
                else:
                    rowTags.append('NA')
                
                rowLinelist.append(row[idx + 6])
                rowQN.append(row[idx + 4])
        
        molForms.append(rowForms)
        allIso.append(rowIso)
        allSmiles.append(rowSmiles)
        allFrequencies.append(rowFreq)
        molTags.append(rowTags)
        molLinelist.append(rowLinelist)
        allQn.append(rowQN)
    
    file.close()
    
    return (actualFrequencies, intensities, allSmiles, allIso, 
            allFrequencies, molForms, molTags, molLinelist, 
            allQn, totalSmiles, totalForms)


def run_assignment(temp: float, direc: str, subdirec: str, splatDict: Dict, 
                  validAtoms: List[str], dv_value_freq: float, 
                  rms: float, peak_freqs_full: np.ndarray,
                  known_molecules: List[str] = None, consider_structure: bool = True) -> Tuple:
    """
    Main function to run the iterative spectrum assignment.
    
    Args:
        temp: Temperature in Kelvin
        direc: Directory containing data files
        splatDict: Dictionary of spectral catalogs
        validAtoms: List of valid atomic symbols
        dv_value_freq: Frequency tolerance
        rms: RMS noise level
        peak_freqs_full: Array of all peak frequencies
        known_molecules: Optional list of SMILES strings for molecules known to be present.
                        These molecules will always be maintained in the detected list.
        subdirec: Subdirectory in which the output files will be saved
    
    Returns:
        Tuple of (assigner, statistics)
    """
    
    if known_molecules is None:
        known_molecules = []
    
    print(f"Loading dataset from {direc}...")
    
    # Load the dataset
    (actualFrequencies, intensities, allSmiles, allIso, 
     allFrequencies, molForms, molTags, molLinelist, 
     allQn, totalSmiles, totalForms) = load_dataset(subdirec)
    
    print(f"Loaded {len(actualFrequencies)} spectral lines")
    print(f"Found {len(totalSmiles)} unique molecular candidates")
    
    if known_molecules:
        print(f"Using {len(known_molecules)} known molecules that will always be maintained in detected list:")
        for smiles in known_molecules:
            print(f"  - {smiles}")
    
    # Create scoring context
    context = ScoringContext(
        dv_value_freq=dv_value_freq,
        temperature=temp,
        rms=rms,
        max_observed_intensity=max(intensities),
        global_threshold_original=global_thresh,
        local_threshold=local_thresh,
        valid_atoms=validAtoms,
        ignore_isotopologues=ignoreIso,
        peak_freqs_full=peak_freqs_full,
        total_smiles=totalSmiles,
        known_molecules=known_molecules,
        consider_structure = consider_structure
    )
    
    # Initialize the assignment system
    assigner = IterativeSpectrumAssignment(
        frequencies=np.array(actualFrequencies),
        intensities=np.array(intensities),
        context=context
    )
    
    print("Creating candidate scores for all lines...")
    
    # Add all candidates to their respective lines
    for line_idx in range(len(actualFrequencies)):
        candidates = []
        
        # Skip if no candidates for this line
        if len(allSmiles[line_idx]) == 0:
            continue
        
        # Create CandidateScore for each molecular candidate
        for cand_idx in range(len(allSmiles[line_idx])):
            try:
                candidate = create_candidate_score(
                    line_idx, cand_idx,
                    allSmiles, molForms, allIso, allFrequencies,
                    allQn, molTags, molLinelist,
                    actualFrequencies, intensities,
                    splatDict, dv_value_freq
                )
                candidates.append(candidate)
            except Exception as e:
                print(f"Error creating candidate at line {line_idx}, "
                      f"candidate {cand_idx}: {e}")
                continue
        
        # Add all candidates to this line
        if candidates:
            assigner.add_candidates_to_line(line_idx, candidates)
    
    print(f"Added candidates to {len(assigner.lines)} lines")
    
    # Run the iterative assignment
    print("\nStarting iterative assignment algorithm...")
    start_time = time.perf_counter()
    
    assigner.assign_all_iteratively()
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    # Get statistics
    stats = assigner.statistics()
    
    print("\n" + "="*60)
    print("ASSIGNMENT COMPLETE")
    print("="*60)
    #print(f"Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    #print(f"Total lines: {stats['total_lines']}")
    #print(f"Assigned: {stats['assigned']} ({100*stats['assigned']/stats['total_lines']:.1f}%)")
    #print(f"Multiple carriers: {stats['multiple_carriers']}")
    #print(f"Unidentified: {stats['unidentified']} ({100*stats['unidentified']/stats['total_lines']:.1f}%)")
    #print(f"Detected molecules: {stats['detected_molecules']}")
    #print(f"Structural calculations: {stats['structural_calculations']}")
    #print(f"Override molecules: {stats['override_molecules']}")
    
    return assigner, stats

def save_results(assigner: IterativeSpectrumAssignment, 
                direc: str, temp: float, subdirec: str):
    """
    Save assignment results to files.
    """
    # Prepare data for saving
    combined_scores_list = []
    testing_scores_list = []
    
    for line in assigner.lines:
        # Get combined scores for this line
        combined_scores = []
        for candidate in line.top_candidates[:5]:  # Top 5 by combined score
            combined_scores.append((
                (candidate.smiles, candidate.formula),
                candidate.combined_score
            ))
        combined_scores_list.append(combined_scores)
        
        # Get all scores for this line
        line_scores = []
        for candidate in line.candidates:
            line_scores.append([
                candidate.smiles,
                candidate.global_score,
                candidate.formula,
                candidate.penalties,
                candidate.quantum_number,
                10,  # Default value from old code
                candidate.isotope
            ])
        testing_scores_list.append(line_scores)
    
    # Save to pickle files
    saveCombFile = os.path.join(subdirec, f'combined_list_{int(temp)}.pkl')
    saveTestFile = os.path.join(subdirec, f'testing_list_{int(temp)}.pkl')
    
    with open(saveCombFile, "wb") as fp:
        pickle.dump(combined_scores_list, fp)
    
    with open(saveTestFile, "wb") as fp:
        pickle.dump(testing_scores_list, fp)
    
    #print(f"\nResults saved to:")
    #print(f"  {saveCombFile}")
    #print(f"  {saveTestFile}")
    
    # Save assigned molecules list
    assigned_molecules = assigner.get_assigned_molecules()
    #assigned_file = os.path.join(direc, f'assigned_molecules_{int(temp)}.csv')
    
    #with open(assigned_file, 'w', newline='') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(['Formula', 'Database'])
    #    writer.writerows(assigned_molecules)
    
    #print(f"  {assigned_file}")


def run_full_assignment(temp, direc, subdirec, splatDict, valid_atoms, dv_value_freq, 
                       rms, peak_freqs_full, consider_structure, known_molecules=None):
    """
    Complete assignment workflow with result saving.
    
    Args:
        temp: Temperature in Kelvin
        direc: Directory containing data files
        subdirec: Directory in which output files will be saved.
        splatDict: Dictionary of spectral catalogs
        valid_atoms: List of valid atomic symbols
        dv_value_freq: Frequency tolerance
        rms: RMS noise level
        peak_freqs_full: Array of all peak frequencies
        known_molecules: Optional list of SMILES strings for molecules known to be present.
                        These molecules will always be maintained in the detected list.
    
    Returns:
        Tuple of (assigner, statistics)
    """
    assigner, stats = run_assignment(
        temp=temp,
        direc=direc,
        subdirec = subdirec,
        splatDict=splatDict,
        validAtoms=valid_atoms,
        dv_value_freq=dv_value_freq,
        rms=rms,
        peak_freqs_full=peak_freqs_full,
        known_molecules=known_molecules,
        consider_structure=consider_structure
    )
    
    # Save results
    save_results(assigner, direc, temp, subdirec)

    #output_file = os.path.join(direc, f'unassigned_analysis_{int(temp)}.txt')
    #assigner.generate_unassigned_analysis(output_file)


    return assigner, stats

