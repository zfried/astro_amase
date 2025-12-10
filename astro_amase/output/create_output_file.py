"""
Create output.txt file to report line assignment results.
"""

import os
from ..constants import global_thresh, local_thresh


def write_output_text(assignment: 'IterativeSpectrumAssignment', 
                                      direc: str, 
                                      temp: float, 
                                      dv_value: float, 
                                      vlsr_value: float):
    """
    Generate comprehensive text report from IterativeSpectrumAssignment object.
    
    Creates a detailed output_report.txt file documenting the assignment status of each spectral
    line, including assigned carriers, candidate molecules, confidence scores, and quality issues.
    Lines are categorized as: (1) assigned to single carrier, (2) multiple possible carriers, or
    (3) unidentified. Also compiles a final summary and returns the list of assigned molecules
    for subsequent best-fit analysis.
    
    Parameters
    ----------
    assignment : IterativeSpectrumAssignment
        The completed assignment object containing all lines and their candidates.
    direc : str
        Output directory path where 'output_report.txt' will be written.
    temp : float
        Determined excitation temperature (K) from line analysis.
    dv_value : float
        Determined line width (km/s) from analysis.
    vlsr_value : float
        Determined source velocity (km/s) relative to the local standard of rest.
    
    Returns
    -------
    assignedMols : list of tuple
        List of (molecule_name, database_source) tuples for all molecules assigned to
        at least one spectral line. Used as input for subsequent best-fit modeling.
        Example: [('CH3OH', 'CDMS'), ('HC3N', 'JPL')]
    """
    
    globalThreshOriginal = global_thresh
    thresh = local_thresh
    
    f = open(os.path.join(direc, 'output_report.txt'), 'w')
    
    # Write header
    f.write('Determined Line Width: ' + str(round(dv_value, 2)) + ' km/s\n')
    f.write('Determined VLSR: ' + str(round(vlsr_value, 2)) + ' km/s\n')
    f.write('Determined Temperature: ' + str(round(temp, 2)) + ' K\n')
    f.write('\n')
    f.write('\n')
    f.write('--------------------------------------\n')
    
    # Counters
    assignCount = 0
    mulCount = 0
    unCount = 0
    
    assignList = []
    
    # Loop through all lines
    for i, line in enumerate(assignment.lines):
        f.write('LINE ' + str(i + 1) + ':\n')
        f.write('Sky Frequency: ' + str(round(line.frequency, 4)) + ' MHz\n')
        
        # Determine assignment category
        if line.assignment_status is None:
            f.write('Unidentified\n')
            unCount += 1
        
        elif line.assignment_status.value == 'unidentified':
            f.write('Unidentified\n')
            unCount += 1
        
        elif line.assignment_status.value == 'multiple_carriers':
            f.write('Several Possible Carriers\n')
            f.write('Listed from highest to lowest ranked, these are:\n')
            
            # Get unique molecules passing global threshold
            passing_mols = {}
            for candidate in line.candidates:
                if candidate.global_score >= globalThreshOriginal:
                    key = (candidate.smiles, candidate.formula)
                    if key not in passing_mols or candidate.combined_score > passing_mols[key].combined_score:
                        passing_mols[key] = candidate
            
            # Sort by combined score
            sorted_passing = sorted(passing_mols.values(), 
                                   key=lambda c: c.combined_score, 
                                   reverse=True)
            
            for candidate in sorted_passing:
                f.write(candidate.formula + ' (' + candidate.smiles + ')\n')
                assignList.append((candidate.smiles, candidate.formula, 
                                  candidate.global_score, candidate.linelist))
            
            mulCount += 1
        
        elif line.assignment_status.value == 'assigned':
            f.write('Assigned to: ' + line.assigned_molecule + ' (' + line.assigned_smiles + ')\n')
            
            assignList.append((line.assigned_smiles, line.assigned_molecule, 
                             line.best_candidate.global_score, line.best_candidate.linelist))
            assignCount += 1
        
        # Write all candidate transitions
        f.write('\n')
        f.write('All Candidate Transitions:\n')
        
        # Sort candidates by global score
        sorted_candidates = sorted(line.candidates, 
                                  key=lambda c: c.global_score, 
                                  reverse=True)
        
        for candidate in sorted_candidates:
            f.write('\n')
            f.write('Molecule: ' + candidate.formula + ' (' + candidate.smiles + ')\n')
            f.write('Global Score: ' + str(round(candidate.global_score, 2)) + '\n')
            
            # Write issues/penalties
            if len(candidate.penalties) == 0:
                f.write('No issues with this line\n')
            else:
                f.write('Issues with this line: ')
                for penalty in candidate.penalties:
                    f.write(penalty + ' ')
                f.write('\n')
        
        f.write('\n')
        f.write('\n')
        f.write('------------------------------------------------------------\n')
    
    # Write summary
    f.write('\n')
    f.write('Final Report\n')
    f.write('Total number of lines: ' + str(len(assignment.lines)) + '\n')
    f.write('Number of lines assigned to a single carrier: ' + str(assignCount) + '\n')
    f.write('Number of lines with more than one possible carrier: ' + str(mulCount) + '\n')
    f.write('Number of unassigned lines: ' + str(unCount) + '\n')
    
    f.close()
    
    # Create list of assigned molecules for best-fit modeling
    assignedMols = []
    assignedNames = []
    
    for entry in assignList:
        smiles, formula, score, linelist = entry
        if (formula, linelist) not in assignedMols:
            assignedMols.append((formula, linelist))
            assignedNames.append(formula)
    
    return assignedMols


def remove_molecules_and_write_output(assignment: 'IterativeSpectrumAssignment',
                                       molecules_to_remove: list,
                                       direc: str,
                                       temp: float,
                                       dv_value: float,
                                       vlsr_value: float):
    """
    Remove specified molecules by penalizing their scores, reassign lines, and generate report.
    
    For each molecule in molecules_to_remove, this function:
    1. Reduces global_score by factor of 2 for all matching candidates
    2. Adds penalty "Removed during fitting stage."
    3. Recalculates softmax and combined scores for affected lines
    4. Reassigns lines based on new scores
    5. Generates comprehensive output_report.txt
    
    Parameters
    ----------
    assignment : IterativeSpectrumAssignment
        The completed assignment object containing all lines and their candidates.
    molecules_to_remove : list of str
        List of formula strings to remove. Example: ['CH3OH', 'HC3N']
    direc : str
        Output directory path where 'output_report.txt' will be written.
    temp : float
        Determined excitation temperature (K) from line analysis.
    dv_value : float
        Determined line width (km/s) from analysis.
    vlsr_value : float
        Determined source velocity (km/s) relative to the local standard of rest.
    
    Returns
    -------
    assignedMols : list of tuple
        List of (molecule_name, database_source) tuples for all molecules assigned to
        at least one spectral line after removal. Used as input for subsequent best-fit modeling.
    """
    
    globalThreshOriginal = global_thresh
    thresh = local_thresh
    
    # Step 1: Penalize molecules to be removed
    unique_remove_list = list(set(molecules_to_remove))
    print(f"Removing {len(unique_remove_list)} molecules from consideration...")
    
    for line in assignment.lines:
        modified = False
        
        for candidate in line.candidates:
            # Check if this candidate should be removed
            if candidate.formula in molecules_to_remove:
                # Reduce global score by factor of 2
                candidate.global_score *= 0.5
                
                # Add penalty
                if "Removed during fitting stage." not in candidate.penalties:
                    candidate.penalties.append("Removed during fitting stage.")
                
                modified = True
        
        # If any candidates were modified, recalculate softmax and combined scores
        if modified:
            line.calculate_softmax_and_combined()
    
    # Step 2: Reassign all lines with new scores
    print("Reassigning lines based on updated scores...")
    
    for line in assignment.lines:
        line.assign(thresh)
    
    # Step 3: Generate output report
    print(f"Writing output report to {os.path.join(direc, 'output_report.txt')}...")
    
    f = open(os.path.join(direc, 'output_report.txt'), 'w')
    
    # Write header
    f.write('Determined Line Width: ' + str(round(dv_value, 2)) + ' km/s\n')
    f.write('Determined (or inputted) VLSR: ' + str(round(vlsr_value, 2)) + ' km/s\n')
    f.write('Determined (or inputted) Temperature: ' + str(round(temp, 2)) + ' K\n')
    f.write('\n')
    f.write('--------------------------------------\n')
    
    # Counters
    assignCount = 0
    mulCount = 0
    unCount = 0
    
    assignList = []
    
    # Loop through all lines
    for i, line in enumerate(assignment.lines):
        f.write('LINE ' + str(i + 1) + ':\n')
        f.write('Sky Frequency: ' + str(round(line.frequency, 4)) + ' MHz\n')
        
        # Determine assignment category
        if line.assignment_status is None:
            f.write('Unidentified\n')
            unCount += 1
        
        elif line.assignment_status.value == 'unidentified':
            f.write('Unidentified\n')
            unCount += 1
        
        elif line.assignment_status.value == 'multiple_carriers':
            f.write('Several Possible Carriers\n')
            f.write('Listed from highest to lowest ranked, these are:\n')
            
            # Get unique molecules passing global threshold
            passing_mols = {}
            for candidate in line.candidates:
                if candidate.global_score >= globalThreshOriginal:
                    key = (candidate.smiles, candidate.formula)
                    if key not in passing_mols or candidate.combined_score > passing_mols[key].combined_score:
                        passing_mols[key] = candidate
            
            # Sort by combined score
            sorted_passing = sorted(passing_mols.values(), 
                                   key=lambda c: c.combined_score, 
                                   reverse=True)
            
            for candidate in sorted_passing:
                f.write(candidate.formula + ' (' + candidate.smiles + ')\n')
                assignList.append((candidate.smiles, candidate.formula, 
                                  candidate.global_score, candidate.linelist))
            
            mulCount += 1
        
        elif line.assignment_status.value == 'assigned':
            f.write('Assigned to: ' + line.assigned_molecule + ' (' + line.assigned_smiles + ')\n')
            
            assignList.append((line.assigned_smiles, line.assigned_molecule, 
                             line.best_candidate.global_score, line.best_candidate.linelist))
            assignCount += 1
        
        # Write all candidate transitions
        f.write('\n')
        f.write('All Candidate Transitions:\n')
        
        # Sort candidates by global score
        sorted_candidates = sorted(line.candidates, 
                                  key=lambda c: c.global_score, 
                                  reverse=True)
        
        for candidate in sorted_candidates:
            f.write('\n')
            f.write('Molecule: ' + candidate.formula + ' (' + candidate.smiles + ')\n')
            f.write('Global Score: ' + str(round(candidate.global_score, 2)) + '\n')
            
            # Write issues/penalties
            if len(candidate.penalties) == 0:
                f.write('No issues with this line\n')
            else:
                f.write('Issues with this line: ')
                for penalty in candidate.penalties:
                    f.write(penalty + ' ')
                f.write('\n')
        
        f.write('\n')
        f.write('\n')
        f.write('------------------------------------------------------------\n')
    
    # Write summary
    f.write('\n')
    f.write('Final Report\n')
    f.write('Total number of lines: ' + str(len(assignment.lines)) + '\n')
    f.write('Number of lines assigned to a single carrier: ' + str(assignCount) + '\n')
    f.write('Number of lines with more than one possible carrier: ' + str(mulCount) + '\n')
    f.write('Number of unassigned lines: ' + str(unCount) + '\n')
    
    f.close()
    
    print("Output report complete!")
    
    # Create list of assigned molecules for best-fit modeling
    assignedMols = []
    assignedNames = []
    
    for entry in assignList:
        smiles, formula, score, linelist = entry
        if (formula, linelist) not in assignedMols:
            assignedMols.append((formula, linelist))
            assignedNames.append(formula)
    
    statDict = {'total_lines': len(assignment.lines), 'assigned_lines': assignCount, 'unidentified_lines': unCount, 'assigned_molecules': len(assignedNames)}

    return assignedMols, statDict