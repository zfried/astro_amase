"""
Best-Fit Spectral Modeling and Optimization

This module fits optimal molecular abundances to observed astronomical spectra
by minimizing residuals between synthetic and observed line intensities. The
fitting uses scipy's least_squares optimizer with a lookup table approach for
computational efficiency.

Key Features:
- Pre-computes synthetic spectra at a grid of column densities for each molecule
- Uses interpolation between grid points for fast spectrum generation during optimization
- Performs iterative fitting with quality control to remove low-significance molecules
- Generates interactive HTML visualizations using Bokeh for result inspection
- Exports best-fit column densities and peak identification results

The workflow:
1. Build lookup tables: Pre-compute spectra for each molecule at multiple column densities
2. Initial fit: Optimize all molecular column densities simultaneously
3. Quality control: Remove molecules with weak contributions or insignificant improvements
4. Final fit: Re-optimize remaining molecules for best residuals
5. Visualization: Create interactive plots and export results

Performance optimizations:
- Lookup tables avoid repeated simulations during optimization
- Linear interpolation provides fast spectrum evaluation at arbitrary column densities
- Float32 arrays reduce memory usage for large spectral datasets
- Selective molecule filtering removes incorrectly assigned molecules.
"""



import pandas as pd
import numpy as np
import os
import pickle 
import warnings
from bokeh.plotting import figure, save, output_file
from bokeh.models import (
    HoverTool, CheckboxGroup, CustomJS, Button, ColumnDataSource
)
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, Button, CustomJS, CheckboxGroup
from scipy.optimize import least_squares
from scipy.interpolate import interp1d, PchipInterpolator
import gc
from ..utils.molsim_classes import Source, Simulation
from ..constants import global_thresh, ckm
from ..utils.molsim_utils import find_peaks
from ..output.create_output_file import molecule_summary_pipeline,molecule_summary_pipeline_df





def get_assigned_molecules(assignment: 'IterativeSpectrumAssignment'):
    """
    Extract list of assigned molecules and identify those only in blended single lines.
    
    Parameters
    ----------
    assignment : IterativeSpectrumAssignment
        The completed assignment object containing all lines and their candidates.
    
    Returns
    -------
    assignedMols : list of tuple
        List of (molecule_name, database_source) tuples for all molecules assigned to
        at least one spectral line.
    blended_single_line_mols : list of tuple
        List of (molecule_name, database_source) tuples for molecules that only
        appear in blended assignments and only in a single line.
    """
    
    globalThreshOriginal = global_thresh
    
    assignList = []
    
    # Track molecules in blended vs single assignments
    blended_mols = {}  # {(formula, linelist): count}
    single_mols = set()  # {(formula, linelist)}
    
    # Loop through all lines
    for line in assignment.lines:
        
        # Handle multiple carriers case (blended)
        if line.assignment_status is not None and line.assignment_status.value == 'multiple_carriers':
            # Get unique molecules passing global threshold
            passing_mols = {}
            for candidate in line.candidates:
                if candidate.global_score >= globalThreshOriginal:
                    key = (candidate.smiles, candidate.formula)
                    if key not in passing_mols or candidate.combined_score > passing_mols[key].combined_score:
                        passing_mols[key] = candidate
            
            # Add all passing molecules to assignList
            for candidate in passing_mols.values():
                assignList.append((candidate.smiles, candidate.formula, 
                                  candidate.global_score, candidate.linelist))
                
                # Count occurrences in blended lines
                mol_key = (candidate.formula, candidate.linelist)
                blended_mols[mol_key] = blended_mols.get(mol_key, 0) + 1
        
        # Handle single assignment case
        elif line.assignment_status is not None and line.assignment_status.value == 'assigned':
            assignList.append((line.assigned_smiles, line.assigned_molecule, 
                             line.best_candidate.global_score, line.best_candidate.linelist))
            
            mol_key = (line.assigned_molecule, line.best_candidate.linelist)
            single_mols.add(mol_key)
    
    # Create list of unique assigned molecules
    assignedMols = []
    for entry in assignList:
        smiles, formula, score, linelist = entry
        if (formula, linelist) not in assignedMols:
            assignedMols.append((formula, linelist))
    
    # Find molecules that are ONLY in blended lines (not in single) and appear only once
    blended_single_line_mols = [
        mol_key for mol_key, count in blended_mols.items()
        if count == 1 and mol_key not in single_mols
    ]


    return assignedMols, blended_single_line_mols

def compute_molecule_lookup_table(mol, label, log_columns, tempInput, dv_value, cont, ll0, ul0, dataScrape, vlsr_value, source_size):
    """
    Compute a lookup table of synthetic spectra for a single molecule across a grid of column densities.
    
    This function creates a Source object and simulates spectra at each column density in the grid,
    reusing the same Source object for efficiency by only updating the column density parameter.
    
    Parameters
    ----------
    mol : Molecule
        molsim Molecule object for species.
    label : str
        Identifying name/label for the molecule.
    log_columns : array_like
        Array of column density values at which to compute spectra.
    tempInput : float
        Excitation temperature (K) for the molecular transitions.
    dv_value : float
        Line width (km/s) for the Gaussian line profile.
    cont : Continuum
        Continuum object defining background radiation field.
    ll0 : float
        Lower frequency bound array (MHz) for the simulation.
    ul0 : float
        Upper frequency bound array (MHz) for the simulation.
    dataScrape : Observation
        Observation object containing frequency grid and observational spectrum.
    vlsr_value : float
        Source velocity (km/s) relative to the local standard of rest.
    
    Returns
    -------
    label : str
        The molecule label (passed through for identification).
    spectra_grid : ndarray
        2D array of shape (n_columns, n_frequencies) containing synthetic spectra
        at each column density grid point.
    """

    spectra_grid = []

    # Create Source object once
    #src = molsim.classes.Source(Tex=temp, column=log_columns[0], dV=dv_value, velocity = vlsr_value)
    src = Source(size=source_size, dV=dv_value, velocity=vlsr_value, Tex=tempInput, column=log_columns[0], continuum=cont)


    for col_idx, column_density in enumerate(log_columns):
        # Update only the column density
        src.column = column_density

        # Run the simulation
        sim = Simulation(
            mol=mol,
            ll=ll0,
            ul=ul0,
            source=src,
            line_profile='Gaussian',
            use_obs=True,
            observation=dataScrape
        )
        spectrum = np.array(sim.spectrum.int_profile)
        
        # Handle empty spectra in lookup table generation
        if spectrum.size == 0:
            # Get expected size from dataScrape observation
            if hasattr(dataScrape, 'spectrum') and hasattr(dataScrape.spectrum, 'frequency'):
                expected_size = len(dataScrape.spectrum.frequency)
                spectrum = np.zeros(expected_size)
                warnings.warn(f"Empty spectrum generated for {label} at column density {column_density:.2e}, using zeros")
            else:
                warnings.warn(f"Empty spectrum generated for {label} at column density {column_density:.2e}, cannot determine size")
        
        spectra_grid.append(spectrum)

    return label, np.array(spectra_grid)


def build_lookup_tables_serial(mol_list, labels, log_columns, tempInput, dv_value, cont, ll0, ul0, dataScrape, vlsr_value, source_size):
    """
    Build lookup tables for multiple molecules by computing spectra at gridded column densities.
    
    Processes molecules serially, calling compute_molecule_lookup_table for each one.
    Each lookup table contains the column density grid, corresponding spectra, and
    log-transformed column densities for efficient interpolation.
    
    Parameters
    ----------
    mol_list : list of Molecule
        List of molecular species (as molsim Molecule objects) to process.
    labels : list of str
        Corresponding names/labels for each molecule.
    log_columns : array_like
        Array of column density values (in cm^-2) at which to compute spectra.
    tempInput : float
        Excitation temperature (K) for molecular transitions.
    dv_value : float
        Line width (km/s) for Gaussian line profiles.
    cont : Continuum
        Continuum object defining background radiation.
    ll0 : float
        Lower frequency bound (MHz) for simulations.
    ul0 : float
        Upper frequency bound (MHz) for simulations.
    dataScrape : Observation
        Observation object containing frequency grid.
    vlsr_value : float
        Source velocity (km/s) relative to LSR.
    
    Returns
    -------
    lookup_tables : dict
        Dictionary mapping molecule labels to their lookup table data:
        - 'column_grid': array of column densities
        - 'spectra_grid': 2D array of synthetic spectra
        - 'log_column_grid': log10 of column densities for interpolation
    """

    lookup_tables = {}

    for mol, label in zip(mol_list, labels):
        label, spectra_grid = compute_molecule_lookup_table(mol, label, log_columns, tempInput, dv_value, cont, ll0, ul0, dataScrape, vlsr_value, source_size)
        lookup_tables[label] = {
            'column_grid': log_columns,
            'spectra_grid': spectra_grid,
            'log_column_grid': np.log10(log_columns)
        }


    return lookup_tables


def setup_interpolators(lookup_tables):
    """
    Create linear interpolation functions for fast spectrum evaluation at arbitrary column densities.
    
    For each molecule's lookup table, constructs a scipy interp1d object that interpolates
    between pre-computed spectra in log(column density) space. This enables rapid spectrum
    generation during optimization without re-running full simulations.
    
    Parameters
    ----------
    lookup_tables : dict
        Dictionary of lookup tables (output from build_lookup_tables_serial).
        Each entry contains 'log_column_grid' and 'spectra_grid'.
    
    Returns
    -------
    lookup_tables : dict
        Updated dictionary with 'interpolator' function added to each molecule's entry.
        Interpolators use linear interpolation with extrapolation beyond grid bounds.
    """

    for label in lookup_tables:
        table = lookup_tables[label]
        log_columns = table['log_column_grid']
        spectra = table['spectra_grid']

        interp_func = interp1d(
            log_columns,
            spectra,
            kind='linear',
            axis=0,
            bounds_error=False,
            fill_value='extrapolate'
        )
        '''

        interp_func = PchipInterpolator(
            log_columns,
            spectra,
            axis=0,
            extrapolate=True
        )
        '''
        table['interpolator'] = interp_func
    return lookup_tables

def get_spectrum_from_lookup(label, column_density, lookup_tables):
    """
    Retrieve synthetic spectrum for a molecule at a specific column density via interpolation.
    
    Uses the pre-built interpolation function to quickly evaluate the spectrum at any
    column density, not just grid points. Works in log(column density) space for
    better interpolation accuracy across orders of magnitude.
    
    Parameters
    ----------
    label : str
        Molecule identifier matching a key in lookup_tables.
    column_density : float
        Column density (in cm^-2) at which to evaluate the spectrum.
    lookup_tables : dict
        Dictionary of lookup tables with interpolators (from setup_interpolators).
    
    Returns
    -------
    spectrum : ndarray
        1D array of intensity values at the observation's frequency grid.
    """
    """Get spectrum for one molecule at arbitrary column density using interpolation."""
    table = lookup_tables[label]
    interp_func = table['interpolator']
    log_col = np.log10(column_density)
    return interp_func(log_col)

def simulate_sum_lookup(columns, labels, lookup_tables):
    """
    Generate total synthetic spectrum as the sum of multiple molecular contributions.
    
    For each molecule in the input list, retrieves its spectrum at the specified
    column density and adds it to create the composite model spectrum. This is the
    forward model used during optimization.
    
    Parameters
    ----------
    columns : array_like
        Array of column densities (in cm^-2), one per molecule.
    labels : list of str
        Molecule identifiers corresponding to column densities.
    lookup_tables : dict
        Dictionary of lookup tables with interpolators.
    
    Returns
    -------
    total_spectrum : ndarray
        1D array containing the summed intensity profile across all molecules.
    """
    total_spectrum = None
    expected_length = None
    
    for column_density, label in zip(columns, labels):
        spectrum = get_spectrum_from_lookup(label, column_density, lookup_tables)
        
        # Determine expected length from first non-empty spectrum
        if expected_length is None and spectrum.size > 0:
            expected_length = spectrum.size
        
        # Replace empty spectra with zeros
        if spectrum.size == 0:
            if expected_length is not None:
                spectrum = np.zeros(expected_length)
                warnings.warn(f"Empty spectrum generated for {label}, using zeros")
            else:
                warnings.warn(f"Empty spectrum generated for {label}, skipping (no reference length yet)")
                continue
        
        if total_spectrum is None:
            total_spectrum = spectrum.copy()
        else:
            total_spectrum += spectrum
    
    if total_spectrum is None:
        raise ValueError("All molecular spectra are empty. Cannot proceed with fitting.")
    
    return total_spectrum

def residuals_lookup(columns, labels, lookup_tables, y_exp):
    """
    Compute residuals between synthetic and observed spectra for optimization.
    
    This is the objective function minimized by scipy's least_squares optimizer.
    Calculates the difference between the model (sum of molecular contributions)
    and experimental spectrum at each frequency point.
    
    Parameters
    ----------
    columns : array_like
        Current column density values (in cm^-2) for each molecule being optimized.
    labels : list of str
        Molecule identifiers corresponding to column densities.
    lookup_tables : dict
        Dictionary of lookup tables with interpolators.
    y_exp : ndarray
        Observed intensity spectrum (1D array).
    
    Returns
    -------
    residuals : ndarray
        1D array of (model - observed) differences at each frequency point. Scipy optimizes the squared differences.
    """
    y_sim = simulate_sum_lookup(columns, labels, lookup_tables)
    return y_sim - y_exp

def filter_lookup_tables(lookup_tables, mol_list, labels, keep_labels):
    """
    Filter molecules and lookup tables based on quality control criteria.
    
    Removes molecules that don't significantly contribute to the fit or fail
    quality checks. Creates new filtered versions of lookup tables, molecule
    objects, and label lists containing only the molecules to keep.
    
    Parameters
    ----------
    lookup_tables : dict
        Full lookup tables for all molecules.
    mol_list : list
        Original list of Molecule objects.
    labels : list
        Original molecule names corresponding to mol_list.
    keep_labels : list
        Labels of molecules that passed quality control.
    
    Returns
    -------
    new_lookup_tables : dict
        Filtered lookup tables containing only kept molecules.
    new_mol_list : list
        Filtered list of Molecule objects.
    new_labels : list
        Filtered list of molecule labels.
    """

    new_lookup_tables = {lab: lookup_tables[lab] for lab in keep_labels if lab in lookup_tables}
    new_indices = [i for i, lab in enumerate(labels) if lab in keep_labels]
    new_mol_list = [mol_list[i] for i in new_indices]
    new_labels = [labels[i] for i in new_indices]
    return new_lookup_tables, new_mol_list, new_labels

def fit_spectrum_lookup(mol_list, labels, initial_columns, y_exp, bounds,
                       tempInput, dv_value, cont, ll0, ul0, dataScrape,
                       column_range=(1e10, 1e20), n_grid_points=30, vlsr_value = None, source_size = 1.E20, stricter = False):
    """
    Perform spectral fitting using pre-computed lookup tables and nonlinear optimization.
    
    Main fitting function that builds lookup tables across a column density grid,
    sets up interpolators, and optimizes molecular column densities to minimize
    residuals between model and observed spectra. Uses scipy's trust-region
    reflective algorithm with bounded optimization.
    
    Parameters
    ----------
    mol_list : list of Molecule objects
        molsim Molecule objects to include in the fit.
    labels : list of str
        Names/identifiers for each molecule.
    initial_columns : array_like
        Initial guess for column densities (in cm^-2), one per molecule.
    y_exp : ndarray
        Observed intensity spectrum to fit.
    bounds : tuple of array_like
        Lower and upper bounds for column densities: (lower_bounds, upper_bounds).
    tempInput : float
        Excitation temperature (K) for simulations.
    dv_value : float
        Line width (km/s) for line profiles.
    cont : Continuum
        molsim Continuum background object.
    ll0 : float
        Lower frequency bound array (MHz).
    ul0 : float
        Upper frequency bound array (MHz).
    dataScrape : Observation
        Observation object with frequency grid.
    column_range : tuple, optional
        (min, max) column density range for lookup table grid. Default (1e10, 1e20).
    n_grid_points : int, optional
        Number of grid points for lookup table. Default 30.
    vlsr_value : float, optional
        Source velocity (km/s) relative to LSR. Default None.
    
    Returns
    -------
    lookup_tables : dict
        Built lookup tables with interpolators for all molecules.
    result : OptimizeResult
        Scipy optimization result object containing fitted column densities (result.x),
        residuals, convergence information, and other optimization diagnostics.
    """
    log_columns = np.logspace(np.log10(column_range[0]), np.log10(column_range[1]), n_grid_points)

    #print(log_columns)
    lookup_tables = build_lookup_tables_serial(
        mol_list, labels, log_columns, tempInput, dv_value, cont, ll0, ul0, dataScrape, vlsr_value, source_size
    )
    lookup_tables = setup_interpolators(lookup_tables)

    if stricter == False:
        result = least_squares(
            residuals_lookup,
            x0=initial_columns,
            bounds=bounds,
            args=(labels, lookup_tables, y_exp),
            method='trf',
            verbose=2,
            ftol=1e-6,
            max_nfev=23
        )
    else:
        result = least_squares(
            residuals_lookup,
            x0=initial_columns,
            bounds=bounds,
            args=(labels, lookup_tables, y_exp),
            method='trf',
            verbose=2,
            ftol=1e-6,
            max_nfev=30
        )
    return lookup_tables, result

def get_fitted_spectrum_lookup(fitted_columns, labels, lookup_tables):
    """
    Generate the best-fit total spectrum using optimized column densities.
    
    After optimization completes, this function creates the final model spectrum
    by summing molecular contributions at their fitted column density values.
    
    Parameters
    ----------
    fitted_columns : array_like
        Optimized column densities (in cm^-2) from the fitting procedure.
    labels : list of str
        Molecule identifiers.
    lookup_tables : dict
        Dictionary of lookup tables with interpolators.
    
    Returns
    -------
    spectrum : ndarray
        1D array of the total fitted intensity spectrum.
    """
    return simulate_sum_lookup(fitted_columns, labels, lookup_tables)

def get_individual_contributions_lookup(fitted_columns, labels, lookup_tables):
    """
    Extract individual molecular spectra at their best-fit column densities.
    
    Decomposes the total fit into separate contributions from each molecule,
    useful for understanding which molecules dominate specific spectral features
    and for quality control analysis.
    
    Parameters
    ----------
    fitted_columns : array_like
        Best-fit column densities (in cm^-2) for each molecule.
    labels : list of str
        Molecule identifiers.
    lookup_tables : dict
        Dictionary of lookup tables with interpolators.
    
    Returns
    -------
    contributions : dict
        Dictionary mapping molecule labels to their individual spectrum arrays (1D).
        Each spectrum shows that molecule's contribution to the total fit.
    """
    contributions = {}
    expected_length = None
    
    # First pass: determine expected length
    for column_density, label in zip(fitted_columns, labels):
        spectrum = get_spectrum_from_lookup(label, column_density, lookup_tables)
        if spectrum.size > 0:
            expected_length = spectrum.size
            break
    
    # Second pass: get contributions, replacing empty with zeros
    for column_density, label in zip(fitted_columns, labels):
        spectrum = get_spectrum_from_lookup(label, column_density, lookup_tables)
        
        if spectrum.size == 0 and expected_length is not None:
            spectrum = np.zeros(expected_length)
            warnings.warn(f"Empty spectrum generated for {label}, using zeros in contributions")
        
        contributions[label] = spectrum
    
    return contributions

def plot_simulation_vs_experiment_html_bokeh_compact_float32(
    y_exp, mol_list, best_columns, labels, filename, ll0, ul0, observation,
    peak_freqs, peak_intensities, temp, dv_value, vlsr_value, cont, direc, subdirec, save_individual_contributions, sourceSize=1.0E20,
    peak_window=1.0, max_initial_traces=0, save_html=True
):
    """
    Create interactive HTML visualization of fitted spectra using Bokeh.
    
    Generates a comprehensive interactive plot showing experimental spectrum, total fitted
    spectrum, and individual molecular contributions. Features include toggleable traces
    via checkbox controls, hover tooltips, and automated peak analysis. Also exports peak
    identification results to CSV.
    
    Parameters
    ----------
    y_exp : array_like
        Experimental intensity spectrum (converted to float32 internally for memory purposes).
    mol_list : list of Molecule objects
        molsim Molecule species included in the fit.
    best_columns : array_like
        Best-fit column densities (in cm^-2) for each molecule.
    labels : list of str
        Molecule names for plot legend.
    filename : str
        Output HTML filename (without directory path).
    ll0 : float
        Lower frequency bound array (MHz).
    ul0 : float
        Upper frequency bound array (MHz).
    observation : Observation
        moslsim Observation object containing frequency grid and spectrum data.
    peak_freqs : array_like
        Frequencies (MHz) of identified spectral peaks for analysis.
    peak_intensities : array_like
        Peak intensity values corresponding to peak_freqs.
    temp : float
        Excitation temperature (K) used in fit.
    dv_value : float
        Line width (km/s) used in fit.
    vlsr_value : float
        Source velocity (km/s) relative to LSR.
    cont : Continuum
        molsim Continuum background object.
    direc : str
        Output directory path for HTML file and CSV results.
    sourceSize : float, optional
        Source size parameter. Default 1.0E20.
    peak_window : float, optional
        Frequency window (MHz) around each peak for carrier identification. Default 1.0.
    max_initial_traces : int, optional
        Maximum number of molecular traces visible initially. Default 20.
    save_html : bool, optional
        If True, saves plot to HTML file. If False, just returns layout. Default: True.
    
    Returns
    -------
    maxSimInts : list of float
        Maximum intensity value for each molecular spectrum.
    peak_results : list of dict
        Peak analysis results for each identified peak containing:
        - peak_freq: peak frequency (MHz)
        - experimental_intensity_max: observed peak intensity
        - total_simulated_intensity: fitted peak intensity
        - difference: residual at peak
        - carrier_molecules: list of molecules contributing to peak
    peak_df : DataFrame
        Pandas DataFrame of peak_results, also saved as 'final_peak_results.csv'.
    layout : Bokeh layout object
        The plot layout for displaying in notebooks or further manipulation.
    
    Notes
    -----
    - All spectral arrays converted to float32 to reduce memory usage
    - Molecules contributing >20% of observed peak intensity are identified as carriers
    - Interactive controls include Show/Hide All buttons and individual trace checkboxes
    - Hover tool displays trace name, frequency, and intensity at cursor position
    """

    # Initial garbage collection for clean slate
    gc.collect()
    print("Starting plot generation (memory cleanup complete)")

    # Convert experimental spectrum to float32
    freqs = observation.spectrum.frequency.astype(np.float32)
    y_exp = y_exp.astype(np.float32)
    total_sim = np.zeros_like(y_exp, dtype=np.float32)
    individual_sims = []
    maxSimInts = []

    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]

    # Main plot with WebGL backend for better performance
    p = figure(
        width=1200,
        height=700,
        title="Observed vs Simulated Spectra",
        x_axis_label="Frequency (MHz)",
        y_axis_label="Intensity",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        output_backend="webgl"  # Use WebGL for large datasets
    )

    renderers = []
    names = []

    # Experimental spectrum using ColumnDataSource for efficiency
    exp_source = ColumnDataSource(data=dict(x=freqs, y=y_exp))
    r_exp = p.line('x', 'y', source=exp_source, line_color='black', 
                   line_width=2, visible=True, name="Observations")
    renderers.append(r_exp)
    names.append("Observations")


    # Individual molecules
    for i, (mol, col) in enumerate(zip(mol_list, best_columns)):
        src = Source(
            Tex=temp, column=col, size=sourceSize,
            dV=dv_value, velocity=vlsr_value, continuum=cont
        )
        sim = Simulation(
            mol=mol, ll=ll0, ul=ul0, source=src,
            line_profile='Gaussian', use_obs=True, observation=observation
        )

        spec = np.array(sim.spectrum.int_profile, dtype=np.float32)  # convert to float32
        
        # Handle empty spectra
        if spec.size == 0:
            spec = np.zeros_like(total_sim)
            warnings.warn(f"Empty spectrum generated for {labels[i] if i < len(labels) else f'Molecule {i+1}'}, using zeros")
        
        
        individual_sims.append(spec)
        total_sim += spec
        maxSimInts.append(np.max(spec))

        mol_name = labels[i] if i < len(labels) else f"Molecule {i+1}"
        if save_individual_contributions:
            save_indiv_direc = os.path.join(subdirec,'individual_contributions')
            np.savetxt(os.path.join(save_indiv_direc, mol_name + '_simulated_spectrum.txt'), np.column_stack((freqs, spec)), delimiter='\t')



        visible = i < max_initial_traces
        
        # Use ColumnDataSource for better memory management
        mol_source = ColumnDataSource(data=dict(x=freqs, y=spec))
        r = p.line(
            'x', 'y', source=mol_source,
            line_color=base_colors[i % len(base_colors)],
            line_width=1.8,  # slightly thinner lines
            line_dash="dashed",
            visible=visible,
            name=mol_name
        )
        renderers.append(r)
        names.append(mol_name)
        
        # Free Source and Simulation objects immediately
        del src, sim
        
        # Periodic garbage collection every 10 molecules
        if (i + 1) % 10 == 0:
            gc.collect()
            print(f"  Progress: {i+1}/{len(mol_list)} molecules (GC performed)")

    # Total simulated spectrum using ColumnDataSource
    total_source = ColumnDataSource(data=dict(x=freqs, y=total_sim))

    if save_individual_contributions:
        save_indiv_direc = os.path.join(subdirec,'individual_contributions')
        np.savetxt(os.path.join(save_indiv_direc, 'total_simulated_spectrum.txt'), np.column_stack((freqs, total_sim)), delimiter='\t')
        remaining_intensity = y_exp - total_sim
        np.savetxt(os.path.join(save_indiv_direc, 'simulation_residual.txt'), np.column_stack((freqs, remaining_intensity)), delimiter='\t')


    r_total = p.line('x', 'y', source=total_source, line_color='red', 
                    line_width=2, visible=True, name="Total Simulated")
    renderers.append(r_total)
    names.append("Total Simulated")

    # Hover tool
    hover = HoverTool(tooltips=[
        ("Trace", "$name"),
        ("Frequency", "$x{0.000} MHz"),
        ("Intensity", "$y{0.0000}")
    ])
    p.add_tools(hover)

    # Checkbox legend
    checkbox_group = CheckboxGroup(labels=names, active=[0, len(names)-1], width=300, height=500)
    checkbox_callback = CustomJS(args=dict(renderers=renderers, checkbox=checkbox_group), code="""
        for (let i = 0; i < renderers.length; i++) {
            renderers[i].visible = checkbox.active.includes(i);
        }
    """)
    checkbox_group.js_on_change("active", checkbox_callback)

    # Show/Hide buttons
    mol_indices = list(range(1, len(names)-1))
    show_button = Button(label="Show All Molecules", button_type="success", width=150)
    hide_button = Button(label="Hide All Molecules", button_type="primary", width=150)
    show_button.js_on_click(CustomJS(args=dict(checkbox=checkbox_group, mol_indices=mol_indices),
                                     code="checkbox.active = Array.from(new Set([...checkbox.active, ...mol_indices])); checkbox.change.emit();"))
    hide_button.js_on_click(CustomJS(args=dict(checkbox=checkbox_group, mol_indices=mol_indices),
                                     code="checkbox.active = checkbox.active.filter(i => !mol_indices.includes(i)); checkbox.change.emit();"))

    # Layout
    controls = column(row(show_button, hide_button), checkbox_group)
    layout = row(p, controls)

    # Conditionally save HTML
    if save_html:
        print("Preparing to save HTML file...")
        gc.collect()  # CRITICAL: Free all possible memory before Bokeh serialization
        print("  Memory cleanup complete, saving to disk...")
        
        filename = os.path.join(subdirec, filename)
        output_file(filename)
        save(layout)
        
        # Check file size to detect potential truncation
        filesize_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"Interactive plot saved to {filename} ({filesize_mb:.1f} MB)")

    # Peak analysis
    peak_results = []
    for peak, exp_intensity_max in zip(peak_freqs, peak_intensities):
        idxs = np.where((freqs >= peak - peak_window) & (freqs <= peak + peak_window))[0]
        if len(idxs) == 0:
            continue

        sim_intensities = [np.max(sim[idxs]) for sim in individual_sims]
        total_sim_intensity = np.max(total_sim[idxs])
        threshold = 0.1 * exp_intensity_max

        carriers = [
            labels[i] if i < len(labels) else f"Molecule {i+1}"
            for i, inten in enumerate(sim_intensities)
            if inten >= threshold and inten > 0
        ] or ['Unidentified']

        diff = exp_intensity_max - total_sim_intensity
        peak_results.append({
            'peak_freq': float(peak),  # ensure float32 not object
            'experimental_intensity_max': float(exp_intensity_max),
            'total_simulated_intensity': float(total_sim_intensity),
            'difference': float(diff),
            'carrier_molecules': carriers
        })

    peak_results.sort(key=lambda x: x['experimental_intensity_max'], reverse=True)
    peak_df = pd.DataFrame(peak_results)
    
    if save_html:
        peak_df.to_csv(os.path.join(subdirec, 'final_peak_results.csv'), index=False)

    # Final cleanup
    gc.collect()

    return maxSimInts, peak_results, peak_df, layout




def full_fit(direc, subdirec, assigner, dataScrape, tempInput, dv_value, dv_value_freq, ll0, ul0, 
             vlsr_value, actualFrequencies, intensities, rms, cont, force_include_mols, 
             sourceSize, column_density_range, resolution, dv_value_freq_og, stricter, save_individual_contributions, rms_arr_full,
             number_iterations=0):
    """
    Execute complete spectral fitting workflow with quality control and visualization.
    
    This is the main high-level function that orchestrates the entire fitting process:
    1. Loads molecular catalog data from CDMS and JPL databases
    2. Performs quality control on assigned molecules (removes hyperfine duplicates, handles database priorities)
    3. Runs initial fit with all molecules using lookup table optimization
    4. Applies quality control filters to remove weak or insignificant contributors
    5. Performs refined fit with filtered molecule set (repeated for number_iterations - 1 times)
    6. Generates interactive visualization and exports results
    
    Quality Control Criteria:
    - Removes molecules with peak intensity ≤ 2.5 x RMS and that contribute in a minor way to the fit quality
    - Prioritizes CDMS entries over JPL for duplicate molecules
    
    Parameters
    ----------
    direc : str
        Main directory containing catalog files and for output storage.
        Expected subdirectories: 'cdms_pkl/', 'jpl_pkl/'
        Expected catalog files: 'all_cdms_final_official.csv', 'all_jpl_final_official.csv'
    dataScrape : Observation
        molsim Observation object containing observational spectrum and frequency grid.
    tempInput : float
        Excitation temperature (K) for molecular transitions.
    dv_value : float
        Line width (km/s) for Gaussian line profiles.
    ll0 : float
        Lower frequency bound array (MHz) for fitting range.
    ul0 : float
        Upper frequency bound array (MHz) for fitting range.
    vlsr_value : float
        Source velocity (km/s) relative to the local standard of rest.
    actualFrequencies : array_like
        Frequencies (MHz) of identified spectral peaks for carrier analysis.
    intensities : array_like
        Intensity values at peak frequencies.
    rms : float
        Root-mean-square noise level of the spectrum for quality filtering.
    cont : Continuum
        molsim Continuum background object for simulations.
    force_include_mols : list
        List of molecule names to always include in the fit (won't be filtered out).
    sourceSize : float
        Source size in arcseconds.
    column_density_range : tuple
        (min, max) column density bounds for fitting.
    resolution : float
        Spectral resolution for peak finding.
    dv_value_freq_og : float
        Original linewidth in frequency units (MHz).
    stricter : bool
        If True, applies stricter filtering criteria.
    number_iterations : int, optional
        Number of fitting iterations to perform. Must be 0 or >= 2. Default is 2.
        Each iteration after the first performs molecule filtering followed by refitting.
        If num_iterations = 0, will loop until convergence.
    
    Returns
    -------
    delMols : list
        List of molecule names that were removed during quality control
    cdDF : DataFrame
        Best-fit column densities and SMILES strings for retained molecules
    peak_df : DataFrame
        Peak identification and carrier analysis results
    internal_data : dict
        Dictionary containing all data needed to recreate the plot in a notebook
    
    Notes
    -----
    - Initial fit uses column density bounds specified by column_density_range
    - Lookup tables span 30 grid points for accurate interpolation
    - Memory cleanup performed after optimization via garbage collection
    - number_iterations controls how many rounds of fitting occur:
        * number_iterations=2: Initial fit → filter → final fit (default)
        * number_iterations=3: Initial fit → filter → fit → filter → final fit
        * etc.
    
    """

    cdmsDirec = os.path.join(direc, 'cdms_pkl/')
    cdmsFullDF = pd.read_csv(os.path.join(direc, 'all_cdms_final_official.csv'))
    df_cdms = cdmsFullDF
    cdmsSmiles = list(cdmsFullDF['smiles'])
    cdmsMols = list(cdmsFullDF['mol'])
    cdms_mols = list(df_cdms['mol'])
    cdms_tags = list(df_cdms['tag'])
    jplDirec = os.path.join(direc, 'jpl_pkl/')
    jplFullDF = pd.read_csv(os.path.join(direc, 'all_jpl_final_official.csv'))
    df_jpl = jplFullDF
    jplSmiles = list(jplFullDF['smiles'])
    jplMols = list(jplFullDF['name'])
    jpl_mols = list(df_jpl['name'])
    jpl_tags = list(df_jpl['tag'])
    mol_list = []
    labels = []

    #print(rms_arr_full)
    
    assignedMols, blended_single_mols = get_assigned_molecules(assigner)


    for i in force_include_mols:
        if i in cdms_mols:
            if (i, 'CDMS') not in assignedMols:
                assignedMols.append((i, 'CDMS'))
        elif i in jpl_mols:
            if (i,'JPL') not in assignedMols:
                assignedMols.append((i,'JPL'))
        else:
            warnings.warn(f"{i} was forced to be included in the fit but is not found in the CDMS or JPL databases. Please check the inputted molecule name.", UserWarning)
    
    blended_mols_final = []
    # Hyperfine quality control
    delList1 = []
    for i in assignedMols:
        if '[hf]' in i[0]:
            i_clean = i[0].replace(" [hf]", "")
            for g in assignedMols:
                if g[0] == i_clean:
                    if g[0] not in force_include_mols:
                        delList1.append(g[0])

    assignedMols = [i for i in assignedMols if i[0] not in delList1 and i[0] not in blended_mols_final]

    # Quality control for duplicated entries
    result = {}
    for key, source in assignedMols:
        if key not in result:
            result[key] = source
        else:
            if result[key] != 'CDMS' and source == 'CDMS':
                result[key] = source

    assignedMols = list(result.items())

    if len(assignedMols) == 0:
        raise ValueError(
            "No molecules were assigned. Fitting will therefore fail.")


    # Retrieving molecule objects
    for x in assignedMols:
        if x[1] == 'CDMS':
            idx = cdms_mols.index(x[0])
            tag = cdms_tags[idx]
            tagString = f"{tag:06d}"
            molDirec = cdmsDirec + tagString + '.pkl'
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            mol_list.append(mol)
            labels.append(x[0])
        elif x[1] == 'JPL':
            idx = jpl_mols.index(x[0])
            tag = jpl_tags[idx]
            tagString = str(tag)
            molDirec = jplDirec + tagString + '.pkl'
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            mol_list.append(mol)
            labels.append(x[0])
        else:
            if x[0] in cdms_mols:
                idx = cdms_mols.index(x[0])
                tag = cdms_tags[idx]
                tagString = f"{tag:06d}"
                molDirec = cdmsDirec + tagString + '.pkl'
                with open(molDirec, 'rb') as md:
                    mol = pickle.load(md)
                mol_list.append(mol)
                labels.append(x[0])
            elif x[0] in jpl_mols:
                idx = jpl_mols.index(x[0])
                tag = jpl_tags[idx]
                tagString = str(tag)
                molDirec = jplDirec + tagString + '.pkl'
                with open(molDirec, 'rb') as md:
                    mol = pickle.load(md)
                mol_list.append(mol)
                labels.append(x[0])
            else:
                print('ignoring')
                print(x)

    # Initializing column densities 
    if column_density_range[0] == 1.e10 and column_density_range[1] == 1.e20:
        first_guess = 1.e14
    else:
        first_guess = (column_density_range[0]+column_density_range[1])/2
    initial_columns = np.full(len(mol_list), first_guess)
    bounds = (np.full(len(mol_list), column_density_range[0]), np.full(len(mol_list), column_density_range[1]))
    y_exp = np.array(dataScrape.spectrum.Tb)

    if number_iterations == 0:
        number_iterations = 100000 #set so that it will loop until convergence
        print('\nWill conduct iterations of fitting until no more molecules are removed.')
    else:
        print(f'\nWill conduct {number_iterations} iterations of fitting.')

    converged = False

    #print(labels)

    # ========== FIRST ITERATION ==========
    if number_iterations != 100000:
        print(f'Fitting iteration 1/{number_iterations}')
    else:
        print('Fitting Iteration 1')

    lookup_tables, result = fit_spectrum_lookup(
        mol_list=mol_list,
        labels=labels,
        initial_columns=initial_columns,
        y_exp=y_exp,
        bounds=bounds,
        tempInput=tempInput,
        dv_value=dv_value,
        cont=cont,
        ll0=ll0,
        ul0=ul0,
        dataScrape=dataScrape,
        column_range=(column_density_range[0], column_density_range[1]),
        n_grid_points=30,
        vlsr_value=vlsr_value,
        source_size=sourceSize,
        stricter=stricter
    )

    fitted_columns = result.x
    
    # ========== ITERATIVE FILTERING AND FITTING ==========
    # Track all deleted molecules across iterations
    all_delMols = []
    
    for iteration in range(2, number_iterations + 1): #loop through (fitting_iterations - 1) times
        if number_iterations != 100000:
            print(f'\nFiltering molecules before iteration {iteration}/{number_iterations}')
        else:
            print(f'\nFiltering molecules before iteration {iteration}')

        
        # Get individual contributions
        fitted_spectrum = get_fitted_spectrum_lookup(fitted_columns, labels, lookup_tables)
        individual_contributions = get_individual_contributions_lookup(fitted_columns, labels, lookup_tables)

        cont_array = []
        for i in labels:
            cont_array.append(individual_contributions[i])

        cont_array = np.array(cont_array)
        summed_spectrum = np.sum(cont_array, axis=0)
        ssd_og = np.sum((y_exp - summed_spectrum) ** 2)
        leave_one_out_ssd = []

        # Checking relative contribution to fit quality for each molecule
        for i in range(cont_array.shape[0]):
            spectrum_wo_i = summed_spectrum - cont_array[i]
            ssd = np.sum((y_exp - spectrum_wo_i) ** 2)
            leave_one_out_ssd.append(ssd)

        delMols = []

        #Steps to filter out incorrect assignments
        freqs = dataScrape.spectrum.frequency
        y_exp = dataScrape.spectrum.Tb

        # Finding the carrier of each line
        all_carriers = {}
        non_blended_lines = {}
        num_assignments = {}

        for l in labels:
            all_carriers[l] = 0
            non_blended_lines[l] = 0
            num_assignments[l] = 0 

        peak_window = 0.5 * dv_value_freq
        for peak, exp_intensity_max in zip(actualFrequencies, intensities):
            idxs = np.where((freqs >= peak - peak_window) & (freqs <= peak + peak_window))[0]
            if len(idxs) == 0:
                continue

            threshold = 0.2 * exp_intensity_max

            sim_intensities = [np.max(individual_contributions[lbl][idxs]) for lbl in labels]
            carriers = [
                labels[i]
                for i, inten in enumerate(sim_intensities)
                if inten >= threshold and inten > 0
            ]

            for c in carriers:
                num_assignments[c] += 1

            if len(carriers) == 1:
                non_blended_lines[carriers[0]] += 1

            for i, inten in enumerate(sim_intensities):
                if inten >= threshold and inten > 0:
                    if inten/exp_intensity_max > 0.4 or len(carriers) == 1:
                        all_carriers[labels[i]] = all_carriers[labels[i]] + 1

        '''
        Removing molecules if their maximum intensity is < 2.5 sigma and they don't contribute significantly
        '''
        for i in range(len(labels)):
            snr_arr = cont_array[i]/rms_arr_full
            #maxInt = max(cont_array[i])
            if max(snr_arr) <= 2.5:
                diff_ssd = leave_one_out_ssd[i]-ssd_og
                if (diff_ssd/ssd_og) < 0.1:
                    if labels[i] not in force_include_mols:
                        delMols.append(labels[i])

        '''
        Removing molecules if only assigned to one blended line and is less than 40% of its strength
        '''
        for de in all_carriers:
            if all_carriers[de] == 0 and de not in force_include_mols:
                delMols.append(de)

        '''
        Extra strict removal policy if molecule is only in blended lines or only assigned to one line.
        '''
        if stricter == True:
            for l in labels:
                if l not in delMols:
                    if non_blended_lines[l] == 0 or num_assignments[l] <= 1:
                        snr_arr = individual_contributions[l]/rms_arr_full
                        #if max(individual_contributions[l]) <= 3.0*rms:
                        if max(snr_arr) <= 3.0:
                            delMols.append(l)

        '''
        Removing molecules if too many lines are missing.
        Does this by checking whether the integral of the observations is significantly lower than the integral of the simulated spectrum. 
        '''
        for i in labels:
            if i not in delMols:
                indiv_intensity = individual_contributions[i]
                peak_indicesIndiv = find_peaks(freqs, indiv_intensity, res=resolution, min_sep=max(resolution * ckm / np.amax(freqs),0.5*dv_value_freq), is_sim=True)
                peak_freqs_full = freqs[peak_indicesIndiv]
                peak_ints_full = abs(indiv_intensity[peak_indicesIndiv])
                peak_rms = rms_arr_full[peak_indicesIndiv]
                mask = peak_ints_full > 2*peak_rms 
                #mask = peak_ints_full > 2*rms #get all peaks above 2 sigma 
                peak_freqs_filtered = peak_freqs_full[mask]
                peak_ints_filtered = peak_ints_full[mask]
                missingCount = 0
                missingMaxCount = 0
                if len(peak_freqs_filtered) > 0:
                    for z in range(len(peak_freqs_filtered)):
                        # Find indices within the frequency window
                        freq_window = (freqs >= peak_freqs_filtered[z] - 0.5*dv_value_freq) & \
                                    (freqs <= peak_freqs_filtered[z] + 0.5*dv_value_freq)
                        
                        # Get the frequency and intensity values in this window
                        freqs_in_window = freqs[freq_window]
                        sim_intensity_in_window = abs(indiv_intensity[freq_window])
                        obs_intensity_in_window = abs(y_exp[freq_window])

                        #max_sim_intensity_in_window = max(sim_intensity_in_window)
                        #max_obs_intensity_in_window = max(obs_intensity_in_window)
                        
                        # Calculate the integral (using trapezoidal rule)
                        integral_sim = np.trapz(sim_intensity_in_window, freqs_in_window)
                        integral_obs = np.trapz(obs_intensity_in_window, freqs_in_window)
                        if (integral_obs/integral_sim) <= 0.30:
                            missingCount += 1

                        #if max_obs_intensity_in_window/max_sim_intensity_in_window <= 0.4:
                        #    missingMaxCount += 1



                    #print(i)
                    #print('missing count')
                    #print(missingCount/len(peak_freqs_filtered))
                    #print('missing max count')
                    #print(missingMaxCount/len(peak_freqs_filtered))
                    #print(max(individual_contributions[i])/rms)
                    

                    missingDeletion = False
                    snr_arr = individual_contributions[i]/rms_arr_full
                    #tiered checks of percent of missing lines based on maximum intensity
                    #if max(individual_contributions[i]) <= 3.1*rms:
                    if max(snr_arr) <= 3.1:
                        if missingCount/len(peak_freqs_filtered) >= 0.1:
                            missingDeletion = True
                    #if max(individual_contributions[i]) < 10*rms:
                    if max(snr_arr) < 10:
                        if missingCount/len(peak_freqs_filtered) >= 0.2:
                            missingDeletion = True
                    #if max(individual_contributions[i]) < 15*rms:
                    if max(snr_arr) < 15:
                        if missingCount/len(peak_freqs_filtered) >= 0.5:
                            missingDeletion = True

                    #deleting if too many missing lines 
                    if missingDeletion == True:
                        delMols.append(i)

                #now checking if there are any missing lines by maximum intensity
                mask = peak_ints_full > 1.5*peak_rms #get all peaks above 1.5 sigma
                peak_freqs_filtered = peak_freqs_full[mask]
                peak_ints_filtered = peak_ints_full[mask]
                missingMaxCount = 0
                if len(peak_freqs_filtered) > 0:
                    for z in range(len(peak_freqs_filtered)):
                        # Find indices within the frequency window
                        freq_window = (freqs >= peak_freqs_filtered[z] - 0.2*dv_value_freq) & \
                                    (freqs <= peak_freqs_filtered[z] + 0.2*dv_value_freq)
                        
                        # Get the frequency and intensity values in this window
                        freqs_in_window = freqs[freq_window]
                        sim_intensity_in_window = abs(indiv_intensity[freq_window])
                        obs_intensity_in_window = abs(y_exp[freq_window])

                        max_sim_intensity_in_window = max(sim_intensity_in_window)
                        max_obs_intensity_in_window = max(obs_intensity_in_window)
                        
                    

                        if max_obs_intensity_in_window/max_sim_intensity_in_window <= 0.4:
                            missingMaxCount += 1

                    #print('missing max count')
                    #print(missingMaxCount/len(peak_freqs_filtered))
                    if missingMaxCount/len(peak_freqs_filtered) >= 0.18: #rule out if theres too many missing lines
                        snr_arr = individual_contributions[i]/rms_arr_full
                        #if max(individual_contributions[i]) < 8*rms:
                        if max(snr_arr) < 8:
                            delMols.append(i)

                    #print(max(individual_contributions[i])/rms)


        
        #print(delMols)
        #print(force_include_mols)
        # Filtering lists of molecules and labels
        delMols = [i for i in delMols if i not in force_include_mols]
        #print(delMols)
    

        
        # Track deleted molecules
        all_delMols.extend([mol for mol in delMols if mol not in all_delMols])
        
        unique_delMols = list(set(delMols))
        print(f'Removed {len(unique_delMols)} molecules in this iteration')
        
        # Filter molecule lists and lookup tables
        keep_labels = [labels[i] for i in range(len(mol_list)) if labels[i] not in delMols]
        keep_mol_list = [mol_list[i] for i in range(len(mol_list)) if labels[i] not in delMols]
        
        filtered_lookup, filtered_mols, filtered_labels = filter_lookup_tables(
            lookup_tables, mol_list, labels, keep_labels
        )

        bounds_filtered = (np.full(len(filtered_labels), column_density_range[0]),
                          np.full(len(filtered_labels), column_density_range[1]))

        # Prepare initial conditions for next fit (BEFORE updating labels!)
        initial_columns_filtered = []
        # storing the best-fit abundances to initialize the next fit
        for z in keep_labels:
            idx = labels.index(z)  # Using OLD labels to index into fitted_columns
            initial_columns_filtered.append(fitted_columns[idx])

        initial_columns_filtered = np.array(initial_columns_filtered)

        # Update variables for next iteration (AFTER extracting initial columns)
        mol_list = keep_mol_list
        labels = keep_labels

        # Clean up old lookup tables
        if iteration > 2:
            del lookup_tables
            gc.collect()
        
        lookup_tables = filtered_lookup


        if iteration > 2 and len(delMols) == 0: #checking if converged after the second iteration.
            if number_iterations == 100000: #check if looping until convergence
                converged = True
                print(f'\nConverged after iteration {iteration-1}!\n')
                break #break out of loop if converged


        # ========== FIT AGAIN ==========
        if number_iterations != 100000:
            print(f'Fitting iteration {iteration}/{number_iterations}')
        else:
            print(f'Fitting iteration {iteration}')
        result_filtered = least_squares(
            residuals_lookup,
            x0=initial_columns_filtered,
            bounds=bounds_filtered,
            args=(filtered_labels, filtered_lookup, y_exp),
            method='trf',
            verbose=2,
            ftol=1e-7
        )

        fitted_columns = result_filtered.x

    # ========== CHECK IF MORE ITERATIONS NEEDED ==========
    # Run filtering check one more time to see if molecules would still be removed
    #converged is only true if num_iterations parameter is set to 0 and it converged.
    #If thats the case, don't need an additional check
    if converged == False: 
        print(f'\nChecking if additional iterations would be beneficial...')
        
        fitted_spectrum = get_fitted_spectrum_lookup(fitted_columns, labels, lookup_tables)
        individual_contributions = get_individual_contributions_lookup(fitted_columns, labels, lookup_tables)

        cont_array = []
        for i in labels:
            cont_array.append(individual_contributions[i])

        cont_array = np.array(cont_array)
        summed_spectrum = np.sum(cont_array, axis=0)
        ssd_og = np.sum((y_exp - summed_spectrum) ** 2)
        leave_one_out_ssd = []

        for i in range(cont_array.shape[0]):
            spectrum_wo_i = summed_spectrum - cont_array[i]
            ssd = np.sum((y_exp - spectrum_wo_i) ** 2)
            leave_one_out_ssd.append(ssd)

        potential_delMols = []

        # ========== RUN SAME FILTERING CRITERIA ==========
        freqs = dataScrape.spectrum.frequency
        y_exp_check = dataScrape.spectrum.Tb

        all_carriers = {}
        non_blended_lines = {}
        num_assignments = {}

        for l in labels:
            all_carriers[l] = 0
            non_blended_lines[l] = 0
            num_assignments[l] = 0 

        #finding the assigned carriers of all peaks
        peak_window = 0.5 * dv_value_freq
        for peak, exp_intensity_max in zip(actualFrequencies, intensities):
            idxs = np.where((freqs >= peak - peak_window) & (freqs <= peak + peak_window))[0]
            if len(idxs) == 0:
                continue

            threshold = 0.2 * exp_intensity_max

            sim_intensities = [np.max(individual_contributions[lbl][idxs]) for lbl in labels]
            carriers = [
                labels[i]
                for i, inten in enumerate(sim_intensities)
                if inten >= threshold and inten > 0
            ]

            for c in carriers:
                num_assignments[c] += 1

            if len(carriers) == 1:
                non_blended_lines[carriers[0]] += 1

            for i, inten in enumerate(sim_intensities):
                if inten >= threshold and inten > 0:
                    if inten/exp_intensity_max > 0.4 or len(carriers) == 1:
                        all_carriers[labels[i]] = all_carriers[labels[i]] + 1

        # Remove if maximum simulated intensity is too weak (<2.5 sigma)
        for i in range(len(labels)):
            #maxInt = max(cont_array[i])
            snr_arr = cont_array[i]/rms_arr_full
            #if maxInt <= 2.5*rms:
            if max(snr_arr) <= 2.5:
                diff_ssd = leave_one_out_ssd[i]-ssd_og
                if (diff_ssd/ssd_og) < 0.1:
                    if labels[i] not in force_include_mols:
                        potential_delMols.append(labels[i])

        '''
        Removing molecules if only assigned to one blended line and is less than 40% of its strength
        '''
        for de in all_carriers:
            if all_carriers[de] == 0 and de not in force_include_mols:
                potential_delMols.append(de)

        '''
        Stricter criteria if flag is True
        '''
        if stricter == True:
            for l in labels:
                if l not in potential_delMols:
                    if non_blended_lines[l] == 0 or num_assignments[l] <= 1:
                        snr_arr = individual_contributions[l]/rms_arr_full
                        #if max(individual_contributions[l]) <= 3.0*rms:
                        if max(snr_arr) <= 3.0:
                            potential_delMols.append(l)

        '''
        Finding if too many missing lines
        '''
        for i in labels:
            if i not in potential_delMols:
                indiv_intensity = individual_contributions[i]
                peak_indicesIndiv = find_peaks(freqs, indiv_intensity, res=resolution, min_sep=max(resolution * ckm / np.amax(freqs),0.5*dv_value_freq), is_sim=True)
                peak_freqs_full = freqs[peak_indicesIndiv]
                peak_ints_full = abs(indiv_intensity[peak_indicesIndiv])
                peak_rms = rms_arr_full[peak_indicesIndiv]
                print('peak_rms')
                print(peak_rms)
                mask = peak_ints_full > 2*peak_rms 
                peak_freqs_filtered = peak_freqs_full[mask]
                peak_ints_filtered = peak_ints_full[mask]


                missingCount = 0
                missingMaxCount = 0
                if len(peak_freqs_filtered) > 0:
                    for z in range(len(peak_freqs_filtered)):
                        freq_window = (freqs >= peak_freqs_filtered[z] - 0.3*dv_value_freq) & \
                                    (freqs <= peak_freqs_filtered[z] + 0.3*dv_value_freq)
                        
                        freqs_in_window = freqs[freq_window]
                        sim_intensity_in_window = abs(indiv_intensity[freq_window])
                        obs_intensity_in_window = abs(y_exp_check[freq_window])
                        
                        integral_sim = np.trapz(sim_intensity_in_window, freqs_in_window)
                        integral_obs = np.trapz(obs_intensity_in_window, freqs_in_window)
                        if (integral_obs/integral_sim) <= 0.30:
                            missingCount += 1




        
                    missingDeletion = False
                    snr_arr = individual_contributions[i]/rms_arr_full
                    #if max(individual_contributions[i]) <= 3.1*rms:
                    if max(snr_arr) <= 3.1:
                        if missingCount/len(peak_freqs_filtered) >= 0.1:
                            missingDeletion = True
                    #if max(individual_contributions[i]) < 10*rms:
                    if max(snr_arr) < 10:
                        if missingCount/len(peak_freqs_filtered) >= 0.2:
                            missingDeletion = True
                    #if max(individual_contributions[i]) < 15*rms:
                    if max(snr_arr) < 15:
                        if missingCount/len(peak_freqs_filtered) >= 0.5:
                            missingDeletion = True

                    if missingDeletion == True:
                        potential_delMols.append(i)


                #now checking if there are any missing lines by maximum intensity
                mask = peak_ints_full > 1.5*peak_rms #get all peaks above 1.5 sigma
                peak_freqs_filtered = peak_freqs_full[mask]
                peak_ints_filtered = peak_ints_full[mask]
                missingMaxCount = 0
                if len(peak_freqs_filtered) > 0:
                    for z in range(len(peak_freqs_filtered)):
                        # Find indices within the frequency window
                        freq_window = (freqs >= peak_freqs_filtered[z] - 0.2*dv_value_freq) & \
                                    (freqs <= peak_freqs_filtered[z] + 0.2*dv_value_freq)
                        
                        # Get the frequency and intensity values in this window
                        freqs_in_window = freqs[freq_window]
                        sim_intensity_in_window = abs(indiv_intensity[freq_window])
                        obs_intensity_in_window = abs(y_exp[freq_window])

                        max_sim_intensity_in_window = max(sim_intensity_in_window)
                        max_obs_intensity_in_window = max(obs_intensity_in_window)
                        
                    

                        if max_obs_intensity_in_window/max_sim_intensity_in_window <= 0.4:
                            missingMaxCount += 1

                    #print('missing max count')
                    #print(missingMaxCount/len(peak_freqs_filtered))
                    if missingMaxCount/len(peak_freqs_filtered) >= 0.18: #rule out if theres too many missing lines
                        snr_arr = individual_contributions[i]/rms_arr_full
                        #if max(individual_contributions[i]) < 8*rms:
                        if max(snr_arr) < 8:
                            delMols.append(i)        

        potential_delMols = [i for i in potential_delMols if i not in force_include_mols]
        potential_delMols = list(dict.fromkeys(potential_delMols))  # Remove duplicates while preserving order

        if len(potential_delMols) > 0:
            warnings.warn(
                f"\nAfter iteration {number_iterations}/{number_iterations}, {len(potential_delMols)} molecule(s) "
                f"would still be removed by the filtering criteria: {potential_delMols}\n"
                f"Consider increasing 'number_iterations' parameter (currently {number_iterations}) "
                f"to {number_iterations + 1} or higher for better convergence.",
                UserWarning
            )
            print(f"  Molecules that would be removed: {potential_delMols}")
        else:
            print(f"  No additional molecules would be removed. Fitting has converged.")

    # ========== FINAL OUTPUT (after all iterations) ==========

    # Free memory
    try:
        del lookup_tables
        gc.collect()
    except:
        pass

    # Plotting and saving
    maxSimInts, peak_results, peak_df, _ = plot_simulation_vs_experiment_html_bokeh_compact_float32(
        y_exp, mol_list, fitted_columns, labels, "fit_spectrum.html", ll0, ul0, dataScrape,
        actualFrequencies, intensities, tempInput, dv_value, vlsr_value, cont, direc, subdirec, save_individual_contributions,
        save_html=True, peak_window=0.5*dv_value_freq, sourceSize=sourceSize
    )


    #getting dictionary that stores line assignment information
    molecule_assignment_dict = molecule_summary_pipeline_df(peak_df)
    #print(molecule_assignment_dict)

    # Make column density dataframe
    cdDF = pd.DataFrame()
    cdDF['molecule'] = labels
    cdDF['column_density'] = fitted_columns
    outputSmiles = []
    outputLines = []
    outputBlended = []

    for l in labels:
        if l in cdmsMols:
            idx = cdmsMols.index(l)
            outputSmiles.append(cdmsSmiles[idx])
        elif l in jplMols:
            idx = jplMols.index(l)
            outputSmiles.append(jplSmiles[idx])

        if l in molecule_assignment_dict:
            outputLines.append(molecule_assignment_dict[l][0])
            outputBlended.append(molecule_assignment_dict[l][1])
        else:
            outputLines.append(0)
            outputBlended.append(False)


        #if l in molecule_assignment_dict:
        #    outputLines.append()
    cdDF['smiles'] = outputSmiles
    cdDF['number_assigned_lines'] = outputLines
    cdDF['has_unblended_line'] = outputBlended

    cdDF.to_csv(os.path.join(subdirec, 'column_density_results.csv'), index=False)
    
    # Store internal data for notebook plotting
    internal_data = {
        'directory_path': direc,
        'y_exp': y_exp,
        'filtered_mols': mol_list,
        'fitted_columns': fitted_columns,
        'filtered_labels': labels,
        'll0': ll0,
        'ul0': ul0,
        'dataScrape': dataScrape,
        'actualFrequencies': actualFrequencies,
        'intensities': intensities,
        'cont': cont
    }
    
    unique_all_delMols = list(set(all_delMols))
    print(f'\nTotal molecules removed across all iterations: {len(unique_all_delMols)}')
    
    return all_delMols, cdDF, peak_df, internal_data