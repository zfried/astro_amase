'''
Utilities to plot the results of the fit.
'''




import pandas as pd
import numpy as np
import os
import pickle 
import warnings
from .molsim_classes import Source, Simulation, Continuum, Observatory
from .astro_utils import load_parameters
from .molsim_utils import load_obs, find_limits
from .molsim_utils import find_peaks
from ..constants import ckm
from ..data.create_dataset import apply_vlsr_shift
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def show_fit_in_notebook(results, mols_to_display = ['all']):
    """
    Display the fitted spectrum plot in a Jupyter notebook.
    
    This function recreates the interactive Bokeh plot from the fitting results
    and displays it inline in a Jupyter notebook. It uses the same data that was
    saved to the HTML file but creates a fresh plot object to avoid document conflicts.
    
    Parameters
    ----------
    results : dict
        Results dictionary returned from assign_observations().
        Must contain '_internal_data' with the necessary fitting information.
    
    Returns
    -------
    None
        Displays the plot inline in the notebook.
    
    Examples
    --------
    >>> import astro_amase
    >>> 
    >>> results = astro_amase.assign_observations(
    ...     spectrum_path='spectrum.txt',
    ...     directory_path='./data/',
    ...     temperature=170,
    ...     observation_type='interferometric',
    ...     beam_major_axis=0.4,
    ...     beam_minor_axis=0.4
    ... )
    >>> 
    >>> astro_amase.show_fit_in_notebook(results)
    
    Notes
    -----
    - This function will automatically set up notebook output
    - The plot is identical to the saved HTML file
    """
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure
    from bokeh.models import HoverTool, Button, CustomJS, CheckboxGroup
    from bokeh.layouts import column, row
    import numpy as np
    
    # Set up notebook output
    output_notebook()
    
    # Extract necessary data from results
    internal = results['_internal_data']
    
    # Get the data
    freqs = internal['dataScrape'].spectrum.frequency.astype(np.float32)
    y_exp = internal['y_exp'].astype(np.float32)
    filtered_mols = internal['filtered_mols']
    fitted_columns = internal['fitted_columns']
    filtered_labels = internal['filtered_labels']
    ll0 = internal['ll0']
    ul0 = internal['ul0']
    temp = results['temperature']
    dv_value = results['linewidth']
    vlsr_value = results['vlsr']
    cont = internal['cont']
    
    # Simulate individual molecules
    total_sim = np.zeros_like(y_exp, dtype=np.float32)
    individual_sims = []
    
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]
    
    # Create the plot
    p = figure(
        width=1200,
        height=700,
        title="Observed vs Simulated Spectra",
        x_axis_label="Frequency (MHz)",
        y_axis_label="Intensity",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    renderers = []
    names = []


    #filtered_mols = internal['filtered_mols']
    #fitted_columns = internal['fitted_columns']
    #filtered_labels = internal['filtered_labels']

    if mols_to_display != ['all']:
        for m in mols_to_display:
            if m not in filtered_labels:
                warnings.warn("Molecule " + m + " not in assignments.")

        fitted_columns = [fitted_columns[i] for i in range(len(filtered_labels)) if filtered_labels[i] in mols_to_display]
        filtered_mols = [filtered_mols[i] for i in range(len(filtered_labels)) if filtered_labels[i] in mols_to_display]
        filtered_labels = [filtered_labels[i] for i in range(len(filtered_labels)) if filtered_labels[i] in mols_to_display]


    
    # Experimental spectrum
    r_exp = p.line(freqs, y_exp, line_color='black', line_width=2, visible=True, name="Observations")
    renderers.append(r_exp)
    names.append("Observations")
    
    # Individual molecules
    for i, (mol, col) in enumerate(zip(filtered_mols, fitted_columns)):
        src = Source(
            Tex=temp, column=col, size=results['source_size'],
            dV=dv_value, velocity=vlsr_value, continuum=cont
        )
        sim = Simulation(
            mol=mol, ll=ll0, ul=ul0, source=src,
            line_profile='Gaussian', use_obs=True, observation=internal['dataScrape']
        )
        
        spec = np.array(sim.spectrum.int_profile, dtype=np.float32)
        
        # Handle empty spectra
        if spec.size == 0:
            spec = np.zeros_like(total_sim)
            warnings.warn(f"Empty spectrum generated for {filtered_labels[i] if i < len(filtered_labels) else f'Molecule {i+1}'}, using zeros")
        
        individual_sims.append(spec)
        total_sim += spec
        
        mol_name = filtered_labels[i] if i < len(filtered_labels) else f"Molecule {i+1}"
        visible = i < 20
        r = p.line(
            freqs, spec,
            line_color=base_colors[i % len(base_colors)],
            line_width=1.8,
            line_dash="dashed",
            visible=visible,
            name=mol_name
        )
        renderers.append(r)
        names.append(mol_name)
    
    # Total simulated spectrum
    r_total = p.line(freqs, total_sim, line_color='red', line_width=2, visible=True, name="Total Simulated")
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
    
    # Display in notebook
    show(layout)



def plot_from_saved(spectrum_path, directory_path, column_density_csv, stored_json, mols_to_display = ['all']):
    """
    Generate an interactive Bokeh plot comparing observed and simulated molecular spectra.
    
    This function loads a previously saved spectroscopic analysis, reconstructs the molecular
    simulations, and creates an interactive visualization with toggleable traces for individual
    molecules, total simulated spectrum, and observed data.
    
    Parameters
    ----------
    spectrum_path : str
        Path to the observed spectrum file (text format) containing frequency and intensity data.
    directory_path : str
        Base directory containing the molecular database files (CDMS and JPL catalogs).
        Expected structure:
            - directory/cdms_pkl/ : pickled CDMS molecule objects
            - directory/jpl_pkl/ : pickled JPL molecule objects
            - directory/all_cdms_final_official.csv : CDMS catalog metadata
            - directory/all_jpl_final_official.csv : JPL catalog metadata
    column_density_csv : str
        Path to CSV file containing fitted column densities for assigned molecules.
        Expected columns: 'molecule', 'column_density'
    stored_json : str
        Path to JSON file containing saved simulation parameters from a previous analysis.
        Must include:
            - input_parameters: source size, continuum temperature, etc.
            - determined_parameters: temperature, linewidth, VLSR
            - observation_type_description: 'single_dish' or other
            - beam parameters: beam_major_axis_or_dish_diameter, beam_minor_axis (if applicable)
    mols_to_display : list of str, optional
        List of molecule names to display in the plot. Default is ['all'] to show all 
        assigned molecules. If specific molecules are requested but not found in assignments,
        a warning is issued.
    
    Returns
    -------
    None
        Displays an interactive Bokeh plot in a Jupyter notebook environment.
    
    Notes
    -----
    The function creates an interactive plot with the following features:
    - Black line: observed spectrum
    - Red line: total simulated spectrum (sum of all molecules)
    - Dashed colored lines: individual molecular contributions
    - Checkbox legend for toggling visibility of each trace
    - "Show All Molecules" and "Hide All Molecules" buttons for bulk control
    - Hover tooltips showing trace name, frequency, and intensity
    - Interactive pan, zoom, and reset tools
    
    The plot is displayed directly in the notebook using Bokeh's output_notebook() mode.
    Only the first 20 molecular traces are visible by default to avoid visual clutter.
    
    Dependencies
    ------------
    Requires the following modules:
    - bokeh.io, bokeh.plotting, bokeh.models, bokeh.layouts
    - numpy
    - pandas
    - pickle
    - Custom modules: load_parameters, load_obs, find_limits, Continuum, Observatory,
                      Source, Simulation
    
    Examples
    --------
    >>> plot_from_saved(
    ...     spectrum_path='data/spectrum.txt',
    ...     directory='databases/',
    ...     column_density_csv='results/column_densities.csv',
    ...     stored_json='results/parameters.json'
    ... )
    
    >>> # Display only specific molecules
    >>> plot_from_saved(
    ...     spectrum_path='data/spectrum.txt',
    ...     directory='databases/',
    ...     column_density_csv='results/column_densities.csv',
    ...     stored_json='results/parameters.json',
    ...     mols_to_display=['CH3OH', 'H2CO', 'CH3CN']
    ... )
    
    Warnings
    --------
    - Issues a warning if a requested molecule in mols_to_display is not in the assignments
    - Issues a warning if an empty spectrum is generated for any molecule (uses zeros as fallback)
    - Prints "ignoring" message for molecules not found in either CDMS or JPL catalogs
    """

    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure
    from bokeh.models import HoverTool, Button, CustomJS, CheckboxGroup
    from bokeh.layouts import column, row
    import numpy as np
    
    # Set up notebook output
    output_notebook()
    #loading required files 
    cdmsDirec = os.path.join(directory_path, 'cdms_pkl/')
    cdmsFullDF = pd.read_csv(os.path.join(directory_path, 'all_cdms_final_official.csv'))
    df_cdms = cdmsFullDF
    cdms_mols = list(df_cdms['mol'])
    cdms_tags = list(df_cdms['tag'])
    jplDirec = os.path.join(directory_path, 'jpl_pkl/')
    jplFullDF = pd.read_csv(os.path.join(directory_path, 'all_jpl_final_official.csv'))
    df_jpl = jplFullDF
    jpl_mols = list(df_jpl['name'])
    jpl_tags = list(df_jpl['tag'])


    loaded_params = load_parameters(stored_json) #loading saved parameters
    input_params = loaded_params['input_parameters']
    determined_params = loaded_params['determined_parameters']
    df = pd.read_csv(column_density_csv)
    assigned_mols = list(df['molecule']) #retrieving assigned molecules
    assigned_columns = list(df['column_density']) #retrieving fitted column densities

    data = load_obs(spectrum_path, type = 'txt') #load spectrum
    ll0, ul0 = find_limits(data.spectrum.frequency) #determine upper and lower limits of the spectrum
    freqs = data.spectrum.frequency #frequency array of spectrum
    y_exp = data.spectrum.Tb #intensity array of spectrum

    cont = Continuum(type='thermal', params=input_params['continuum_temperature'])

    #creating correct observatory object
    if input_params['observation_type_description'] == 'single_dish':
        observatory1 = Observatory(sd=True, dish=input_params['beam_major_axis_or_dish_diameter'])
    else:
        observatory1 = Observatory(sd=False, array=True,synth_beam = [input_params['beam_major_axis_or_dish_diameter'],input_params['beam_minor_axis']])

    data.observatory = observatory1

    temp = determined_params['temperature']
    dv_value = determined_params['linewidth_kms']
    vlsr_value = determined_params['vlsr']

    fitted_columns = []
    mol_list = []
    labels = []
    #retrieving required molecule objects for simulation
    c = 0
    for a_mol in assigned_mols: 
        if a_mol in cdms_mols:
            idx = cdms_mols.index(a_mol)
            tag = cdms_tags[idx]
            tagString = f"{tag:06d}"
            molDirec = cdmsDirec + tagString + '.pkl'
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            mol_list.append(mol)
            labels.append(a_mol)
            fitted_columns.append(assigned_columns[c])
        elif a_mol in jpl_mols:
            idx = jpl_mols.index(a_mol)
            tag = jpl_tags[idx]
            tagString = str(tag)
            molDirec = jplDirec + tagString + '.pkl'
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            mol_list.append(mol)
            labels.append(a_mol)
            fitted_columns.append(assigned_columns[c])
        else:
            print('ignoring')
            print(a_mol)
        c += 1

    

    # Simulate individual molecules
    total_sim = np.zeros_like(y_exp, dtype=np.float32)
    individual_sims = []
    
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]
    
    # Create the plot
    p = figure(
        width=1200,
        height=700,
        title="Observed vs Simulated Spectra",
        x_axis_label="Frequency (MHz)",
        y_axis_label="Intensity",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    renderers = []
    names = []


    filtered_mols = mol_list
    fitted_columns = fitted_columns
    filtered_labels = labels

    if mols_to_display != ['all']:
        for m in mols_to_display:
            if m not in filtered_labels:
                warnings.warn("Molecule " + m + " not in assignments.")

        fitted_columns = [fitted_columns[i] for i in range(len(filtered_labels)) if filtered_labels[i] in mols_to_display]
        filtered_mols = [filtered_mols[i] for i in range(len(filtered_labels)) if filtered_labels[i] in mols_to_display]
        filtered_labels = [filtered_labels[i] for i in range(len(filtered_labels)) if filtered_labels[i] in mols_to_display]


     # Experimental spectrum
    r_exp = p.line(freqs, y_exp, line_color='black', line_width=2, visible=True, name="Observations")
    renderers.append(r_exp)
    names.append("Observations")
    
    # Individual molecules
    for i, (mol, col) in enumerate(zip(filtered_mols, fitted_columns)):
        src = Source(
            Tex=temp, column=col, size=input_params['source_size'],
            dV=dv_value, velocity=vlsr_value, continuum=cont
        )
        sim = Simulation(
            mol=mol, ll=ll0, ul=ul0, source=src,
            line_profile='Gaussian', use_obs=True, observation=data
        )
        
        spec = np.array(sim.spectrum.int_profile, dtype=np.float32)
        
        # Handle empty spectra
        if spec.size == 0:
            spec = np.zeros_like(total_sim)
            warnings.warn(f"Empty spectrum generated for {filtered_labels[i] if i < len(filtered_labels) else f'Molecule {i+1}'}, using zeros")
        
        individual_sims.append(spec)
        total_sim += spec
        
        mol_name = filtered_labels[i] if i < len(filtered_labels) else f"Molecule {i+1}"
        visible = i < 20
        r = p.line(
            freqs, spec,
            line_color=base_colors[i % len(base_colors)],
            line_width=1.8,
            line_dash="dashed",
            visible=visible,
            name=mol_name
        )
        renderers.append(r)
        names.append(mol_name)
    
    # Total simulated spectrum
    r_total = p.line(freqs, total_sim, line_color='red', line_width=2, visible=True, name="Total Simulated")
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
    
    # Display in notebook
    show(layout)

def get_individual_plots(spectrum_path, directory_path, column_density_csv, stored_json, mols_to_display = ['all'], minimum_intensity = 'default'):
    """
    Generate individual peak plots for assigned molecules in a molecular spectrum.
    
    This function creates PDF files containing subplots of spectral peaks for each assigned
    molecule. Each subplot shows a zoomed-in view around a detected peak, displaying both
    the observed spectrum and the simulated spectrum for comparison, along with quantum
    number assignments.
    
    Parameters
    ----------
    spectrum_path : str
        Path to the observed spectrum file (txt format).
    directory_path : str
        Path to the directory containing molecular catalogs (CDMS and JPL pickle files)
        and reference CSV files.
    column_density_csv : str
        Path to CSV file containing assigned molecules and their fitted column densities.
        Expected columns: 'molecule', 'column_density'.
    stored_json : str
        Path to JSON file containing stored simulation parameters including input parameters
        (continuum temperature, observation type, beam size, source size) and determined
        parameters (temperature, linewidth, vlsr, resolution, rms_noise).
    mols_to_display : list of str, optional
        List of molecule names to generate plots for. Default is ['all'] which processes
        all assigned molecules.
    minimum_intensity : float or 'default', optional
        Minimum intensity threshold for including peaks in the plots. If 'default', uses
        the RMS noise level from determined parameters. Otherwise, uses the specified value.
    
    Returns
    -------
    None
        Generates PDF files named '{molecule_name}_peaks.pdf' in the directory_path,
        each containing a grid of subplots showing individual spectral peaks with their
        quantum number assignments.
    
    Notes
    -----
    - Each PDF contains a 3-column grid of subplots, with each subplot showing a ±4*linewidth
      frequency range around a detected peak.
    - Peaks are sorted by intensity in descending order.
    - Quantum numbers are retrieved from the molecular catalog and displayed on each subplot.
    - Observed spectrum is plotted in black, simulated spectrum in red.
    - Molecules not found in CDMS or JPL catalogs are ignored with a warning message.
    - If a molecule has no peaks above the minimum intensity, a warning is issued.
    """

    #loading required files 
    cdmsDirec = os.path.join(directory_path, 'cdms_pkl/')
    cdmsFullDF = pd.read_csv(os.path.join(directory_path, 'all_cdms_final_official.csv'))
    df_cdms = cdmsFullDF
    cdms_mols = list(df_cdms['mol'])
    cdms_tags = list(df_cdms['tag'])
    jplDirec = os.path.join(directory_path, 'jpl_pkl/')
    jplFullDF = pd.read_csv(os.path.join(directory_path, 'all_jpl_final_official.csv'))
    df_jpl = jplFullDF
    jpl_mols = list(df_jpl['name'])
    jpl_tags = list(df_jpl['tag'])


    loaded_params = load_parameters(stored_json) #loading saved parameters
    input_params = loaded_params['input_parameters']
    determined_params = loaded_params['determined_parameters']
    df = pd.read_csv(column_density_csv)
    assigned_mols = list(df['molecule']) #retrieving assigned molecules
    assigned_columns = list(df['column_density']) #retrieving fitted column densities

    data = load_obs(spectrum_path, type = 'txt') #load spectrum
    ll0, ul0 = find_limits(data.spectrum.frequency) #determine upper and lower limits of the spectrum
    freq_arr = data.spectrum.frequency #frequency array of spectrum
    y_exp = data.spectrum.Tb #intensity array of spectrum

    cont = Continuum(type='thermal', params=input_params['continuum_temperature'])

    #creating correct observatory object
    if input_params['observation_type_description'] == 'single_dish':
        observatory1 = Observatory(sd=True, dish=input_params['beam_major_axis_or_dish_diameter'])
    else:
        observatory1 = Observatory(sd=False, array=True,synth_beam = [input_params['beam_major_axis_or_dish_diameter'],input_params['beam_minor_axis']])

    data.observatory = observatory1

    temp = determined_params['temperature']
    dv_value = determined_params['linewidth_kms']
    vlsr_value = determined_params['vlsr']
    resolution = determined_params['resolution']
    dv_value_freq = determined_params['linewidth_mhz']
    rms_noise = determined_params['rms_noise']

    #print('temp',temp)
    #print('vlsr_value', vlsr_value )
    
    if minimum_intensity == 'default': #if default, get all lines down to the rms noise
        minimum_intensity = rms_noise
    else:
        minimum_intensity = minimum_intensity #otherwise, use the inputted value

    fitted_columns = []
    mol_list = []
    labels = []
    #retrieving required molecule objects for simulation
    c = 0
    for a_mol in assigned_mols: 
        if a_mol in cdms_mols:
            idx = cdms_mols.index(a_mol)
            tag = cdms_tags[idx]
            tagString = f"{tag:06d}"
            molDirec = cdmsDirec + tagString + '.pkl'
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            mol_list.append(mol)
            labels.append(a_mol)
            fitted_columns.append(assigned_columns[c])
        elif a_mol in jpl_mols:
            idx = jpl_mols.index(a_mol)
            tag = jpl_tags[idx]
            tagString = str(tag)
            molDirec = jplDirec + tagString + '.pkl'
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            mol_list.append(mol)
            labels.append(a_mol)
            fitted_columns.append(assigned_columns[c])
        else:
            print('ignoring')
            print(a_mol)
        c += 1

    
    '''
    # Simulate individual molecules
    total_sim = np.zeros_like(y_exp, dtype=np.float32)
    individual_sims = []
    
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]
    
    # Create the plot
    p = figure(
        width=1200,
        height=700,
        title="Observed vs Simulated Spectra",
        x_axis_label="Frequency (MHz)",
        y_axis_label="Intensity",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    renderers = []
    names = []
    '''

    filtered_mols = mol_list
    fitted_columns = fitted_columns
    filtered_labels = labels

    if mols_to_display != ['all']:
        for m in mols_to_display:
            if m not in filtered_labels:
                warnings.warn("Molecule " + m + " not in assignments.")

        fitted_columns = [fitted_columns[i] for i in range(len(filtered_labels)) if filtered_labels[i] in mols_to_display]
        filtered_mols = [filtered_mols[i] for i in range(len(filtered_labels)) if filtered_labels[i] in mols_to_display]
        filtered_labels = [filtered_labels[i] for i in range(len(filtered_labels)) if filtered_labels[i] in mols_to_display]

    #print(f"Total filtered molecules to process: {len(filtered_mols)}")
    #print(f"Filtered labels: {filtered_labels}")
    #print(f"Fitted columns: {fitted_columns}")
    # Individual molecules
    for i, (mol, col, label) in enumerate(zip(filtered_mols, fitted_columns, filtered_labels)):
        #print(label)
        #print(f"Scientific notation: {col:e}")
        cat = mol.catalog
        cat_freqs = cat.frequency
        logints = cat.logint
        shifted_freqs = apply_vlsr_shift(cat_freqs, vlsr_value)
        qn_ups = cat.qnup_str
        qn_lows = cat.qnlow_str

        #print("=" * 60)
        #print("Values in Scientific Notation")
        #print("=" * 60)
        #print(f"Tex (Temperature):     {temp:e}")
        #print(f"column:                {col:e}")
        #print(f"size:                  {input_params['source_size']:e}")
        #print(f"dV:                    {dv_value:e}")
        #print(f"velocity:              {vlsr_value:e}")
        #print(f"continuum:             {cont:e}")
        #print("=" * 60)

        src = Source(
            Tex=temp, column=col, size=input_params['source_size'],
            dV=dv_value, velocity=vlsr_value, continuum=cont
        )
        sim = Simulation(
            mol=mol, ll=ll0, ul=ul0, source=src,
            line_profile='Gaussian', use_obs=True, observation=data
        )
       
        #finding peaks in simulation
        peak_indicesIndiv = find_peaks(sim.spectrum.freq_profile, sim.spectrum.int_profile, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_value_freq), is_sim=True)
        
        if len(peak_indicesIndiv) > 0:
            peak_freqs2 = sim.spectrum.freq_profile[peak_indicesIndiv]
            peak_ints2 = abs(sim.spectrum.int_profile[peak_indicesIndiv])
            mask = peak_ints2 > minimum_intensity #keeping only the transitions above a minimum intensity
            # Filter both arrays
            peak_ints2 = peak_ints2[mask]
            peak_freqs2 = peak_freqs2[mask]

            if len(peak_freqs2) == 0:
                warnings.warn(f"No peaks above threshold for {label}. Skipping PDF.")
                continue


            # Then sort by intensity
            sort_indices = np.argsort(-peak_ints2)
            peak_ints2 = peak_ints2[sort_indices]
            peak_freqs2 = peak_freqs2[sort_indices]  



            int_sim = np.array(sim.spectrum.int_profile, dtype=np.float32)
            freq_sim = np.array(sim.spectrum.freq_profile, dtype=np.float32)

            # Create PDF to save all peaks
            # Create PDF to save all peaks
            save_label = label.replace('/','_')
            pdf_filename = os.path.join(directory_path,f"{save_label}_peaks.pdf")  # m is your molecule name
            with PdfPages(pdf_filename) as pdf:
                n_peaks = len(peak_freqs2)
                n_cols = 3
                n_rows = int(np.ceil(n_peaks / n_cols))
                
                # Create figure with subplots
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
                
                # Flatten axes array for easier iteration
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                axes_flat = axes.flatten()
                
                #looping through peak freqs
                for idx, (peak_freq, peak_int) in enumerate(zip(peak_freqs2, peak_ints2)):
                    ax = axes_flat[idx]
                    #selecting nearby line with largest logint in catalog (in case nearby transitions)
                    indices = np.where(np.abs(shifted_freqs - peak_freq) <= 0.25*dv_value_freq)[0]  
                    # Check if any matching lines were found
                    if len(indices) == 0:
                        qn_full_str = ''  # Empty string if no catalog match
                    else:
                        cat_i = indices[logints[indices].argmax()]
                        #print(indices)
                        #print(logints[cat_i])

                        #formatting quantum numbers string 
                        qn_us = []
                        for qn_u in [cat.qn1up[cat_i], cat.qn2up[cat_i], cat.qn3up[cat_i], cat.qn4up[cat_i], cat.qn5up[cat_i], cat.qn6up[cat_i], cat.qn7up[cat_i], cat.qn8up[cat_i]]:
                            if qn_u is not None:
                                qn_us.append(qn_u)
                        qn_ls = []
                        for qn_l in [cat.qn1low[cat_i], cat.qn2low[cat_i], cat.qn3low[cat_i], cat.qn4low[cat_i], cat.qn5low[cat_i], cat.qn6low[cat_i], cat.qn7low[cat_i], cat.qn8low[cat_i]]:
                            if qn_l is not None:
                                qn_ls.append(qn_l)

                        qn_up_str = ''
                        for q in qn_us:
                            qn_up_str = qn_up_str + str(q) + ','
                        qn_up_str = qn_up_str[:-1]
                        qn_low_str = ''
                        for q in qn_ls:
                            qn_low_str = qn_low_str + str(q) + ','
                        qn_low_str = qn_low_str[:-1]

                        qn_full_str = qn_up_str + ' - ' + qn_low_str
        
                    # Define frequency range: ±4*dv_value_freq from peak center
                    freq_min = peak_freq - 8 * dv_value_freq
                    freq_max = peak_freq + 8 * dv_value_freq
                    
                    # Create mask for simulated data in this frequency range
                    freq_mask_sim = (freq_sim >= freq_min) & (freq_sim <= freq_max)
                    freq_range_sim = freq_sim[freq_mask_sim]
                    int_range_sim = int_sim[freq_mask_sim]
                    
                    # Create mask for experimental data in this frequency range
                    freq_mask_exp = (freq_arr >= freq_min) & (freq_arr <= freq_max)
                    freq_range_exp = freq_arr[freq_mask_exp]
                    int_range_exp = y_exp[freq_mask_exp]
                    
                    # Plot
                    ax.plot(freq_range_exp, int_range_exp, 'k-', linewidth=1)  # Experimental in black
                    ax.plot(freq_range_sim, int_range_sim, 'r-', linewidth=1)  # Simulated in red

                    # Set limits
                    ax.set_xlim(freq_min, freq_max)
                    ax.set_ylim(-0.15*max(int_range_exp), 2*max(int_range_exp))

                    # Add x-axis label
                    ax.set_xlabel('Sky Frequency (MHz)')

                    freq_range = freq_max - freq_min
                    xticks = [freq_min + 0.2*freq_range, freq_min + 0.5*freq_range, freq_min + 0.8*freq_range]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([f'{x:.2f}' for x in xticks])

                    label_formatted = label.replace('le', '≤')
                    # Add molecule name in top right
                    ax.text(0.98, 0.98, label_formatted, transform=ax.transAxes, 
                            verticalalignment='top', horizontalalignment='right',
                            fontsize=15)
                
                    #add quantum number string below name
                    ax.text(0.98, 0.92, qn_full_str, transform=ax.transAxes, 
                            verticalalignment='top', horizontalalignment='right',
                            fontsize=15)
                
                # Hide unused subplots
                for idx in range(n_peaks, len(axes_flat)):
                    axes_flat[idx].axis('off')
                
                # Adjust layout and save
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        else:
            warnings.warn("Molecule " + label + " has no peaks.")


def create_interactive_vlsr_plot(spectrum_path, directory_path, molecule_name, 
                                 excitation_temperature, linewidth, 
                                 vlsr_min=-50, vlsr_max=50, vlsr_step=0.05,
                                 column_density=1.E15, source_size=1.E20, 
                                 observation_type='single_dish', dish_diameter=100, 
                                 beam_major_axis=0.5, beam_minor_axis=0.5, 
                                 continuum_temperature=2.7):
    
    """
    Create an interactive Bokeh plot with slider control for adjusting VLSR in real-time.
    
    This function computes a molecular spectrum once at VLSR=0, then allows the user to 
    interactively shift it to different velocities by dragging a slider. The frequency 
    shift is applied using the Doppler formula without recomputing the spectrum, making 
    updates instantaneous. The observed spectrum is plotted in black and the simulated 
    spectrum in red.
    
    Parameters
    ----------
    spectrum_path : str
        Path to the observed spectrum file (expected format: txt with frequency and 
        intensity columns).
        
    directory_path : str
        Path to the directory containing molecular catalog files (CDMS and JPL pkl files 
        and CSV metadata).
        
    molecule_name : str
        Name of the molecule to simulate (must match entries in CDMS or JPL catalogs).
        Example: 'CH3OH, vt = 0 - 2' or 'HC3N, (0,0,0,0)'
        
    excitation_temperature : float
        Excitation temperature of the source in Kelvin. This sets the population 
        distribution of molecular energy levels.
        
    linewidth : float
        Line width (FWHM) of the molecular transitions in km/s. This represents the 
        velocity dispersion in the source.
        
    vlsr_min : float, optional
        Minimum value for the VLSR slider in km/s. Default is -50 km/s.
        
    vlsr_max : float, optional
        Maximum value for the VLSR slider in km/s. Default is 50 km/s.
        
    vlsr_step : float, optional
        Step size (resolution) of the VLSR slider in km/s. Smaller values provide 
        finer control. Default is 0.05 km/s. 
        
    column_density : float, optional
        Column density of the molecule in cm^-2. Default is 1.0e15 cm^-2.
        
    source_size : float, optional
        Source size in arcsec^2. Default is 1.0e20 (effectively infinite, filling the 
        beam). Use smaller values for compact sources.
        
    observation_type : str, optional
        Type of observation: 'single_dish' or 'interferometric'. Default is 'single_dish'.
        
    dish_diameter : float, optional
        Diameter of the single dish telescope in meters. Only used when 
        observation_type='single_dish'. Default is 100 meters.
        
    beam_major_axis : float, optional
        Major axis of the synthesized beam in arcseconds. Only used when 
        observation_type='interferometric'. Default is 0.5 arcsec.
        
    beam_minor_axis : float, optional
        Minor axis of the synthesized beam in arcseconds. Only used when 
        observation_type='interferometric'. Default is 0.5 arcsec.
        
    continuum_temperature : float, optional
        Background continuum temperature in Kelvin. Default is 2.7 K (CMB temperature).
    
    Returns
    -------
    None
        Displays an interactive Bokeh plot in the notebook with slider control.
    
    Notes
    -----
    - The spectrum is computed only once at VLSR=0 km/s for efficiency
    - VLSR shifts are applied using: freq_shifted = freq_ref - vlsr * freq_ref / c
    - Updates are instantaneous since no spectral recomputation is needed
    - Drag the slider to adjust VLSR and see the spectrum shift in real-time
    - The column density affects spectral intensity but cannot be adjusted interactively
      in this function (use create_interactive_vlsr_plot_2 for that capability)
    
    Interactive Controls
    --------------------
    - Slider: Drag left/right to adjust VLSR
    - Status Display: Shows current VLSR value
    - Bokeh Tools: Pan, zoom, reset, and save built into the plot
    - Legend: Click items to hide/show spectra
    
    The function requires molecular catalog files in the specified directory:
    - cdms_pkl/ directory with CDMS catalog pickle files
    - jpl_pkl/ directory with JPL catalog pickle files  
    - all_cdms_final_official.csv metadata file
    - all_jpl_final_official.csv metadata file
    """
    
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, Slider, Div, CustomJS
    from bokeh.layouts import column, row
    from bokeh.io import output_notebook
    
    # Enable Bokeh output in notthe ebook
    output_notebook()
    
    # Load all the static data
    cdmsDirec = os.path.join(directory_path, 'cdms_pkl/')
    cdmsFullDF = pd.read_csv(os.path.join(directory_path, 'all_cdms_final_official.csv'))
    df_cdms = cdmsFullDF
    cdms_mols = list(df_cdms['mol'])
    cdms_tags = list(df_cdms['tag'])
    jplDirec = os.path.join(directory_path, 'jpl_pkl/')
    jplFullDF = pd.read_csv(os.path.join(directory_path, 'all_jpl_final_official.csv'))
    df_jpl = jplFullDF
    jpl_mols = list(df_jpl['name'])
    jpl_tags = list(df_jpl['tag'])
    
    # Load observed spectrum
    data = load_obs(spectrum_path, type='txt')
    ll0, ul0 = find_limits(data.spectrum.frequency)
    freq_arr = data.spectrum.frequency
    y_exp = data.spectrum.Tb
    
    # Set up observatory and continuum
    cont = Continuum(type='thermal', params=continuum_temperature)
    
    if observation_type == 'single_dish':
        observatory1 = Observatory(sd=True, dish=dish_diameter)
    else:
        observatory1 = Observatory(sd=False, array=True, 
                                   synth_beam=[beam_major_axis, beam_minor_axis])
    
    data.observatory = observatory1
    
    # Load molecule
    if molecule_name in cdms_mols:
        idx = cdms_mols.index(molecule_name)
        tag = cdms_tags[idx]
        tagString = f"{tag:06d}"
        molDirec = cdmsDirec + tagString + '.pkl'
        with open(molDirec, 'rb') as md:
            mol = pickle.load(md)
    elif molecule_name in jpl_mols:
        idx = jpl_mols.index(molecule_name)
        tag = jpl_tags[idx]
        tagString = str(tag)
        molDirec = jplDirec + tagString + '.pkl'
        with open(molDirec, 'rb') as md:
            mol = pickle.load(md)
    else:
        raise ValueError('Could not find molecule. Please check inputted molecule_name')
    
    # Compute spectrum ONCE at vlsr=0
    src_ref = Source(
        Tex=excitation_temperature, column=column_density, size=source_size,
        dV=linewidth, velocity=0.0, continuum=cont
    )
    sim_ref = Simulation(
        mol=mol, ll=ll0, ul=ul0, source=src_ref,
        line_profile='Gaussian', use_obs=True, observation=data
    )
    spec_ref_y = np.array(sim_ref.spectrum.int_profile, dtype=np.float32)
    spec_ref_x = freq_arr.copy()  # Reference frequency at vlsr=0
    
    #print("Reference spectrum computed!")
    
    # Create ColumnDataSources
    source_obs = ColumnDataSource(data=dict(x=freq_arr, y=y_exp))
    source_sim = ColumnDataSource(data=dict(x=spec_ref_x, y=spec_ref_y))
    
    # Store reference spectrum (unshifted)
    source_ref = ColumnDataSource(data=dict(
        freq_ref=spec_ref_x.tolist(),
        intensity_ref=spec_ref_y.tolist()
    ))
    
    # Create the figure
    p = figure(width=900, height=500, 
               title=f'{molecule_name} - Interactive VLSR Adjustment',
               x_axis_label='Frequency (MHz)',
               y_axis_label='Intensity (K)')
    
    # Plot observed spectrum (black)
    p.line('x', 'y', source=source_obs, line_width=2, color='black', 
           alpha=0.7, legend_label='Observed')
    
    # Plot simulated spectrum (red)
    p.line('x', 'y', source=source_sim, line_width=2, color='red', 
           legend_label='Simulated (vlsr=0.00 km/s)')
    
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    
    # Create slider
    slider = Slider(start=vlsr_min, end=vlsr_max, value=0.0, step=vlsr_step, 
                    title="VLSR (km/s)", width=400)
    
    # Create status div
    status_div = Div(text="<p>Drag slider to adjust VLSR. Current: 0.00 km/s</p>", width=400)
    
    # JavaScript callback to update plot
    callback = CustomJS(args=dict(
        source_ref=source_ref,
        source_sim=source_sim,
        slider=slider,
        legend=p.legend[0],
        status_div=status_div
    ), code="""
        // Get vlsr from slider
        const vlsr = slider.value;
        
        // Speed of light in km/s
        const ckm = 299792.458;
        
        // Get reference spectrum
        const freq_ref = source_ref.data['freq_ref'];
        const intensity_ref = source_ref.data['intensity_ref'];
        
        // Apply Doppler shift: freq_shifted = freq_ref - vlsr * freq_ref / ckm
        const n = freq_ref.length;
        const freq_shifted = new Array(n);
        
        for (let i = 0; i < n; i++) {
            freq_shifted[i] = freq_ref[i] - vlsr * freq_ref[i] / ckm;
        }
        
        // Update simulated spectrum
        source_sim.data['x'] = freq_shifted;
        source_sim.data['y'] = intensity_ref;
        source_sim.change.emit();
        
        // Update legend
        legend.items[1].label.value = 'Simulated (vlsr=' + vlsr.toFixed(2) + ' km/s)';
        
        // Update status
        status_div.text = '<p style="color:green;">Drag slider to adjust VLSR. Current: ' + vlsr.toFixed(2) + ' km/s</p>';
    """)
    
    # Connect slider to callback
    slider.js_on_change('value', callback)
    
    # Create layout
    controls = column(slider, status_div)
    layout = column(controls, p)
    
    # Show the plot
    show(layout)
    print("\nDrag the slider to adjust VLSR and see the spectrum shift in real-time!")

