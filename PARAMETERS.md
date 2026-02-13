# Parameter Guide for Astro AMASE

This guide provides detailed information about all parameters available in `astro_amase.assign_observations()`, including recommendations for different use cases.

## Table of Contents
- [Required Parameters](#required-parameters)
- [Core Parameters](#core-parameters)
- [Observation Configuration](#observation-configuration)
- [Advanced Parameters](#advanced-parameters)
- [Output Configuration](#output-configuration)
- [Tips and Best Practices](#tips-and-best-practices)

---


## Required Parameters

### Input Files

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `spectrum_path` | `str` | Path to spectrum file (.txt with frequency and intensity columns and no headers) | `'data/spectrum.txt'` |
| `directory_path` | `str` | Path to directory containing files downloaded from [Dropbox folder](https://www.dropbox.com/scl/fo/s1dhye6mrdistrm0vbim7/ALRlugfuxnsHZU4AisPWjig?rlkey=7fk1obwvkeihlo8jt84g2wqfr&st=hqrts8cd&dl=0) | `'./database/'` |
| `temperature` | `float` | Excitation temperature in Kelvin | `150.0` |



---

## Core Parameters

### Temperature and VLSR

| Parameter | Type | Default | Description | Recommendation |
|-----------|------|---------|-------------|----------------|
| `temperature` | `float` | **Required** | Excitation temperature in Kelvin | If unknown, provide best estimate (10K for cold clouds, 150K for hot cores) |
| `temperature_is_exact` | `bool` | `False` | If `False`, optimizes temperature ±100K around provided value | Set `False` if temperature uncertain; automatically `True` if `vlsr` provided |
| `vlsr` | `float` | `None` | Source VLSR in km/s. If not provided, determined automatically | Provide if known for faster results |
| `linewidth` | `float` | `None` | Line width (FWHM) in km/s. If not provided, determined automatically | Usually auto-determined|
| `vlsr_range` | `list[float, float]` | `[-250, 250]` | Min/max VLSR bounds for fitting [min, max] | Adjust if source has extreme velocities or if range is known |
| `vlsr_mols` | `list[str]` | `'all'` | Molecules to use for VLSR determination | Inputting `'all'` uses all the molecules in the `notebooks/vlsr_molecules.csv` file. Other molecules outside of this file can be used. The inputted name must match the first column of the `all_cdms_final_official.csv` or `all_jpl_final_offiical.csv` files downloaded from [Dropbox](https://www.dropbox.com/scl/fo/s1dhye6mrdistrm0vbim7/ALRlugfuxnsHZU4AisPWjig?rlkey=7fk1obwvkeihlo8jt84g2wqfr&st=hqrts8cd&dl=0).|

**Temperature/VLSR Logic:**
- **If `vlsr` is provided:** `temperature_is_exact` is forced to `True` (you must provide exact temperature)
- **If `vlsr` is NOT provided:**
  - `temperature_is_exact=False`: Code optimizes temperature within ±100K
  - `temperature_is_exact=True`: Code uses exact temperature

**Recommendations:**
- **Cold clouds (TMC-1):** `temperature=10.0`, `temperature_is_exact=False`
- **Hot cores (Sgr B2, Orion KL):** `temperature=150.0`, `temperature_is_exact=False`
- **If you know VLSR:** Always provide it: `vlsr=5.5`, `temperature=10.0` (exact)

---

### Line Detection

| Parameter | Type | Default | Description | Recommendation |
|-----------|------|---------|-------------|----------------|
| `sigma_threshold` | `float` | `5.0` | Sigma threshold for line detection |Don't go too far below 5σ. Molecules whose maximum simulated intensity is below 2.5σ are removed from the assignment. |
| `rms_noise` | `None`, `float`, or `dict` | `None` | RMS noise level(s). Auto-calculated if not provided | `None` will automatically calculate the noise level in each region of the spectrum. If a `float` is entered, this value will be used for the entire frequency range. The noise level in each frequency range can also be inputted manually using a `dict`. Inputting a dictionary makes the process much slower.|

**RMS Noise Options:**
1. **Auto-calculate (default):** `rms_noise=None`
2. **Single value:** `rms_noise=0.015` (Kelvin)
3. **Frequency-dependent dict:** 
   ```python
   rms_noise = {
       (9500.0, 12000.0): 0.0062,
       (12000.0, 27000.0): 0.015,
       (27000.0, 31500.0): 0.034
   }
   ```

**Notes:**
- When using a frequency-dependent dictionary, the dictionary keys are frequency ranges in MHz `(freq_min, freq_max)`, and values are RMS noise levels in Kelvin. For frequency gaps between defined ranges, the higher RMS of the adjacent regions is used. If the dictionary doesn't cover the full spectrum, the first and last RMS values are extended to the spectrum edges.

**Recommendations:**
- Using auto-calculation is generally the best approach but can determine too high of a noise level in line-dense surveys
- Use frequency-dependent RMS for wide-band surveys with varying sensitivity

---

## Observation Configuration

### Observation Type

| Parameter | Type | Default | Description | Valid Values |
|-----------|------|---------|-------------|--------------|
| `observation_type` | `str` | `'single_dish'` | Type of telescope observation | `'single_dish'`, `'interferometric'` |

### Beam Parameters

**For Single Dish Observations:**

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `dish_diameter` | `float` | `100.0` | Dish diameter in meters | `100.0` (GBT), `30.0` (IRAM 30m) |

**For Interferometric Observations:**

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `beam_major_axis` | `float` | `0.5` | Beam major axis in arcseconds | `0.5` |
| `beam_minor_axis` | `float` | `0.5` | Beam minor axis in arcseconds | `0.5` |

### Source Properties

| Parameter | Type | Default | Description | Recommendation/Notes |
|-----------|------|---------|-------------|----------------|
| `source_size` | `float` | `1E20` | Source diameter in arcseconds | Use `1E20` if source fills beam; otherwise provide measured size |
| `continuum_temperature` | `float` | `2.7` | Continuum background temperature in Kelvin | If excitation temperature is below the continuum temperature, simulated lines will appear in absorption |

**Examples:**
```python
# Single dish (GBT)
observation_type='single_dish',
dish_diameter=100.0

# Interferometer (ALMA)
observation_type='interferometric',
beam_major_axis=0.5,
beam_minor_axis=0.4
```

---

## Advanced Parameters

### Chemical Constraints

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `valid_atoms` | `list[str]` | `['C', 'O', 'H', 'N', 'S']` | Atomic symbols that could be present | `['C', 'O', 'H', 'N', 'S', 'Si']` |
| `force_ignore_molecules` | `list[str]` | `[]` | Molecules to forcibly exclude | `['CH334SH, vt le 2', 'l-13CC3H2']` |
| `force_include_molecules` | `list[str]` | `[]` | Molecules to forcibly include | `['HC3N, (0,0,0,0)']` |


**Notes:**
- **force_ignore_molecules:** Can add molecules to this list if there are false-positive assignments
- **force_include_molecules:** Can add molecules to test if they are present in the data
- **Important:** Molecule names in list must match the first columns of the `all_cdms_final_official.csv` or `all_jpl_final_offiical.csv` files downloaded from [Dropbox](https://www.dropbox.com/scl/fo/s1dhye6mrdistrm0vbim7/ALRlugfuxnsHZU4AisPWjig?rlkey=7fk1obwvkeihlo8jt84g2wqfr&st=hqrts8cd&dl=0). For example, `CH3OH, vt = 0 - 2` and `HC3N, (0,0,0,0)`. These are the names stored in the `column_density_results.csv` output file, so they can be copied from there into the `force_ignore_molecules` list if needed.


---

### Fitting Parameters

| Parameter | Type | Default | Description | Recommendation |
|-----------|------|---------|-------------|----------------|
| `column_density_range` | `list[float, float]` | `[1e10, 1e20]` | Min/max column density bounds [min, max] in cm⁻² | Can adjust based on source but default is generally fine|
| `fitting_iterations` | `int` | `0` | Number of fitting iterations. `0` = iterate to convergence | Use `0` for automatic; `≥2` for fixed iterations. Highly recommended to use `0` |
| `stricter` | `bool` | `False` | Apply additional filtering checks when removing molecules | Set `True` if some false positive assignments |

**Iteration Behavior:**
- `fitting_iterations=0`: Iterates until no molecules are removed (convergence)
- `fitting_iterations=2+`: Performs fixed number of filter-and-refit cycles. Cannot be less than 2
- Each iteration: assigns → filters weak molecules → refits

---


### Structural Relevance

| Parameter | Type | Default | Description | Recommendation |
|-----------|------|---------|-------------|----------------|
| `consider_structure` | `bool` | `True` | Use structural similarity scoring during assignment | Can set to `False` if you suspect molecules in source are not chemically related or if there are some notable outlier molecules (like ringed species or PAHs) |
| `known_molecules` | `list[str]` | `None` | SMILES strings of molecules used to initialize the structural relevance scoring | `['C=O', 'C=O', 'CC(C)=O', 'CC#N']` |


**What it does:**
- Scores molecules based on structural similarity to detected molecules
- Prioritizes chemically related species
- Influenced by `known_molecules` parameter

**`known_molecules` Details:**
- Uses SMILES notation
- Can include duplicates to require multiple copies and further up-score molecules similar to that species
- Example: `['C=O', 'C=O']` if you want to especially prioritize molecules near formaldehyde or if formaldehyde is struggling to be properly assigned due to low structural relevance score

**Struggle Cases:**
- Sometimes it can struggle with assigning larger cyanopolyynes or cummulene molecules. For example, if HC7N, HC9N, C3S, H2C4, H2C6, etc. are expected to be present, I would recommend initializing the `known_molecules` list with these SMILES strings (along with smaller species like HC3N, HC5N and C2S).
- Sometimes it can also struggle with acetone `CC(C)=O`. Can also initialize `known_molecules` list with this SMILES string (along with common molecules like methanol, formaldehyde, ethanol, etc.) if hot core source.

**Recommendations:**
- Cold Cloud: `['C#CC#N', 'C#CC#CC#N', 'C#CC#CC#CC#N', 'C#CC#CC#CC#CC#N', '[C]=C=S', '[C]=C=C=S', '[C]=C=C=C', '[C]=C=C=C=C']`
- Hot Core: `['[C-]#[O+]', 'CO', 'CCO', 'C=O', 'COC=O', 'CC#N', 'CC(C)=O']`
---

## Output Configuration

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `save_name` | `str` | `'default_name'` | Subdirectory name for output files | `'tmc1_analysis'` |
| `overwrite_old_files` | `bool` | `False` | Overwrite existing output files in the directory with the same  `save_name`| `True` to replace old results (and save space on computer) |
| `save_individual_contributions` | `bool` | `False` | Save individual molecule simulated spectra, total summed simulated spectrum, and residual between observational data and summed simulated spectrum | `True` for detailed analysis |

**Output Structure:**
```
directory_path/
└── save_name/
    ├── analysis_parameters.json
    ├── assignment_results.txt
    ├── interactive_spectrum.html
    ├── column_densities.txt
    └── [individual spectra if save_individual_contributions=True]
```

**Recommendations:**
- Use descriptive `save_name`: `'tmc1_10K_analysis'`, `'sgrb2_hot_core'`

---


## Return Values

The function returns a dictionary with:

```python
{
    'assigner': <IterativeSpectrumAssignment object>,
    'statistics': {
        'total_lines': 150,
        'assigned_lines': 120,
        'unidentified_lines': 30,
        'assigned_molecules': 23
    },
    'vlsr': 5.5,              # km/s
    'temperature': 10.0,      # K
    'linewidth': 0.5,         # km/s
    'linewidth_freq': 0.017,  # MHz
    'rms': 0.015 or {...},    # K or dict
    'resolution': 0.1,        # MHz
    'source_size': 1E20,      # arcsec
    'execution_time': 245.3,  # seconds
    'output_files': {
        'interactive_plot': 'path/to/fit_spectrum.html',
        'peak_results': 'path/to/final_peak_results.txt',
        'detailed_report': 'path/to/output_report.txt',
        'column_densities': 'path/to/column_density_results.txt'
    }
}
```

---

## Tips and Best Practices

### Recommendation:
- If false positive assignments, add the molecules to `force_ignore_molecules` list and run again

### Common Pitfalls:
- ❌ Providing VLSR without exact temperature
- ❌ Setting `sigma_threshold` or `rms_noise` too low (creates false detections) or too high (not finding enough lines)
- ❌ Restricting `valid_atoms` incorrectly (misses real molecules or allows false-positive molecules)


---

## Further Help

- **Contact:** zfried@mit.edu

---

*Last updated: February 2026*
