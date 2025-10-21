# Astro AMASE

**Automated Molecular Assignment and Source Parameter Estimation in Radio Astronomical Observations**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Astro AMASE is a comprehensive Python package for automated molecular line identification in radio astronomical observations. It combines spectroscopic analysis, structural relevance scoring, and best-fit modeling to provide robust molecular assignments.

## Features

✨ **Automated Analysis Pipeline**
- Spectral line peak detection with adaptive sigma thresholding
- Automatic VLSR and temperature determination
- Linewidth calculation via Gaussian fitting

🔬 **Molecular Line Assignment**
- Query CDMS and JPL molecular databases
- Iterative assignment with structural relevance scoring (VICGAE embeddings)
- Context-aware rescoring as detected molecules accumulate
- Handles blended lines and multiple carriers

📊 **Best-Fit Modeling**
- Column density optimization via least-squares fitting
- Interactive Bokeh visualizations
- Quality control and molecule filtering
- Comprehensive output reports

## Installation

### From PyPI (once published, which hasn't happened yet)

```bash
pip install astro_amase
```

### From Source

```bash
git clone https://github.com/zfried/astro_amase.git
cd astro_amase
pip install -e .
```

## Quick Start

### Required Database Files

The package requires several files to be downloaded from the following [Dropbox folder](https://www.dropbox.com/scl/fo/s1dhye6mrdistrm0vbim7/ALRlugfuxnsHZU4AisPWjig?rlkey=7fk1obwvkeihlo8jt84g2wqfr&st=hqrts8cd&dl=0). These files are relatively large and include local copies of the CDMS and JPL molecular databases, as well as **molsim** `Molecule` objects for the catalogs. All files should be saved in the same local directory where your output files will be written. The path to this directory should then be provided as the directory_path argument in the relevant functions.

### Using Direct Parameters

There are several usage examples in the `notebooks/example_notebook.ipynb` file.

```python
import astro_amase

results = astro_amase.assign_observations(
    spectrum_path='spectrum.txt',
    directory_path='./analysis/',
    temperature=150.0,
    sigma_threshold=5.0,
    observation_type='interferometric',
    beam_major_axis=0.5,
    beam_minor_axis=0.5,
    source_size=1E20,
    continuum_temperature=2.7,
    valid_atoms=['C', 'O', 'H', 'N', 'S']
)
```


### Using a Configuration File

```python
import astro_amase

# Run complete analysis
results = astro_amase.assign_observations('config.yaml')

# Access results
print(f"Assigned lines: {results['statistics']['assigned']}")
print(f"Detected molecules: {results['statistics']['unique_detected_molecules']}")
print(f"VLSR: {results['vlsr']:.2f} km/s")
print(f"Temperature: {results['temperature']:.1f} K")
```


## Configuration File Example

See `examples/config_template.yaml` for a complete template.

## Input Requirements

### Spectrum File Format

Plain text file with two columns (space or tab separated) and no header:
- Column 1: Frequency (MHz)
- Column 2: Intensity (Kelvin)

The code was designed for data with intensity units of K. For accurate determination of column density and temperature, the intensity units must indeed be in K. However, even if the data are in Jy/beam, the line assignments should still be reasonably reliable.

Example:
```
345000.0        0.05
345000.1        0.06
345000.2        0.08
...
```

## Output Files

Running the analysis produces several output files:

- **`fit_spectrum.html`**: Interactive Bokeh plot showing:
  - Observed spectrum (black)
  - Total fitted spectrum (red)
  - Individual molecular contributions (colored, toggleable)

- **`final_peak_results.csv`**: Peak-by-peak assignments
  ```csv
  peak_freq,experimental_intensity_max,total_simulated_intensity,difference,carrier_molecules
  345123.456,10.5,9.8,0.7,"['CH3OH', 'H2CO']"
  ```

- **`output_report.txt`**: Detailed text report with:
  - Assignment status for each line
  - Candidate molecules and scores
  - Quality issues and penalties
  - Summary statistics

- **`column_density_results.csv`**: Best-fit column densities
  ```csv
  molecule,column_density,smiles
  CH3OH,1.5e15,CO
  H2CO,8.2e14,C=O
  ```

## Algorithm Overview

1. **Data Loading & Peak Detection**
   - Load spectrum and detect peaks above σ threshold
   - Calculate RMS noise

2. **Linewidth Determination**
   - Gaussian fitting to strongest peaks
   - Median FWHM calculation
   - Conversion to velocity width

3. **VLSR & Temperature Estimation** (if unknown)
   - Database query for candidate transitions
   - VLSR clustering analysis
   - Least-squares optimization

4. **Dataset Creation**
   - Query CDMS/JPL/LSD for candidates within Δν
   - Simulate spectra at observational parameters
   - Filter duplicates and apply quality control

5. **Iterative Line Assignment**
   - Static checks (invalid atoms, vibrational states, intensity checks)
   - Dynamic scoring (structural relevance via VICGAE)
   - Softmax and combined score calculation
   - Reassignment when new molecules detected

6. **Best-Fit Modeling**
   - Build lookup tables for rapid simulation
   - Optimize column densities via least-squares
   - Quality filtering (remove weak contributors)
   - Generate visualizations

## Advanced Usage

### Accessing Detailed Results

```python
results = astro_amase.assign_observations('config.yaml')

# Get the assigner object
assigner = results['assigner']

# Individual line details
for line in assigner.lines[:10]:  # First 10 lines
    if line.assignment_status:
        print(f"{line.frequency:.4f} MHz: {line.assignment_status.value}")
        if line.assigned_molecule:
            print(f"  → {line.assigned_molecule}")
            print(f"  Score: {line.best_candidate.global_score:.2f}")
```

### Programmatic Analysis

```python
import pandas as pd

# Load peak results
peaks = pd.read_csv('directory/final_peak_results.csv')

# Find strongest assigned lines
assigned = peaks[peaks['carrier_molecules'] != "['Unidentified']"]
strongest = assigned.nlargest(10, 'experimental_intensity_max')

# Load column densities
columns = pd.read_csv('directory/column_density_results.csv')
print(columns.sort_values('column_density', ascending=False))
```

## Citation

Paper is in prep!

## Requirements

- Python ≥ 3.8
- pandas ≥ 1.3.0
- numpy ≥ 1.20.0
- rdkit ≥ 2021.09.1
- scipy ≥ 1.7.0
- bokeh ≥ 2.4.0
- numba ≥ 0.54.0
- astropy ≥ 4.3.0
- matplotlib ≥ 3.4.0
- pyyaml ≥ 5.4.0
- astrochem_embedding ≥ 0.1.0

## License

TBD


## Support

For questions, issues, or feedback:
- 📧 Email: zfried@mit.edu
- 🐛 Issues: [GitHub Issues](https://github.com/zfried/astro_amase/issues)

## Acknowledgments

- [CDMS](https://cdms.astro.uni-koeln.de/classic/) (Cologne Database for Molecular Spectroscopy)
- [JPL](https://spec.jpl.nasa.gov/) Molecular Spectroscopy Database
- [LSD](https://lsd.univ-lille.fr/) (Lille Spectroscopic Database)
- [astrochem_embedding](https://github.com/laserkelvin/astrochem_embedding) for VICGAE structural relevance scoring
- [molsim](https://github.com/bmcguir2/molsim) for spectral simulation tools


---

**Astro AMASE** - Making molecular line identification in radio astronomy automated.
