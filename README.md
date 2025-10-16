# Astro AMASE

**Automated Molecular Assignment and Source Parameter Estimation in Radio Astronomical Observations**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

### From PyPI (once published)

```bash
pip install astro_amase
```

### From Source

```bash
git clone https://github.com/zfried/astro_amase.git
cd astro_amase
pip install -e .
```

### For Development

```bash
git clone https://github.com/zfried/astro_amase.git
cd astro_amase
pip install -r requirements.txt
pip install -e .
```

## Quick Start

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

### Using Direct Parameters

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

### Command-Line Interface

```bash
# Interactive mode
astro-amase

# With config file
astro-amase --config config.yaml
```

## Configuration File Example

```yaml
# config.yaml
spectrum_path: /path/to/spectrum.txt
directory_path: /path/to/output/

# RMS noise (null for automatic determination)
rms_noise: null

# Line detection threshold
sigma_threshold: 5.0

# Source parameters
vlsr: null  # Will be determined automatically
temperature: 150.0
temperature_is_exact: false

# Observation setup
observation_type: interferometric
beam_major_axis: 0.5
beam_minor_axis: 0.5
source_size: 1E20
continuum_temperature: 2.7

# Valid atoms
valid_atoms: "C, O, H, N, S"
```

See `examples/config_template.yaml` for a complete template.

## Input Requirements

### Spectrum File Format

Plain text file with two columns (space or tab separated):
- Column 1: Frequency (MHz)
- Column 2: Intensity (Kelvin or Jy/beam)

Example:
```
# frequency_MHz  intensity_K
345000.0        0.05
345000.1        0.06
345000.2        0.08
...
```

### Required Database Files

The package requires CDMS/JPL molecular database files in the working directory:
- `all_cdms_final_official.csv`
- `all_jpl_final_official.csv`
- `cdms_pkl/` directory with pickled molecule objects
- `jpl_pkl/` directory with pickled molecule objects
- `transitions_database.pkl.gz`

## Output Files

Running the analysis produces several output files:

- **`fit_spectrum.html`**: Interactive Bokeh plot showing:
  - Observed spectrum (black)
  - Total fitted spectrum (red)
  - Individual molecular contributions (colored, toggleable)
  - Peak identifications

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
   - Query CDMS/JPL for candidates within Δν
   - Simulate spectra at observational parameters
   - Filter duplicates and apply quality control

5. **Iterative Line Assignment**
   - Static checks (invalid atoms, vibrational states)
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

# Detected molecules summary
mol_summary = assigner.get_detected_molecules_summary()
for smiles, info in mol_summary.items():
    print(f"{info['formula']}: {info['count']} detections")

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
peaks = pd.read_csv('output/final_peak_results.csv')

# Find strongest assigned lines
assigned = peaks[peaks['carrier_molecules'] != "['Unidentified']"]
strongest = assigned.nlargest(10, 'experimental_intensity_max')

# Load column densities
columns = pd.read_csv('output/column_density_results.csv')
print(columns.sort_values('column_density', ascending=False))
```

## Citation

If you use Astro AMASE in your research, please cite:

```bibtex
@software{astro_amase,
  author = {Fried, Zachary},
  title = {Astro AMASE: Automated Molecular Assignment and Source parameter Estimation},
  year = {2025},
  url = {https://github.com/zfried/astro_amase}
}
```

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For questions, issues, or feedback:
- 📧 Email: zfried@mit.edu
- 🐛 Issues: [GitHub Issues](https://github.com/zfried/astro_amase/issues)

## Acknowledgments

- CDMS (Cologne Database for Molecular Spectroscopy)
- JPL Molecular Spectroscopy Database
- astrochem_embedding for VICGAE structural relevance scoring
- molsim for spectral simulation tools

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

**Astro AMASE** - Making molecular line identification in radio astronomy automated and accessible.