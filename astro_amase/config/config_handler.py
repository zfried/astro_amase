"""
Configuration file handler for AMASE.
Supports both interactive mode and config file input.
"""

import json
import yaml
import argparse
from typing import Dict, Any, Optional
import os
from .user_input import collect_all_parameters

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file (.json or .yaml/.yml)
        
    Returns:
        Dictionary of configuration parameters
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Determine file type and load
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Config file must be .json, .yaml, or .yml")
    
    return validate_config(config)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and process configuration parameters.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated and processed configuration
    """
    required_fields = [
        'spectrum_path',
        'directory_path',
        'sigma_threshold',
        'temperature',
        'observation_type',
        'source_size',
        'continuum_temperature',
        'valid_atoms'
    ]
    
    # Check required fields
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    # Validate file paths
    if not os.path.isfile(config['spectrum_path']):
        raise FileNotFoundError(f"Spectrum file not found: {config['spectrum_path']}")
    
    if not os.path.exists(config['directory_path']):
        raise FileNotFoundError(f"Directory not found: {config['directory_path']}")
    
    # Ensure directory path ends with /
    if not config['directory_path'].endswith('/'):
        config['directory_path'] += '/'
    
    # Handle RMS noise
    if 'rms_noise' in config and config['rms_noise'] is not None:
        config['rms_noise'] = float(config['rms_noise'])
        config['rms_manual'] = True
    else:
        config['rms_noise'] = None
        config['rms_manual'] = False
    
    # Validate observation type
    obs_type = config['observation_type']
    if obs_type not in ['1', '2', 'single_dish', 'interferometric']:
        raise ValueError("observation_type must be '1' (single_dish) or '2' (interferometric)")
    
    # Standardize observation type to '1' or '2'
    if obs_type == 'single_dish':
        config['observation_type'] = '1'
    elif obs_type == 'interferometric':
        config['observation_type'] = '2'
    
    # Validate beam parameters based on observation type
    if config['observation_type'] == '1':
        if 'dish_diameter' not in config:
            raise ValueError("Single dish observation requires 'dish_diameter'")
        config['bmaj_or_dish'] = float(config['dish_diameter'])
        config['bmin'] = None
    else:  # interferometric
        if 'beam_major_axis' not in config or 'beam_minor_axis' not in config:
            raise ValueError("Interferometric observation requires 'beam_major_axis' and 'beam_minor_axis'")
        config['bmaj_or_dish'] = float(config['beam_major_axis'])
        config['bmin'] = float(config['beam_minor_axis'])
    
    # Handle VLSR
    if 'vlsr' in config and config['vlsr'] is not None:
        config['vlsr_input'] = float(config['vlsr'])
        config['vlsr_known'] = True
    else:
        config['vlsr_input'] = None
        config['vlsr_known'] = False
    
    # Handle temperature
    config['temperature'] = float(config['temperature'])
    if 'temperature_is_exact' in config:
        config['temperature_choice'] = config['temperature_is_exact']
    else:
        # If vlsr is known, temperature must be exact
        config['temperature_choice'] = config['vlsr_known']
    
    # Convert numeric values
    config['sigma_threshold'] = float(config['sigma_threshold'])
    config['source_size'] = float(config['source_size'])
    config['continuum_temperature'] = float(config['continuum_temperature'])
    
    # Validate atoms list
    if isinstance(config['valid_atoms'], str):
        config['valid_atoms'] = [atom.strip() for atom in config['valid_atoms'].split(',')]
    
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AMASE: Automated Molecular Assignment and Source parameter Estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python3 astro_amase.py
    
  Config file mode:
    python3 astro_amase.py --config config.json
    python3 astro_amase.py -c config.yaml
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration file (JSON or YAML). If not provided, runs in interactive mode.'
    )
    
    return parser.parse_args()


def get_parameters() -> Dict[str, Any]:
    """
    Get parameters either from config file or interactive input.
    
    Returns:
        Dictionary of all parameters needed for AMASE
    """
    args = parse_arguments()
    
    if args.config:
        print(f"Loading configuration from: {args.config}")
        print()
        parameters = load_config_file(args.config)
        print("Configuration loaded successfully!")
        print()
        
        # Print loaded parameters
        print("Loaded parameters:")
        print(f"  Spectrum file: {parameters['spectrum_path']}")
        print(f"  Output directory: {parameters['directory_path']}")
        print(f"  RMS noise: {parameters['rms_noise']}" if parameters['rms_manual'] else "  RMS noise: To be determined automatically")
        print(f"  Sigma threshold: {parameters['sigma_threshold']}")
        print(f"  Temperature: {parameters['temperature']} K {'(exact)' if parameters['temperature_choice'] else '(estimate)'}")
        print(f"  VLSR: {parameters['vlsr_input']} km/s" if parameters['vlsr_input'] is not None else "  VLSR: To be determined")
        obs_type = "Single dish" if parameters['observation_type'] == '1' else "Interferometric"
        print(f"  Observation type: {obs_type}")
        if parameters['observation_type'] == '1':
            print(f"  Dish diameter: {parameters['bmaj_or_dish']} m")
        else:
            print(f"  Beam: {parameters['bmaj_or_dish']}\" × {parameters['bmin']}\"")
        print(f"  Source size: {parameters['source_size']}\"")
        print(f"  Continuum temperature: {parameters['continuum_temperature']} K")
        print(f"  Valid atoms: {', '.join(parameters['valid_atoms'])}")
        print()
    else:
        print("Running in interactive mode...")
        print()
        parameters = collect_all_parameters()
    
    return parameters