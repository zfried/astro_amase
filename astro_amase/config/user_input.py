"""
User input handling for AMASE.
Handles all user interactions, input validation, and parameter collection.
"""

import os
import pandas as pd
from rdkit import Chem
from typing import List, Tuple, Optional, Dict, Any
from ..constants import DEFAULT_VALID_ATOMS, ALL_VALID_ATOMS

def get_spectrum_path() -> str:
    """
    Get and validate spectrum file path.
    
    The spectrum file must be a .txt file with:
    - First column: frequency in MHz
    - Second column: intensity
    - Tab or space separated values
    """
    while True:
        print("Spectrum file requirements:")
        print("- Must be a .txt file")
        print("- First column: frequency (MHz)")
        print("- Second column: intensity")
        #print("- Tab or space separated")
        print()
        
        path = input('Please enter full path to spectrum file:\n').strip()
        
        if not os.path.isfile(path):
            print('File not found. Please enter a valid file path.')
            print()
            continue
            
        if not path.lower().endswith('.txt'):
            print('File must be a .txt file.')
            print()
            continue
            
        return path

def get_directory_path() -> str:
    """Get and validate directory path for file storage."""
    while True:
        path = input('Please enter full path to directory where required files are stored. This is also where the outputs will be saved:\n').strip()
        # Clean up path
        path = ''.join(path.split())
        if path[-1] != '/':
            path = path + '/'
        
        if os.path.exists(path) and os.access(path, os.W_OK):
            return path
        else:
            print('Directory not found or not writable. Please enter a valid directory path.')
            print()


def get_local_catalogs_info() -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """Get local catalogs configuration."""
    while True:
        response = input('Do you have catalogs on your local computer that you would like to consider (y/n): \n').strip().lower()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no']:
            return False, None, None
        else:
            print('Invalid input. Please just type y or n')
            print()

    # Get local directory
    while True:
        local_dir = input('Great! Please enter path to directory to your local spectroscopic catalogs:\n').strip()
        local_dir = ''.join(local_dir.split())
        if local_dir[-1] != '/':
            local_dir = local_dir + '/'
        
        if os.path.exists(local_dir):
            break
        else:
            print('Directory not found. Please enter a valid directory path.')
            print()

    # Get local DataFrame
    while True:
        try:
            df_path = input(
                'Please enter path to the csv file that contains the SMILES strings and isotopic composition of molecules in local catalogs:\n'
            ).strip()
            df = pd.read_csv(df_path)
            required_columns = ['name', 'smiles', 'iso']
            if all(col in df.columns for col in required_columns):
                return True, local_dir, df
            else:
                print(f'CSV file must contain columns: {required_columns}')
                print()
        except Exception as e:
            print(f'Error reading CSV file: {e}')
            print()


def get_sigma_threshold() -> float:
    """Get sigma threshold for line detection."""
    while True:
        try:
            sigma = float(input('Sigma threshold (the code will only attempt to assign lines greater than sigma*rms noise):\n'))
            if sigma > 0:
                return sigma
            else:
                print('Please enter a positive value.')
                print()
        except ValueError:
            print('Please enter a valid value.')
            print()


     
def get_vlsr() -> float:
    """Get VLSR (radial velocity with respect to Local Standard of Rest)."""
    while True:
        try:
            # Ask user if they know the VLSR
            choice = input('Do you know the VLSR (velocity with respect to LSR)? (y/n): \n If not, the algorithm will determine it for you. \n '
                           
            'Note: if you know the vlsr, you must also input a temperature since they are determined simultaneously. ').lower().strip()

            if choice == 'y' or choice == 'yes':
                # User knows VLSR
                while True:
                    try:
                        vlsr = float(input('Please enter the VLSR (in km/s): '))
                        return vlsr, True
                    except ValueError:
                        print('Please enter a valid number.')
                        print()
            
            elif choice == 'n' or choice == 'no':
                # User doesn't know VLSR
                print('VLSR unknown - will need to be determined from other measurements.')
                return None, False
            
            else:
                print('Please enter "y" for yes or "n" for no.')
                print()
                
        except KeyboardInterrupt:
            print('\nOperation cancelled.')
            return None, None


def get_rms_noise() -> tuple[float, bool]:
    """
    Get RMS noise level with option for user input or automatic calculation.
    
    Returns:
        tuple: (rms_value, is_manual)
            - rms_value: float or None (RMS noise level, None if auto-calculate)
            - is_manual: bool (True if user provided value, False if auto-calculate)
    """
    
    while True:
        try:
            choice = input(
                'Would you like to input the RMS noise level? (y/n):\n'
                'Otherwise it will be determined automatically from the data.\n'
            ).strip().lower()
            
            if choice in ['y', 'yes']:
                # User wants to input RMS noise
                while True:
                    try:
                        rms = float(input('Please enter the RMS noise level: '))
                        if rms > 0:
                            print(f'RMS noise level set to: {rms}')
                            print()
                            return rms, True
                        else:
                            print('RMS noise must be a positive value.')
                            print()
                    except ValueError:
                        print('Please enter a valid number.')
                        print()
            
            elif choice in ['n', 'no']:
                print('RMS noise will be determined automatically from the data.')
                print()
                return None, False
            
            else:
                print('Please enter "y" for yes or "n" for no.')
                print()
                
        except KeyboardInterrupt:
            print('\nOperation cancelled.')
            return None, False


def get_temperature(know_vlsr) -> float:
    """Get experimental temperature with option for exact or estimated value."""
    while True:
        if know_vlsr == False:
            try:
                # Ask user if they know exact temperature or need to guess
                temp_choice = input('Do you know the temperature or would you like the code to estimate it?\n' \
                            ' If you want the code to estimate, you will need to input a best guess and the code will determine a best-fit temperature within ±100K of this value.\n'
                            '1 - I know the exact temperature\n'
                            '2 - I need to estimate the temperature\n'
                            'Please enter 1 or 2: ')

                if temp_choice == '1':
                    # User knows exact temperature
                    while True:
                        try:
                            temp = float(input('Please enter the exact excitation temperature (in Kelvin): '))
                            if temp > 0:
                                return temp, True
                            else:
                                print('Temperature must be positive.')
                                print()
                        except ValueError:
                            print('Please enter a valid number.')
                            print()

                elif temp_choice == '2':
                    # User needs to estimate temperature
                    while True:
                        try:
                            temp = float(input('Please enter your best estimate of the excitation temperature (in Kelvin): '))
                            if temp > 0:
                                return temp, False
                            else:
                                print('Temperature must be positive.')
                                print()
                        except ValueError:
                            print('Please enter a valid number.')
                            print()
                
                else:
                    print('Please enter either 1 or 2.')
                    print()
                    
            except KeyboardInterrupt:
                print('\nOperation cancelled.')
                return None
        else:
            try:
                temp = float(input('Please enter the exact excitation temperature (in Kelvin): '))
                if temp > 0:
                    return temp, True
                else:
                    print('Temperature must be positive.')
                    print()
            except ValueError:
                print('Please enter a valid number.')
                print()




def get_valid_atoms() -> List[str]:
    """Get list of valid atoms that could be present."""
    
    while True:
        print("Which atoms could feasibly be present in the mixture?")
        print("1. Default (C, O, H, N, S)")
        print("2. All atoms in periodic table")
        print("3. Specify custom atoms as a comma separated list (e.g. C,O,H,N)")
        print()
        
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            return DEFAULT_VALID_ATOMS
        elif choice == '2':
            return ALL_VALID_ATOMS
        elif choice == '3':
            while True:
                atoms_input = input("Enter atoms separated by commas (e.g., C,O,H,N): ").strip()
                
                # Parse comma-separated atoms
                atoms = [atom.strip() for atom in atoms_input.split(',') if atom.strip()]
                
                if not atoms:
                    print("Please enter at least one atom.")
                    print()
                    continue
                
                # Check if all atoms are in ALL_VALID_ATOMS
                invalid_atoms = [atom for atom in atoms if atom not in ALL_VALID_ATOMS]
                
                if invalid_atoms:
                    print(f"Invalid atoms: {', '.join(invalid_atoms)}")
                    print(f"Please only use atoms from the valid list: {', '.join(ALL_VALID_ATOMS)}")
                    print()
                    continue
                
                return atoms
        else:
            print("Please enter 1, 2, or 3.")
            print()


def get_structure_consideration() -> bool:
    """Ask if user wants to consider structural relevance."""
    while True:
        response = input('Do you want to consider structural relevence? If not, only the spectroscopy will be considered (y/n): \n').strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print('Invalid input. Please just type y or n')
            print()


def get_smiles_manual() -> List[str]:
    """Get SMILES strings manually from user input."""
    while True:
        try:
            smiles_input = input(
                'Enter the SMILES strings of the initial detected molecules. '
                'Please separate the SMILES string with a comma: \n'
            )
            smiles_list = [s.strip() for s in smiles_input.split(',') if s.strip()]
            
            # Validate SMILES
            validated_smiles = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    validated_smiles.append(Chem.MolToSmiles(mol))
                else:
                    print(f'Invalid SMILES: {smiles}')
                    raise ValueError('Invalid SMILES detected')
            
            return validated_smiles
            
        except Exception as e:
            print('You entered an invalid SMILES string. Please try again.')
            print()


def get_smiles_from_csv() -> List[str]:
    """Get SMILES strings from CSV file."""
    while True:
        try:
            csv_path = input(
                'Please enter path to csv file. This needs to have the detected molecules '
                'in a column listed "SMILES."\n'
            ).strip()
            
            df = pd.read_csv(csv_path)
            if 'SMILES' not in df.columns:
                print('CSV file must contain a "SMILES" column.')
                print()
                continue
            
            smiles_list = df['SMILES'].dropna().tolist()
            
            # Validate SMILES
            validated_smiles = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    validated_smiles.append(Chem.MolToSmiles(mol))
                else:
                    print(f'Invalid SMILES in CSV: {smiles}')
                    raise ValueError('Invalid SMILES detected')
            
            return validated_smiles
            
        except Exception as e:
            print(f'Error reading CSV file: {e}')
            print()


def get_starting_molecules(consider_structure: bool) -> List[str]:
    """Get initial detected molecules if any."""
    if not consider_structure:
        return []
    
    # Ask if user has known precursors
    while True:
        response = input('Do you have any known molecular precursors? (y/n): \n').strip().lower()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no']:
            return []
        else:
            print('Invalid input. Please just type y or n')
            print()

    # Get input method
    while True:
        method = input(
            'You need to input the SMILES strings of the initial detected molecules.\n'
            'If you would like to type them individually, type 1. If you would like to input a csv file, type 2: \n'
        ).strip()
        
        if method == '1':
            return get_smiles_manual()
        elif method == '2':
            return get_smiles_from_csv()
        else:
            print('Please enter 1 or 2.')
            print()

def get_observation_parameters() -> tuple:
    """
    Get observation type and beam parameters from user input.
    
    Returns:
        tuple: (observation_type, dish_size_or_bmaj, bmin_or_None, source_diameter)
            - observation_type: str ('single_dish' or 'interferometric')
            - dish_size_or_bmaj: float (dish diameter in meters OR beam major axis in arcsec)
            - bmin_or_None: float or None (beam minor axis in arcsec, None for single dish)
            - source_diameter: float (source diameter in arcseconds)
    """
    
    # Get observation type
    while True:
        obs_choice = input(
            'Select observation type:\n'
            '1. Single dish\n'
            '2. Interferometric\n'
            'Enter choice (1 or 2): \n'
        ).strip()
        
        if obs_choice == '1':
            print('Selected: Single dish observation')
            print()
            
            # Get dish size for single dish
            while True:
                try:
                    dish_size = float(input('Enter dish diameter (meters): \n'))
                    
                    if dish_size > 0:
                        print(f'Dish diameter: {dish_size} m')
                        print()
                        break
                    else:
                        print('Dish size must be positive.')
                        print()
                except ValueError:
                    print('Please enter a valid number.')
                    print()
            
            param1 = dish_size
            param2 = None
            break
            
        elif obs_choice == '2':
            print('Selected: Interferometric observation')
            print()
            
            # Get beam parameters for interferometric data
            while True:
                try:
                    bmaj = float(input('Enter synthesized beam major axis (arcseconds): \n'))
                    bmin = float(input('Enter synthesized beam minor axis (arcseconds): \n'))
                    
                    if bmaj > 0 and bmin > 0:
                        print(f'Beam parameters: {bmaj}" × {bmin}" (major × minor axis)')
                        print()
                        break
                    else:
                        print('Beam sizes must be positive values.')
                        print()
                except ValueError:
                    print('Please enter valid numbers.')
                    print()
            
            param1 = bmaj
            param2 = bmin
            break
        else:
            print('Please enter 1 or 2.')
            print()
    
    # Get source diameter (for both observation types)
    while True:
        try:
            source_diameter = float(input('Enter source diameter (arcseconds): \n'
            'Type 1E20 if the source fills the beam: \n'))
            
            if source_diameter > 0:
                print(f'Source diameter: {source_diameter}')
                print()
                return obs_choice, param1, param2, source_diameter
            else:
                print('Source diameter must be positive.')
                print()
        except ValueError:
            print('Please enter a valid number.')
            print()


def get_continuum_temperature() -> float:
    """
    Get continuum temperature from user input.
    
    Returns:
        float: Continuum temperature in Kelvin
    """
    
    while True:
        try:
            temp = float(input('Enter the continuum temperature (K): \n'))
            
            if temp > 0:
                print(f'Continuum temperature set to: {temp} K')
                print()
                return temp
            else:
                print('Temperature must be a positive value.')
                print()
        except ValueError:
            print('Please enter a valid number.')
            print()




def collect_all_parameters() -> Dict[str, Any]:
    """Collect all user parameters in sequence."""
    print('')
    print('')
    
    spectrum_path = get_spectrum_path()
    print('')
    directory_path = get_directory_path()
    print('')
    rms_noise, rms_manual = get_rms_noise()
    print('')
    sigma_threshold = get_sigma_threshold()
    print('')
    vlsr_input, vlsr_choice = get_vlsr()
    print('')
    temperature, temp_choice = get_temperature(vlsr_choice)
    print('')
    obs_type, bmaj_or_dish, bmin, source_size = get_observation_parameters()
    continuum_temp = get_continuum_temperature()
    valid_atoms = get_valid_atoms()
    print('')

    parameters = {
        'spectrum_path': spectrum_path,
        'directory_path': directory_path,
        'vlsr_input': vlsr_input,
        'vlsr_known': vlsr_choice,
        'temperature_choice': temp_choice,
        'sigma_threshold': sigma_threshold,
        'rms_noise': rms_noise,
        'rms_manual': rms_manual,
        'temperature': temperature,
        'valid_atoms': valid_atoms,
        'observation_type': obs_type,
        'bmaj_or_dish': bmaj_or_dish,
        'bmin': bmin,
        'source_size': source_size,
        'continuum_temperature': continuum_temp
    }
    
    print()
    print('Thanks for the input!')
    
    return parameters