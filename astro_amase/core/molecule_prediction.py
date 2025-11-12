"""
Molecular Prediction Module for Astrochemistry

This module implements a fragment-based molecular prediction pipeline for discovering
new molecules likely to exist in astronomical sources based on a set of detected molecules.

Overview:
---------
The prediction workflow combines neural network fragment prediction, stochastic molecular
assembly, and Gaussian scoring in latent chemical space to generate and rank candidate
molecules that are chemically similar to a set of detected molecules.

Key Components:
--------------
1. **Fragment Prediction (SubgroupMLP)**:
   - Neural network that predicts probability distributions over 326 molecular fragments
   - Takes 32D latent space vectors as input
   - Trained to identify which fragments are likely useful in different chemical regions

2. **Latent Space Sampling**:
   - Uses VICGAE embeddings to represent molecules as 32D vectors
   - Creates Gaussian distributions centered on detected molecules
   - Samples new points in latent space to guide fragment selection

3. **Stochastic Molecular Assembly**:
   - Uses Group SELFIES grammar for fragment-based molecule construction
   - Randomly selects and assembles 1-4 fragments based on predicted probabilities
   - Validates generated molecules for chemical correctness and atom composition

4. **Gaussian Scoring**:
   - Ranks candidate molecules by cumulative probability density under Gaussians
   - Molecules closer to detected molecules in latent space score higher
   - Provides interpretable likelihood scores for prioritization

Workflow:
--------
1. Load detected molecules and embed them in 32D latent space
2. Create Gaussian distributions centered on each detected molecule
3. Sample random vectors from latent space
4. Predict fragment probabilities for each sampled vector
5. Stochastically assemble fragments into candidate molecules
6. Filter candidates by chemical properties (radicals, ions, heteroatoms)
7. Embed candidates and score them by Gaussian probability density
8. Return ranked list of most likely candidate molecules


"""

from ..constants import vicgae_model
from scipy.spatial.distance import mahalanobis
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
# Suppress RDKit deprecation warnings
RDLogger.DisableLog('rdApp.warning')
from group_selfies import (
    Group,
    GroupGrammar,
)
import random
from rdkit import Chem
import time
from scipy.stats import multivariate_normal
import os


class SubgroupMLP(torch.nn.Module):
    """
    Multi-layer perceptron for predicting molecular fragment probabilities from latent vectors.

    This neural network takes a 32-dimensional chemical latent space vector and predicts
    probability distributions over 326 possible molecular fragments that could be used
    to construct new molecules.

    Architecture:
        - Input: 32D latent vector
        - Hidden: 256 units (ReLU) → 256 units (ReLU)
        - Output: 326 probabilities (Sigmoid)

    The model is trained to predict which molecular fragments are likely to be useful
    for constructing molecules in regions of chemical space.
    """
    def __init__(self):

        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(32,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 326),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        return x


def compute_pdf(mean, cov, allVectors, scale):
    """
    Compute the probability density function (PDF) values for vectors using a multivariate Gaussian distribution.

    Args:
        mean (np.ndarray): Mean vector of the Gaussian distribution
        cov (np.ndarray): Covariance matrix of the Gaussian distribution
        allVectors (np.ndarray): Array of vectors to evaluate the PDF on
        scale (float): Scaling factor to multiply the PDF values

    Returns:
        np.ndarray: Scaled PDF values for each input vector
    """
    gaussian_pdf = multivariate_normal(mean=mean, cov=cov)
    return scale * gaussian_pdf.pdf(allVectors)



def getRandomVectors(detectedSmiles):
    """
    Generate random vectors in the chemical latent space around detected molecules.

    This function embeds detected SMILES strings into a latent space using VICGAE model,
    creates Gaussian distributions around each embedding, and samples random vectors
    from regions between all molecules. The importance of each molecule is weighted
    by its Mahalanobis distance to other molecules.

    Args:
        detectedSmiles (list): List of SMILES strings for detected molecules

    Returns:
        tuple: (all_random_vectors, gaussian_params)
            - all_random_vectors (np.ndarray): Array of random vectors sampled uniformly in latent space
            - gaussian_params (list): List of tuples (mean, cov, weight) for each Gaussian distribution
    """
    covParam, span = 0.0105, 0.5

    inputVectors = []
    gaussian_params = []
    for g in detectedSmiles:
        t = vicgae_model.embed_smiles(g)
        vic_out = t.squeeze(0).tolist()
        inputVectors.append(vic_out)

    # Check if we have any vectors
    if len(inputVectors) == 0:
        raise ValueError("No molecules could be embedded from detectedSmiles. Cannot proceed with prediction.")

    d = len(inputVectors[0])
    for i in range(len(inputVectors)):
        mean = np.array(inputVectors[i])
        cov = np.eye(d) * covParam
        gaussian_params.append((mean, cov))
        
    
    mScoreList = []
    
    for z in range(len(gaussian_params)):
        mean1 = gaussian_params[z][0]
        cov1 = gaussian_params[z][1]
        inv_cov = np.linalg.inv(cov1)
        iScore = 0
        for q in range(len(gaussian_params)):
            if q != z:
                mean2 = gaussian_params[q][0]
                mahalanobis_score = mahalanobis(mean1, mean2, inv_cov)
                iScore += mahalanobis_score
    
        mScoreList.append(float(iScore))
    
    mScoreList2 = np.array(mScoreList) ** 3.7
    mScoreList = mScoreList2
    gpNew = [(gaussian_params[i][0], gaussian_params[i][1], mScoreList[i]) for i in range(len(gaussian_params))]
    gaussian_params = gpNew
    all_random_vectors = []
    inputVectors = np.array(inputVectors)  # Convert once if not already
    min_vectors = inputVectors - (span * covParam)
    max_vectors = inputVectors + (span * covParam)
    
    # Generate all random samples at once
    all_random_vectors = np.random.uniform(
        low=min_vectors[:, np.newaxis, :],  
        high=max_vectors[:, np.newaxis, :],  
        size=(len(inputVectors), 90, d)     
    ).reshape(-1, d) 


    return all_random_vectors, gpNew
    
    
def getGroupPredictions(detectedSmiles, frag_list_smiles, frag_list_filtered, directory_path):
    """
    Generate subgroup/fragment predictions for candidate molecules using a trained neural network.

    This function creates a molecular grammar from fragment lists, loads a trained MLP model,
    generates random vectors in latent space, and predicts probability distributions over
    molecular fragments for constructing new molecules.

    Args:
        detectedSmiles (list): List of SMILES strings for detected molecules
        frag_list_smiles (list): List of fragment SMILES strings
        frag_list_filtered (list): List of filtered fragment SELFIES strings for grammar
        directory_path (str): Path to directory containing model weights

    Returns:
        tuple: (predictions_np, gaussian_params, grammar)
            - predictions_np (np.ndarray): Predicted probabilities for each fragment (shape: N x 326)
            - gaussian_params (list): Gaussian distribution parameters for each detected molecule
            - grammar (GroupGrammar): Grammar object containing fragment definitions
    """
    
    grammar_list = []
    c = 0
    for i in frag_list_filtered:
        grammar_list.append(Group('frag'+str(c),i))
        c+=1 
    
    grammar = GroupGrammar(grammar_list)
    
    loaded_model = SubgroupMLP()
    loaded_model.load_state_dict(torch.load(os.path.join(directory_path,"model_weights_two_256.pth"), map_location=torch.device('cpu')))
    loaded_model.eval()  # important for inference
    
    random_vectors, gaussian_params = getRandomVectors(detectedSmiles)
    X = torch.tensor(random_vectors, dtype=torch.float32)
    
    # Make predictions (no gradients needed for inference)
    with torch.no_grad():
        predictions = loaded_model(X)
    
    # Convert back to numpy if needed
    predictions_np = predictions.numpy()

    return predictions_np, gaussian_params, grammar

    
def get_all_unique_atoms(smiles_list):
    """
    Get all unique atom symbols across all SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
    
    Returns:
        Sorted list of unique atom symbols
    """
    all_atoms = set()
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                for atom in mol.GetAtoms():
                    all_atoms.add(atom.GetSymbol())
        except:
            continue
    
    return list(all_atoms)

def getCandidates(detectedSmiles, frag_list_smiles, frag_list_filtered, valid_atoms, directory_path):
    """
    Generate candidate molecules by stochastically sampling and assembling molecular fragments.

    This function uses predicted fragment probabilities to construct new molecules by:
    1. Sampling fragments according to their predicted probabilities
    2. Randomly assembling 1-4 fragments together
    3. Optionally inserting ring-forming or branch tokens
    4. Validating that generated molecules only contain allowed atoms

    Args:
        detectedSmiles (list): List of SMILES strings for detected molecules
        frag_list_smiles (list): List of fragment SMILES strings
        frag_list_filtered (list): List of filtered fragment SELFIES strings for grammar
        valid_atoms (str or list): Either 'default' (use atoms from detectedSmiles) or list of valid atom symbols
        directory_path (str): Path to directory containing model weights

    Returns:
        tuple: (pred_smiles, gaussian_params)
            - pred_smiles (list): List of unique candidate SMILES strings
            - gaussian_params (list): Gaussian distribution parameters for detected molecules
    """

    if valid_atoms == 'default':
        validAtoms = get_all_unique_atoms(detectedSmiles)
        validAtoms_set = set(validAtoms)
    else:
        validAtoms = valid_atoms
        validAtoms_set = set(validAtoms)
    #print(validAtoms)
    pred_smiles = set()
    group_predictions, gaussian_params, grammar = getGroupPredictions(detectedSmiles, frag_list_smiles, frag_list_filtered, directory_path)
    #print(group_predictions.shape)
    tick = time.perf_counter()
    inCount = 0
    #print('len detected smiles',len(detectedSmiles))
    totalSmi = 0
    numEach = int(15000/group_predictions.shape[0])
    #print('num each',numEach)
    numEach = max(5,numEach)
    for pred in group_predictions:
        #print(pred.shape)
        for e in range(numEach):
            preds = pred.copy()
            preds_sum = preds.sum()

            # Avoid division by zero - skip if all predictions are zero
            if preds_sum == 0 or np.isclose(preds_sum, 0):
                print("Warning: All fragment predictions are zero, skipping this sample")
                continue

            probs = preds / preds_sum
            num_samples = np.random.randint(1, 5)
            selected_indices = np.random.choice(len(preds), size=num_samples, replace=True, p=probs)
            
            # Build the fragment tokens
            fragments = []
            for g in selected_indices:
                frag_name = 'frag' + str(g)
                start = random.randint(0, len(grammar.vocab[frag_name].attachment_points)-1)
                fragments.append(f"[:{start}{frag_name}]")
            
            # Randomly insert 0-3 tokens total from the special list between fragments
            # These add some randomness/branching to the molecules
            special_tokens = ['Ring1', 'Ring2', 'pop']
            num_insertions = np.random.randint(0, 4)  # 0 to 3 total
            
            # Randomly select which gaps to insert into (gaps are between fragments)
            num_gaps = len(fragments) - 1
            if num_gaps > 0 and num_insertions > 0:
                # Randomly choose which gaps get an insertion (at most num_insertions)
                insertion_positions = np.random.choice(num_gaps, size=min(num_insertions, num_gaps), replace=False)
            else:
                insertion_positions = []
            
            # Build the final string
            new_gselfies = ''
            for i, frag in enumerate(fragments):
                new_gselfies += frag
                # Check if we should insert a token after this fragment
                if i in insertion_positions:
                    token = random.choice(special_tokens)
                    new_gselfies += f"[{token}]"
            #print(new_gselfies)
            mol_out = grammar.decoder(new_gselfies)
            addMol = True
            for atom in mol_out.GetAtoms():
                o = atom.GetSymbol()
                if o not in validAtoms_set:
                    addMol = False
                    break
            if addMol:
                pred_smi = Chem.MolToSmiles(mol_out)
                totalSmi += 1
                if pred_smi in detectedSmiles:
                    inCount += 1
                    
                pred_smiles.add(pred_smi)
    
    tock = time.perf_counter()
    print('Time Taken (mins):', round((tock-tick)/60,3))
    pred_smiles = list(pred_smiles)
    print('Number of Unique and Valid Predicted Molecules:', len(pred_smiles))
    print('Number of Generated Molecules in detectedSmiles List:', inCount)
    print('Total Valid Smiles (including duplicates):',totalSmi)
    return pred_smiles, gaussian_params
    

def analyze_smiles_list(smiles_list):
    """
    Analyze a list of SMILES strings to count radicals, ions, and non-terminal heteroatoms
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        dict: Dictionary with counts and percentages
    """
    total_molecules = len(smiles_list)
    radical_count = 0
    ion_count = 0
    non_terminal_heteroatom_count = 0
    invalid_count = 0
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_count += 1
                continue
                
            # Check for radicals (unpaired electrons)
            is_radical = False
            for atom in mol.GetAtoms():
                if atom.GetNumRadicalElectrons() > 0:
                    is_radical = True
                    break
            
            if is_radical:
                radical_count += 1
            
            # Check for ions (formal charge != 0)
            total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            if total_charge != 0:
                ion_count += 1
            
            # Check for non-carbon atoms that are not terminal
            has_non_terminal_heteroatom = False
            for atom in mol.GetAtoms():
                # Skip carbon atoms
                if atom.GetSymbol() == 'C':
                    continue
                
                # Check if atom is terminal (degree = 1, meaning only one bond)
                # But also check if it's just hydrogen (which we typically ignore)
                if atom.GetSymbol() != 'H' and atom.GetDegree() > 1:
                    has_non_terminal_heteroatom = True
                    break
            
            if has_non_terminal_heteroatom:
                non_terminal_heteroatom_count += 1
                
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            invalid_count += 1
    
    # Calculate percentages
    valid_molecules = total_molecules - invalid_count
    
    results = {
        'total_molecules': total_molecules,
        'valid_molecules': valid_molecules,
        'invalid_molecules': invalid_count,
        'radical_count': radical_count,
        'ion_count': ion_count,
        'non_terminal_heteroatom_count': non_terminal_heteroatom_count,
        'radical_percentage': (radical_count / valid_molecules * 100) if valid_molecules > 0 else 0,
        'ion_percentage': (ion_count / valid_molecules * 100) if valid_molecules > 0 else 0,
        'non_terminal_heteroatom_percentage': (non_terminal_heteroatom_count / valid_molecules * 100) if valid_molecules > 0 else 0
    }
    
    return results

def getTopPredictedMols(detectedSmiles, frag_list_smiles,frag_list_filtered, rad_per, ion_per, hetero_per, valid_atoms, directory_path):
    """
    Generate and rank candidate molecules based on their likelihood in Gaussian latent space.

    This function generates candidate molecules, filters them by chemical properties matching
    the detected set, embeds them in latent space, and ranks them by their cumulative
    probability density under Gaussians centered at detected molecules.

    Args:
        detectedSmiles (list): List of SMILES strings for detected molecules
        frag_list_smiles (list): List of fragment SMILES strings
        frag_list_filtered (list): List of filtered fragment SELFIES strings for grammar
        rad_per (float): Probability of keeping radical molecules (0-1)
        ion_per (float): Probability of keeping ionic molecules (0-1)
        hetero_per (float): Probability of keeping molecules with non-terminal heteroatoms (0-1)
        valid_atoms (str or list): Either 'default' (use atoms from detectedSmiles) or list of valid atom symbols
        directory_path (str): Path to directory containing model weights

    Returns:
        list: Sorted list of candidate SMILES strings, ranked by likelihood (highest first).
              Molecules that fail to embed are excluded. Returns empty list if no molecules can be embedded.
    """

    pred_smiles, gaussian_params = getCandidates(detectedSmiles, frag_list_smiles,frag_list_filtered, valid_atoms, directory_path)

    pred_smiles = [i for i in pred_smiles if i not in detectedSmiles]

    pred_smiles2 = []
    for p in pred_smiles:
        mol_analyze = analyze_single_molecule(p)
        rad, ion, hetero = mol_analyze['is_radical'], mol_analyze['is_ion'], mol_analyze['has_non_terminal_heteroatom']
        per = 1
        if rad == True:
            per *= rad_per
        if ion == True:
            per *= ion_per
        if hetero == True:
            per *= hetero_per

        if should_keep(per):
            pred_smiles2.append(p)
        #else:
        #    print('deleting',per)

    pred_smiles = pred_smiles2

    # Embed molecules and track which ones succeed
    allVectors = []
    valid_pred_smiles = []
    for g in pred_smiles:
        try:
            embedding = vicgae_model.embed_smiles(g)
            allVectors.append(embedding)
            valid_pred_smiles.append(g)
        except Exception as e:
            print(f"Warning: Failed to embed SMILES '{g}': {e}")
            # Skip this molecule - don't add to either list
            continue

    # Only proceed if we have successfully embedded molecules
    if len(valid_pred_smiles) == 0:
        print("Warning: No molecules could be successfully embedded. Returning empty list.")
        return []

    # Initialize results array with correct size (matching valid_pred_smiles)
    accumulated_results = np.zeros(len(valid_pred_smiles))

    # Compute PDF for each detected molecule's Gaussian
    for i in range(len(detectedSmiles)):
        current_result = compute_pdf(gaussian_params[i][0], gaussian_params[i][1], allVectors, gaussian_params[i][2])
        accumulated_results += current_result

    # Sort by likelihood (highest first)
    sorted_indices = np.argsort(accumulated_results)[::-1]
    sorted_top_smiles = [valid_pred_smiles[i] for i in sorted_indices]
    return sorted_top_smiles

def analyze_single_molecule(smiles):
    """
    Analyze a single SMILES string to check if it's a radical, ion, or has non-terminal heteroatoms
    
    Args:
        smiles: Single SMILES string
        
    Returns:
        dict: Dictionary with boolean flags for each property
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'valid': False,
                'is_radical': False,
                'is_ion': False,
                'has_non_terminal_heteroatom': False,
                'error': 'Invalid SMILES'
            }
        
        # Check for radicals (unpaired electrons)
        is_radical = any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())
        
        # Check for ions (formal charge != 0)
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        is_ion = total_charge != 0
        
        # Check for non-carbon atoms that are not terminal
        has_non_terminal_heteroatom = any(
            atom.GetSymbol() not in ['C', 'H'] and atom.GetDegree() > 1 
            for atom in mol.GetAtoms()
        )
        
        return {
            'valid': True,
            'is_radical': is_radical,
            'is_ion': is_ion,
            'has_non_terminal_heteroatom': has_non_terminal_heteroatom,
            'error': None
        }
        
    except Exception as e:
        return {
            'valid': False,
            'is_radical': False,
            'is_ion': False,
            'has_non_terminal_heteroatom': False,
            'error': str(e)
        }

def should_keep(probability):
    """
    Decide whether to keep a molecule based on probability
    
    Args:
        probability: Float between 0.0 and 1.0
    
    Returns:
        bool: True if should keep, False if should reject
    """
    return random.random() < probability


def molecule_prediction(directory_path, detected_smiles, valid_atoms = 'default'):
    """
    Main entry point for predicting new molecules based on a set of detected molecules.

    This function orchestrates the entire molecule prediction pipeline:
    1. Loads molecular grammar and fragments from files
    2. Analyzes detected molecules to determine their chemical property distribution
    3. Generates and ranks candidate molecules that match the detected distribution
    4. Returns top-ranked candidates most likely to exist in the chemical space

    Args:
        directory_path (str): Path to directory containing grammar files ('grammar_smiles.txt' and 'grammar.txt')
        detected_smiles (list): List of SMILES strings for known/detected molecules
        valid_atoms (str or list, optional): Either 'default' (infer from detectedSmiles) or explicit list of valid atom symbols i.e ['H','C','N','O']

    Returns:
        list: Sorted list of predicted SMILES strings, ranked by likelihood (highest first)
    """
    with open(os.path.join(directory_path,'grammar_smiles.txt'), "r") as f:
        frag_list_smiles = [line.strip() for line in f]

    with open(os.path.join(directory_path,'grammar.txt'), "r") as f:
        frag_list_filtered = [line.strip() for line in f]

    detected_analysis = analyze_smiles_list(detected_smiles)
    rad_per, ion_per, hetero_per = detected_analysis['radical_percentage']/100, detected_analysis['ion_percentage']/100, detected_analysis['non_terminal_heteroatom_percentage']/100
    top_smiles = getTopPredictedMols(detected_smiles, frag_list_smiles, frag_list_filtered, rad_per, ion_per, hetero_per, valid_atoms, directory_path)

    return top_smiles