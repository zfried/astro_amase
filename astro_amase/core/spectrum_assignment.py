"""
spectrum_assignment.py

Core classes and functions for iterative spectral line assignment in radio-astronomical observations

This module implements a full pipeline to assign molecular candidates to observed 
spectral lines using an iterative, context-aware scoring approach. The process includes:

1. **Candidate Representation**:
   - `CandidateScore`: Stores both static (fixed) and dynamic (iteration-dependent) 
     information about each molecular candidate, including frequencies, intensities, 
     structural relevance, and penalties.

2. **Spectral Line Representation**:
   - `SpectralLine`: Represents a single observed spectral line and its candidate molecules.
   - Performs static checks (invalid atoms, vibrational excitation, missing strong lines, 
     frequency match) once.
   - Rescores candidates dynamically as the set of detected molecules grows.
   - Calculates softmax and combined scores to account for relative likelihoods.
   - Determines assignment status: ASSIGNED, MULTIPLE_CARRIERS, or UNIDENTIFIED.

3. **Scoring Context**:
   - `ScoringContext`: Tracks global and local scoring thresholds, temperature, RMS, 
     detected molecules, and structural relevance information.
   - Provides helper functions for intensity checks, vibrational checks, and structural scoring.

4. **Iterative Assignment**:
   - `IterativeSpectrumAssignment`: Manages the main iterative loop over all spectral lines.
   - Performs initial static checks.
   - Scores and assigns lines sequentially.
   - Updates the list of detected molecules dynamically.
   - Recalculates structural relevance scores using VICGAE embeddings when enough new molecules are detected.
   - Rescores all previous lines when structural scores are updated.
   - Handles override rules for molecules consistently flagged as only structurally relevant bandut not assigned.
   - Includes a final convergence loop to ensure assignment stability.

5. **Outputs and Statistics**:
   - Assignment results per line, top candidate molecules, and combined scores.
   - Global statistics including number of assigned, multiple-carrier, and unidentified lines.
   - List of detected molecules with highest intensities.

"""


from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import numpy as np
from collections import defaultdict, Counter


class AssignmentStatus(Enum):
    """Status categories for line assignments."""
    ASSIGNED = "assigned"
    MULTIPLE_CARRIERS = "multiple_carriers"
    UNIDENTIFIED = "unidentified"


@dataclass
class CandidateScore:
    """Stores all scoring information for a single molecular candidate."""
    
    # Molecular identity
    smiles: str
    formula: str
    isotope: int
    quantum_number: str
    catalog_tag: int
    linelist: str
    
    # Raw data (NEVER CHANGES)
    catalog_frequency: float
    observed_frequency: float
    observed_intensity: float
    simulated_intensity: float
    max_simulated_intensity: float
    intensity_scale_factor: float
    molecule_rank: int
    
    # Cached intensity check results (NEVER CHANGES)
    sorted_frequencies: np.ndarray = field(repr=False)
    sorted_intensities: np.ndarray = field(repr=False)
    
    # Cached quality checks (CALCULATED ONCE, NEVER CHANGES)
    has_invalid_atoms: bool = False
    is_vibrationally_excited: bool = False
    rule_out_strong_lines: bool = False
    frequency_match_score: float = 0.0  # Also never changes
    
    # Scores (RECALCULATED each iteration)
    structural_score: float = 0.0
    global_score: float = 0.0
    softmax_score: float = 0.0
    combined_score: float = 0.0
    
    # Quality flags (RECALCULATED each iteration based on context)
    penalties: List[str] = field(default_factory=list)
    is_isotopologue_too_strong: bool = False
    has_relative_intensity_mismatch: bool = False
    has_unreasonably_strong_lines: bool = False
    
    @property
    def frequency_offset(self) -> float:
        return self.catalog_frequency - self.observed_frequency
    
    @property
    def passes_global_threshold(self) -> bool:
        """Check if global score exceeds ASSIGNMENT threshold (93.5)."""
        return self.global_score >= 93.5
    
    def passes_detection_threshold(self, iteration: int) -> bool:
        """
        Check if global score exceeds DETECTION threshold.
        Before iteration 100: 93.5
        After iteration 100: 97.0
        """
        threshold = 97.0 if iteration >= 100 else 93.5
        return self.global_score >= threshold
    
    def reset_dynamic_scores(self):
        """Reset only the scores that change with context."""
        self.structural_score = 0.0
        self.global_score = 0.0
        self.softmax_score = 0.0
        self.combined_score = 0.0
        self.penalties = []
        self.is_isotopologue_too_strong = False
        self.has_relative_intensity_mismatch = False
        self.has_unreasonably_strong_lines = False


class SpectralLine:
    """Represents a single observed spectral line with all its candidate molecules."""
    
    def __init__(self, frequency: float, intensity: float, 
                 line_index: int, rms: float):
        self.frequency = frequency
        self.intensity = intensity
        self.line_index = line_index
        self.rms = rms
        
        # Candidates
        self.candidates: List[CandidateScore] = []
        self._candidates_by_molecule: Dict[str, List[CandidateScore]] = defaultdict(list)
        
        # Track if static checks have been performed
        self._static_checks_done: bool = False
        
        # Assignment results (RECALCULATED each iteration)
        self.assignment_status: Optional[AssignmentStatus] = None
        self.assigned_molecule: Optional[str] = None
        self.assigned_smiles: Optional[str] = None
        self.best_candidate: Optional[CandidateScore] = None
        self.top_candidates: List[CandidateScore] = []
        
        # Track which iteration this was last scored
        self._last_scored_iteration: int = -1
    
    def add_candidate(self, candidate: CandidateScore):
        """Add a molecular candidate to this line (only done once during initialization)."""
        self.candidates.append(candidate)
        self._candidates_by_molecule[candidate.formula].append(candidate)
    
    def perform_static_checks(self, scoring_context: 'ScoringContext'):
        """
        Perform one-time checks that never change.
        This is called once after all candidates are added.
        """
        if self._static_checks_done:
            return
        
        for candidate in self.candidates:
            # Check 1: Invalid atoms (never changes)
            candidate.has_invalid_atoms = scoring_context.has_invalid_atoms(candidate.smiles)
            
            # Check 2: Vibrational excitation (never changes)
            candidate.is_vibrationally_excited = scoring_context.has_vibrational(candidate.formula)
            
            #freqs_molecule = scoring_context.splatDict[(candidate.formula, candidate.linelist)][0]
            #idx = np.argmin(np.abs(freqs_molecule - candidate.catalog_frequency))
            #closest_freq = freqs_molecule[idx]  # ← Use this instead of catalog_frequency
            
            # Check 3: Missing strong lines (with correct closest_freq)
            candidate.rule_out_strong_lines = scoring_context.check_strong_lines_missing(
                candidate.formula,
                candidate.sorted_frequencies,
                candidate.sorted_intensities,
                candidate.catalog_frequency
            )
            

            
            # Check 4: Frequency match score (never changes)
            offset = candidate.frequency_offset
            scale_factor = scoring_context.dv_value_freq / 0.13
            candidate.frequency_match_score = max(0, 1 - abs(offset / scale_factor))
        
        self._static_checks_done = True
    
    def rescore_candidates(self, scoring_context: 'ScoringContext'):
        """
        Recalculate dynamic scores for all candidates based on current context.
        This is called whenever detected molecules list changes.
        """
        # Reset dynamic scores only
        for candidate in self.candidates:
            candidate.reset_dynamic_scores()
        
        # Recalculate each candidate's dynamic score
        for candidate in self.candidates:
            self._score_candidate(candidate, scoring_context)
        
        self._last_scored_iteration = scoring_context.iteration
    
    def _score_candidate(self, candidate: CandidateScore, 
                        context: 'ScoringContext'):
        """Calculate dynamic scores for a single candidate."""
        # 1. Structural relevance score (changes with detected molecules)

        candidate.structural_score = context.calculate_structural_score(
            candidate.smiles
        )

        #if candidate.smiles == '[C]=C=C=C=C=C':
        #    print(candidate.smiles)
        #    print(candidate.formula)
        #    print(candidate.structural_score)
        
        # 2. Initial global score (uses pre-calculated frequency_match_score)
        candidate.global_score = candidate.structural_score * candidate.frequency_match_score
        
        # 3. Apply dynamic penalties based on relative intensities
        self._apply_dynamic_penalties(candidate, context)
    
    def _apply_dynamic_penalties(self, candidate: CandidateScore, 
                                context: 'ScoringContext'):
        """
        Apply penalties that depend on current context.
        Static checks (invalid atoms, vibrational, missing lines, frequency match) 
        are already done and stored in the candidate.
        """
        
        # Add penalties for static issues (don't recalculate, just report)
        if candidate.structural_score < context.global_threshold_original:
            candidate.penalties.append('Structural relevance score low.')
        
        if candidate.frequency_match_score * 100 < context.global_threshold_original:
            candidate.penalties.append('Frequency match poor.')
        
        if candidate.has_invalid_atoms:
            candidate.global_score *= 0.5
            candidate.penalties.append('Contains invalid atoms.')
        
        if candidate.is_vibrationally_excited and context.temperature < 50:
            candidate.global_score *= 0.5
            candidate.penalties.append('Vibrationally excited but temperature too low.')
        
        if candidate.rule_out_strong_lines:
            candidate.global_score *= 0.5
            candidate.penalties.append('Strong lines of this molecule missing from spectrum.')
        
        # Dynamic check 1: Unreasonably strong lines. Checks if max simulated intensity of molecule is
        # greater than 5 times strength of strongest observed lines
        if candidate.max_simulated_intensity > 5 * context.max_observed_intensity:
            candidate.global_score *= 0.5
            candidate.has_unreasonably_strong_lines = True
            candidate.penalties.append('Unreasonably strong lines of this molecule predicted.')
        
        # Dynamic check 2: Relative intensity mismatch.
        if candidate.formula in context.highest_intensities: #if molecule was previously assigned
            if candidate.max_simulated_intensity > 12 * context.highest_intensities[candidate.formula]: #is the max simulated intensity much stronger than what was previously assigned
                candidate.global_score *= 0.5
                candidate.has_relative_intensity_mismatch = True
                candidate.penalties.append('Simulated relative intensities inconsistent with that is observed.')
        else:
            if candidate.max_simulated_intensity > 10 * self.intensity: #is the maximum simulated line too strong (since this molecule has yet to be assigned)
                candidate.global_score *= 0.5
                candidate.has_relative_intensity_mismatch = True
                candidate.penalties.append('Strongest observed line of this molecule is too weak.')
        
        # Dynamic check 3: Isotopologue check (depends on detected_smiles)
        if candidate.isotope != 0 and candidate.smiles not in context.ignore_isotopologues:
            if candidate.smiles not in context.detected_smiles:
                candidate.global_score *= 0.5
                candidate.is_isotopologue_too_strong = True
                candidate.penalties.append('Isotopologue too strong.')
    
    def calculate_softmax_and_combined(self):
        """Calculate softmax and combined scores after all global scores set."""
        from ..utils.astro_utils import softmax
        
        if not self.candidates:
            return
        
        # Calculate softmax
        scores = [c.global_score for c in self.candidates]
        softmax_scores = softmax(scores) #softmax function on scores from single line
        
        for candidate, soft_score in zip(self.candidates, softmax_scores):
            candidate.softmax_score = soft_score
        
        # Calculate combined scores (adds the softmax score if the same molecule has several canddiates for this line)
        # This ensures that a molecule is not competing with itself.
        combined_scores = defaultdict(float)
        for candidate in self.candidates:
            key = (candidate.smiles, candidate.formula)
            combined_scores[key] += candidate.softmax_score
        
        for candidate in self.candidates:
            key = (candidate.smiles, candidate.formula)
            candidate.combined_score = combined_scores[key]
    
    def assign(self, local_threshold: float):
        """
        Determine assignment status based on current scores.
        Uses ORIGINAL global threshold (93.5) for assignment.
        """
        if not self.candidates: #if no candidates
            self.assignment_status = AssignmentStatus.UNIDENTIFIED
            return
        
        # Sort by global score
        self.candidates.sort(key=lambda c: c.global_score, reverse=True)
        self.best_candidate = self.candidates[0] #top candidate has the best global score
        
        # Get top unique molecules by combined score
        self.top_candidates = self._get_top_by_combined_score(n=5)
        
        best_global = self.best_candidate.global_score
        best_combined = self.best_candidate.combined_score
        
        # Use ORIGINAL threshold (93.5) for assignment
        if best_global < 93.5: #if global score below assignment threshold, line is unassigned
            self.assignment_status = AssignmentStatus.UNIDENTIFIED
        elif best_combined < local_threshold: #if global score above assignment threshold, but combined score too low, it has multiple carriers
            # Check for multiple carriers
            passing = [c for c in self.candidates if c.global_score >= 93.5]
            unique_mols = len(set((c.smiles, c.formula) for c in passing))
            
            if unique_mols > 1:
                self.assignment_status = AssignmentStatus.MULTIPLE_CARRIERS
            else:
                self.assignment_status = AssignmentStatus.ASSIGNED
                self.assigned_molecule = self.best_candidate.formula
                self.assigned_smiles = self.best_candidate.smiles
        else: #otherwise just assign the top scored molecule since only one candidate surpasses threshold scores.
            self.assignment_status = AssignmentStatus.ASSIGNED
            self.assigned_molecule = self.best_candidate.formula
            self.assigned_smiles = self.best_candidate.smiles
    
    def _get_top_by_combined_score(self, n: int = 5) -> List[CandidateScore]:
        """Get top N unique molecules by combined score."""
        seen = set()
        unique = []
        
        for candidate in sorted(self.candidates, 
                               key=lambda c: c.combined_score, 
                               reverse=True):
            key = (candidate.smiles, candidate.formula)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
                if len(unique) >= n:
                    break
        
        return unique
    
    def get_override_candidates(self) -> List[str]:
        """Get molecules with only structural relevance issues.
        The goal here is to find molecules that are a sufficient spectroscopic match
        but are only ruled out due to structural relevance. 
        In these cases, the structural relevance score can be overriden if there is
        compelling enough spectroscopic evidence.
        """
        override_mols = []
        
        for candidate in self.candidates:
            if (len(candidate.penalties) == 1 and 
                'Structural' in candidate.penalties[0] and
                candidate.formula not in override_mols):
                override_mols.append(candidate.formula)
        
        return override_mols


@dataclass
class ScoringContext:
    """
    Context information needed for scoring candidates.
    """
    # Fixed parameters (never change)
    dv_value_freq: float
    temperature: float
    rms: float
    max_observed_intensity: float
    global_threshold_original: float = 93.5  # Original threshold for assignment
    local_threshold: float = 0.7
    valid_atoms: List[str] = field(default_factory=list)
    ignore_isotopologues: List[str] = field(default_factory=list)
    peak_freqs_full: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Known molecules that must always be present in detected_smiles
    known_molecules: List[str] = field(default_factory=list)
    
    # Dynamic state (changes as algorithm progresses)
    detected_smiles: List[str] = field(default_factory=list)
    highest_intensities: Dict[str, float] = field(default_factory=dict)
    highest_smiles: Dict[str, float] = field(default_factory=dict)
    
    # Minimum required counts for each known molecule (computed from known_molecules)
    _known_molecule_min_counts: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        """
        Initialize detected_smiles with known molecules and compute minimum counts.
        
        The known_molecules list can contain duplicates, e.g.:
        ['C=O', 'C=O', 'CC(=O)O'] means we need at least 2 copies of 'C=O'
        and at least 1 copy of 'CC(=O)O' in detected_smiles at all times.
        """
        # Count how many times each molecule appears in known_molecules
        from collections import Counter
        counts = Counter(self.known_molecules)
        self._known_molecule_min_counts = dict(counts)
        
        # Add known molecules to detected_smiles at initialization
        # (includes all duplicates from the input list)
        self.detected_smiles.extend(self.known_molecules)
    
    # Structural relevance data
    total_smiles: List[str] = field(default_factory=list)
    structural_percentiles: List[float] = field(default_factory=list)
    
    # Tracking
    iteration: int = 0
    current_line_index: int = 0  # Track which line we're processing
    
    def calculate_structural_score(self, smiles: str) -> float:
        """Get structural relevance score for a SMILES string."""
        if len(self.detected_smiles) == 0:
            return 100.0
        
        if smiles not in self.total_smiles: #if for some reason smiles is not findable, return 0
            return 0.0
        
        idx = self.total_smiles.index(smiles)
        
        # If we have detected molecules but no structural scores calculated yet,
        # this means we need to calculate them for the first time
        # This happens when known_molecules are provided
        if len(self.structural_percentiles) == 0:
            # We have detected molecules (from known_molecules) but haven't calculated
            # structural scores yet - return default score
            # The scores will be calculated on the first recalculation trigger
            return 100.0
        
        # Safety check: ensure index is valid
        if idx >= len(self.structural_percentiles):
            raise IndexError(
                f"Structural percentiles index out of range: "
                f"trying to access index {idx} but structural_percentiles "
                f"only has {len(self.structural_percentiles)} entries. "
                f"SMILES: {smiles}, "
                f"total_smiles has {len(self.total_smiles)} entries, "
                f"detected_smiles has {len(self.detected_smiles)} entries "
                f"({len(set(self.detected_smiles))} unique)"
            )
        
        return 100 * float(self.structural_percentiles[idx])
    
    def update_structural_scores(self, percentiles: List[float]):
        """Update structural relevance scores after recalculation."""
        self.structural_percentiles = percentiles
        self.iteration += 1
    
    def check_strong_lines_missing(self, formula: str, 
                                   sorted_freqs: np.ndarray,
                                   sorted_ints: np.ndarray,
                                   closest_freq: float) -> bool:
        """Check if predicted strong lines are missing from spectrum."""
        from ..utils.astro_utils import checkAllLines
        return checkAllLines(
            formula, self.rms, sorted_freqs, sorted_ints,
            self.dv_value_freq, closest_freq, self.peak_freqs_full
        )
    
    def has_vibrational(self, formula: str) -> bool:
        """Check if formula indicates vibrational excitation."""
        from ..utils.astro_utils import hasVib
        return hasVib(formula)
    
    def has_invalid_atoms(self, smiles: str) -> bool:
        """Check if molecule contains invalid atoms."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in self.valid_atoms:
                return True
        return False
    
    def add_detected_molecule(self, smiles: str, formula: str, intensity: float):
        """Add a newly detected molecule to context."""
        if smiles not in self.detected_smiles:
            self.detected_smiles.append(smiles)
        '''
        if formula not in self.highest_intensities:
            self.highest_intensities[formula] = intensity
        else:
            self.highest_intensities[formula] = max(
                self.highest_intensities[formula], intensity
            )
        
        if smiles not in self.highest_smiles:
            self.highest_smiles[smiles] = intensity
        '''
    
    def ensure_known_molecules_present(self):
        """
        Ensure each known molecule appears at least the minimum required number 
        of times in detected_smiles.
        
        The minimum count for each molecule is determined by how many times it 
        appeared in the original known_molecules list.
        
        Example:
            If known_molecules = ['C=O', 'C=O', 'CC(=O)O'], then:
            - 'C=O' must appear at least 2 times
            - 'CC(=O)O' must appear at least 1 time
        
        This method adds missing copies as needed to meet the minimum counts.
        """
        for smiles, min_count in self._known_molecule_min_counts.items():
            current_count = self.detected_smiles.count(smiles)
            
            if current_count < min_count:
                # Add the missing copies
                copies_to_add = min_count - current_count
                for _ in range(copies_to_add):
                    self.detected_smiles.append(smiles)


class IterativeSpectrumAssignment:
    """
    Manages iterative assignment with re-scoring when new molecules detected.
    """
    
    def __init__(self, frequencies: np.ndarray, intensities: np.ndarray,
                 context: ScoringContext):
        self.context = context
        
        # Create all spectral lines
        self.lines: List[SpectralLine] = [
            SpectralLine(freq, inten, idx, context.rms)
            for idx, (freq, inten) in enumerate(zip(frequencies, intensities))
        ]
        
        # Override tracking
        self.override_counter: Dict[str, int] = defaultdict(int)
        self.override_indices: Dict[str, List[int]] = defaultdict(list)
        
        # Track how many times each molecule has been added via override
        # This counts: 1st override = add once, 2nd override = add twice total, etc.
        self.add_counts: Dict[str, int] = defaultdict(int)
        
        # NEW: Track maximum count for each SMILES in detected_smiles
        self.max_detected_counts: Dict[str, int] = defaultdict(int)
        
        # Track when structural calculation is needed
        self._molecules_since_last_calc: int = 0
        self._calc_frequency: int = 3  # Recalc every N new molecules
        self._min_mols_for_calc: int = 7  # Always recalc below this threshold
        
        # Track static checks
        self._static_checks_performed: bool = False
        
        # Map formula to SMILES for override handling
        self._formula_to_smiles: Dict[str, str] = {}
    
    def add_candidates_to_line(self, line_index: int, 
                               candidates: List[CandidateScore]):
        """Add candidate molecules to a line (done once during initialization)."""
        line = self.lines[line_index]
        for candidate in candidates:
            line.add_candidate(candidate)
            # Build formula to SMILES mapping
            if candidate.formula not in self._formula_to_smiles:
                self._formula_to_smiles[candidate.formula] = candidate.smiles
    
    def perform_all_static_checks(self):
        """
        Perform static checks on all candidates for all lines.
        This is done once after all candidates are loaded.
        """
        if self._static_checks_performed:
            return
        
        print("Performing one-time static checks on all candidates...")
        from ..utils.astro_utils import printProgressBar
        
        for i, line in enumerate(self.lines):
            line.perform_static_checks(self.context)
            #if i % 10 == 0:
            #    printProgressBar(i + 1, len(self.lines), 
            #                   prefix='Static checks:', 
            #                   suffix='Complete', length=50)
        
        #printProgressBar(len(self.lines), len(self.lines), 
        #               prefix='Static checks:', 
        #               suffix='Complete', length=50)
        #print()
        
        self._static_checks_performed = True
    
    def should_recalculate_structural_scores(self) -> bool:
        """Determine if structural relevance should be recalculated."""
        if len(self.context.detected_smiles) == 0:
            return False
        
        unique_detected = len(set(self.context.detected_smiles))
        
        if unique_detected < self._min_mols_for_calc:
            return self._molecules_since_last_calc > 0
        else:
            return self._molecules_since_last_calc >= self._calc_frequency
    
    def recalculate_structural_scores(self):
        """Run VICGAE structural relevance calculation."""
        from .structural_relevance import runCalc
        from ..constants import vicgae_model, covParam, span
        
        # Get embeddings for detected molecules (including duplicates for weighting)
        inp = []
        detected_smiles_list = [] 
        #print('recalculating detected smiles')
        #print(self.context.detected_smiles)
        for smiles in self.context.detected_smiles:  # This list can have duplicates
            embed = vicgae_model.embed_smiles(smiles)
            inp.append(embed[0].numpy())
            detected_smiles_list.append(smiles)
        
        # Get embeddings for all candidate molecules
        embed_vectors = []
        for smiles in self.context.total_smiles:
            embed = vicgae_model.embed_smiles(smiles)
            embed_vectors.append(embed[0].numpy())
        
        # Run calculation
        _, percentiles, gaussian_params = runCalc(inp, covParam, embed_vectors, span)
        '''
        print("\n" + "="*80)
        print("DETECTED MOLECULES - ISOLATION SCORES (mScore)")
        print("="*80)
        
        # Extract mScores from gaussian_params (they're the 3rd element of each tuple)
        # Group by unique SMILES to show both individual and summed scores
        smiles_mscores = {}
        for i, (mean, cov, mscore) in enumerate(gaussian_params):
            smiles = detected_smiles_list[i]
            if smiles not in smiles_mscores:
                smiles_mscores[smiles] = []
            smiles_mscores[smiles].append(mscore)
        
        # Print individual entries (for duplicates)
        print(f"\n{'SMILES':<40} {'Individual mScore':>20} {'Count':>10}")
        print("-"*80)
        for i, (mean, cov, mscore) in enumerate(gaussian_params):
            smiles = detected_smiles_list[i]
            print(f"{smiles:<40} {mscore:>20.4f} {smiles_mscores[smiles].count(mscore):>10}")
        
        # Print summary (unique molecules)
        print(f"\n{'SMILES':<40} {'Total mScore':>20} {'Times Added':>15}")
        print("-"*80)
        for smiles, mscores in sorted(smiles_mscores.items(), key=lambda x: sum(x[1]), reverse=True):
            total_mscore = sum(mscores)
            count = len(mscores)
            print(f"{smiles:<40} {total_mscore:>20.4f} {count:>15}")
        
        print("="*80 + "\n")
        '''
        # Extract percentiles for validation vectors (candidate molecules)
        # runCalc returns percentiles for: [random samples (7000*N), validation vectors]
        # We want only the validation vectors part
        num_val = len(embed_vectors)
        
        # Safety check: ensure percentiles array is long enough
        if len(percentiles) < num_val:
            raise ValueError(
                f"Percentiles array too short: got {len(percentiles)}, "
                f"expected at least {num_val} (for {len(self.context.total_smiles)} candidates)"
            )
        
        val_percentiles = percentiles[-num_val:]
        
        # Verify we have the right number of percentiles
        if len(val_percentiles) != len(self.context.total_smiles):
            raise ValueError(
                f"Percentiles mismatch: got {len(val_percentiles)} percentiles "
                f"but have {len(self.context.total_smiles)} candidate molecules"
            )
        
        # Update context
        self.context.update_structural_scores(val_percentiles)
        self._molecules_since_last_calc = 0
    
    def generate_unassigned_analysis(self, output_file='unassigned_molecules_analysis.txt'):
        """
        Generate a report showing only lines that:
        - Have candidates with structural score 0.5+ lower than assigned molecule, OR
        - Have candidates that were NEVER assigned to ANY line AND have structural < 99.5, OR
        - Are UNIDENTIFIED with clean unassigned candidates that have structural relevance < 99.5
        
        Highlights:
        - Molecules that were NEVER assigned to ANY line (only if structural < 99.5)
        - Cases where unassigned molecule has structural score 0.5+ lower than assigned molecule
        
        Args:
            output_file: Path to output file
        """
        with open(output_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("FILTERED CLEAN UNASSIGNED CANDIDATES ANALYSIS\n")
            f.write("="*100 + "\n\n")
            f.write("Only showing lines that:\n")
            f.write("  - Have candidates with structural score 0.5+ lower than assigned molecule, OR\n")
            f.write("  - Have candidates NEVER assigned to ANY line with structural < 99.5, OR\n")
            f.write("  - Are UNIDENTIFIED with clean candidates having structural relevance < 99.5\n\n")
            f.write("SPECIAL HIGHLIGHTS:\n")
            f.write("  *** = Molecule NEVER assigned to ANY line in the dataset\n")
            f.write("  !!! = Structural score 0.5+ lower than assigned molecule\n\n")
            
            total_clean_unassigned = 0
            lines_printed = 0
            
            # First pass: identify molecules that were NEVER assigned to ANY line
            never_assigned_molecules = set()
            
            for line in self.lines:
                for candidate in line.candidates:
                    key = (candidate.smiles, candidate.formula)
                    if key not in never_assigned_molecules:
                        # Check if this molecule was ever assigned to any line
                        was_ever_assigned = False
                        for check_line in self.lines:
                            if check_line.assigned_smiles == candidate.smiles:
                                was_ever_assigned = True
                                break
                        
                        if not was_ever_assigned:
                            never_assigned_molecules.add(key)
            
            # Loop through every line
            for line in self.lines:
                clean_unassigned_on_this_line = []
                highlighted_candidates = []
                
                # Loop through every candidate on this line
                for candidate in line.candidates:
                    # Check if this candidate has intensity problems or invalid atoms
                    has_problems = (
                        candidate.rule_out_strong_lines or
                        candidate.has_relative_intensity_mismatch or
                        candidate.has_unreasonably_strong_lines or
                        candidate.is_isotopologue_too_strong or
                        candidate.has_invalid_atoms
                    )
                    
                    # Check if this candidate was NOT assigned to this line
                    is_not_assigned_here = (line.assigned_smiles != candidate.smiles)
                    
                    # If no problems AND not assigned to this line, include it
                    if not has_problems and is_not_assigned_here:
                        clean_unassigned_on_this_line.append(candidate)
                        
                        # Check if this candidate should be highlighted
                        should_highlight = False
                        
                        # Check if never assigned to any line AND structural < 99.5
                        if ((candidate.smiles, candidate.formula) in never_assigned_molecules and 
                            candidate.structural_score < 99.5):
                            should_highlight = True
                        
                        # Check structural score difference if line is assigned
                        if line.assignment_status == AssignmentStatus.ASSIGNED and line.best_candidate:
                            struct_diff = line.best_candidate.structural_score - candidate.structural_score
                            if struct_diff > 0.5:
                                should_highlight = True
                        
                        # For multiple carriers, check against all top candidates
                        elif line.assignment_status == AssignmentStatus.MULTIPLE_CARRIERS and line.top_candidates:
                            for top_cand in line.top_candidates:
                                struct_diff = top_cand.structural_score - candidate.structural_score
                                if struct_diff > 0.5:
                                    should_highlight = True
                                    break
                        
                        if should_highlight:
                            highlighted_candidates.append(candidate)
                
                # Determine if we should print this line
                should_print_line = False
                
                # Case 1: Has highlighted candidates
                if len(highlighted_candidates) > 0:
                    should_print_line = True
                
                # Case 2: Unidentified with clean candidates having structural < 99.5
                elif line.assignment_status == AssignmentStatus.UNIDENTIFIED and len(clean_unassigned_on_this_line) > 0:
                    # Check if any clean candidate has structural relevance < 99.5
                    for candidate in clean_unassigned_on_this_line:
                        if candidate.structural_score < 99.5:
                            should_print_line = True
                            break
                
                if should_print_line:
                    lines_printed += 1
                    f.write("\n" + "="*100 + "\n")
                    f.write(f"LINE {line.line_index} (Frequency: {line.frequency:.4f} MHz, Intensity: {line.intensity:.4f})\n")
                    f.write("="*100 + "\n")
                    
                    # Print all clean unassigned candidates
                    f.write(f"\nClean unassigned candidates ({len(clean_unassigned_on_this_line)}):\n")
                    f.write("-"*100 + "\n")
                    
                    for candidate in clean_unassigned_on_this_line:
                        # Check for highlighting conditions
                        never_assigned_flag = ""
                        struct_diff_flag = ""
                        
                        # Check if never assigned to any line (only show flag if structural < 99.5)
                        if ((candidate.smiles, candidate.formula) in never_assigned_molecules and 
                            candidate.structural_score < 99.5):
                            never_assigned_flag = " *** NEVER ASSIGNED TO ANY LINE ***"
                        
                        # Check structural score difference if line is assigned
                        if line.assignment_status == AssignmentStatus.ASSIGNED and line.best_candidate:
                            struct_diff = line.best_candidate.structural_score - candidate.structural_score
                            if struct_diff > 0.5:
                                struct_diff_flag = f" !!! STRUCTURAL SCORE {struct_diff:.2f} LOWER THAN ASSIGNED !!!"
                        
                        # For multiple carriers, check against all top candidates
                        elif line.assignment_status == AssignmentStatus.MULTIPLE_CARRIERS and line.top_candidates:
                            max_struct_diff = 0
                            for top_cand in line.top_candidates:
                                struct_diff = top_cand.structural_score - candidate.structural_score
                                if struct_diff > max_struct_diff:
                                    max_struct_diff = struct_diff
                            
                            if max_struct_diff > 0.5:
                                struct_diff_flag = f" !!! STRUCTURAL SCORE UP TO {max_struct_diff:.2f} LOWER THAN CARRIERS !!!"
                        
                        f.write(f"\n  Molecule: {candidate.formula}{never_assigned_flag}{struct_diff_flag}\n")
                        f.write(f"  SMILES: {candidate.smiles}\n")
                        f.write(f"    Frequency match score: {candidate.frequency_match_score:.4f}\n")
                        f.write(f"    Structural relevance:  {candidate.structural_score:.4f}\n")
                        if candidate.penalties:
                            f.write(f"    Penalties: {', '.join(candidate.penalties)}\n")
                        f.write("\n")
                    
                    total_clean_unassigned += len(clean_unassigned_on_this_line)
                    
                    # Print what was actually assigned to this line
                    f.write("-"*100 + "\n")
                    if line.assignment_status == AssignmentStatus.ASSIGNED and line.best_candidate:
                        f.write(f"\nAssigned to this line:\n")
                        f.write(f"  Molecule: {line.best_candidate.formula}\n")
                        f.write(f"  SMILES: {line.best_candidate.smiles}\n")
                        f.write(f"    Frequency match score: {line.best_candidate.frequency_match_score:.4f}\n")
                        f.write(f"    Structural relevance:  {line.best_candidate.structural_score:.4f}\n")
                        if line.best_candidate.penalties:
                            f.write(f"    Penalties: {', '.join(line.best_candidate.penalties)}\n")
                    
                    elif line.assignment_status == AssignmentStatus.MULTIPLE_CARRIERS:
                        f.write(f"\nLine marked as MULTIPLE_CARRIERS\n")
                        f.write(f"All carrier species ({len(line.top_candidates)}):\n\n")
                        
                        for i, carrier in enumerate(line.top_candidates, 1):
                            f.write(f"  Carrier {i}: {carrier.formula}\n")
                            f.write(f"  SMILES: {carrier.smiles}\n")
                            f.write(f"    Frequency match score: {carrier.frequency_match_score:.4f}\n")
                            f.write(f"    Structural relevance:  {carrier.structural_score:.4f}\n")
                            if carrier.penalties:
                                f.write(f"    Penalties: {', '.join(carrier.penalties)}\n")
                            f.write("\n")
                    
                    elif line.assignment_status == AssignmentStatus.UNIDENTIFIED:
                        f.write(f"\nLine is UNIDENTIFIED (no molecule assigned)\n")
                    
                    else:
                        f.write(f"\nLine status: unknown\n")
                    
                    f.write("\n")
            
            # Summary at the end
            f.write("\n" + "="*100 + "\n")
            f.write("SUMMARY\n")
            f.write("="*100 + "\n")
            f.write(f"Lines printed: {lines_printed}\n")
            f.write(f"Total clean unassigned candidates shown: {total_clean_unassigned}\n")
            
            # List molecules never assigned with structural < 99.5
            never_assigned_low_struct = []
            for smiles, formula in never_assigned_molecules:
                # Find structural score for this molecule
                for line in self.lines:
                    for candidate in line.candidates:
                        if candidate.smiles == smiles and candidate.structural_score < 99.5:
                            never_assigned_low_struct.append((smiles, formula, candidate.structural_score))
                            break
                    if any(s == smiles for s, f, _ in never_assigned_low_struct):
                        break
            
            f.write(f"Molecules never assigned with structural < 99.5: {len(never_assigned_low_struct)}\n\n")
            
            # List molecules never assigned
            if never_assigned_low_struct:
                f.write("Molecules never assigned to any line (with structural < 99.5):\n")
                for smiles, formula, struct_score in sorted(never_assigned_low_struct, key=lambda x: x[2]):
                    f.write(f"  - {formula} ({smiles}): Structural = {struct_score:.2f}\n")
        
        print(f"\nFiltered clean unassigned candidates analysis saved to: {output_file}")
                    
    def score_and_assign_line(self, line_index: int) -> Tuple[bool, bool]:
        """
        Score and assign a single line.
        
        Returns:
            (assigned, should_add_to_detected): 
                - assigned: True if line was assigned to a molecule
                - should_add_to_detected: True if molecule should be added to detected list
        """
        line = self.lines[line_index]
        self.context.current_line_index = line_index
        
        # Rescore candidates with current context
        line.rescore_candidates(self.context)
        
        # Calculate softmax and combined scores
        line.calculate_softmax_and_combined()

        structural_scores = [c.structural_score for c in line.candidates]
        cans = [c.formula for c in line.candidates]
        '''
        if structural_scores:
            avg_structural = np.mean(structural_scores)
            print(f"\nLine {line_index} (freq={line.frequency:.4f} MHz):")
            print(f"  Average structural score: {avg_structural:.4f}")
            print(f"  Candidates with structural scores: {len(structural_scores)}/{len(line.candidates)}")
            printList = [(structural_scores[i],cans[i]) for i in range(len(structural_scores))]
            print(printList)

            c6h2_candidates = [c for c in line.candidates if c.formula == 'l-C6H2']
            if c6h2_candidates:
                print("\n" + "="*60)
                print(f"  *** C6H2 FOUND IN LINE {line_index} ***")
                print("="*60)
                for c6h2 in c6h2_candidates:
                    print(f"  Formula: {c6h2.formula}")
                    print(f"  SMILES: {c6h2.smiles}")
                    print(f"  Structural score: {c6h2.structural_score:.4f}")
                    print(f"  Global score: {c6h2.global_score:.2f}")
                    print(f"  Combined score: {c6h2.combined_score:.4f}")
                    print(f"  Softmax score: {c6h2.softmax_score:.4f}")
                    print(f"  Passes threshold: {c6h2.passes_global_threshold}")
                    if c6h2.penalties:
                        print(f"  Penalties: {', '.join(c6h2.penalties)}")
                    print("-"*60)
                print()

        '''
        # Assign (uses original 93.5 threshold)
        line.assign(self.context.local_threshold)
        
        # Check if assigned and if should be added to detected
        assigned = line.assignment_status == AssignmentStatus.ASSIGNED
        should_add_to_detected = False
        
        if assigned:
            # Check if this is a new molecule (not already in detected list)
            if line.assigned_smiles not in self.context.detected_smiles:
                # Determine threshold based on iteration
                if line_index < 100:
                    # Before line 100, use original threshold (93.5)
                    should_add_to_detected = line.best_candidate.global_score >= 93.5
                else:
                    # After line 100, require higher threshold (97.0) for detection
                    should_add_to_detected = line.best_candidate.global_score >= 97.0
                
                if should_add_to_detected:
                    self.context.add_detected_molecule(
                        line.assigned_smiles,
                        line.assigned_molecule,
                        line.intensity
                    )
                    self._molecules_since_last_calc += 1
        
        # Update highest intensities for all high-scoring candidates (regardless of threshold)
        for candidate in line.candidates:
            if candidate.global_score >= 93.5:  # Original threshold
                if candidate.formula not in self.context.highest_intensities:
                    self.context.highest_intensities[candidate.formula] = line.intensity
                else: #store the highest intensity that is reasonable for this molecule
                    self.context.highest_intensities[candidate.formula] = max(
                        self.context.highest_intensities[candidate.formula],
                        line.intensity
                    )
                
                if candidate.smiles not in self.context.highest_smiles:
                    self.context.highest_smiles[candidate.smiles] = line.intensity
                else: #store the highest intensity that is reasonable for this molecule
                    self.context.highest_smiles[candidate.smiles] = max(
                        self.context.highest_smiles[candidate.smiles],
                        line.intensity
                    )
        #print('highest')
        #print(self.context.highest_intensities)
        #print(self.context.highest_smiles)
        return assigned, should_add_to_detected
    
    def _handle_override_in_main_loop(self, old_detected_smiles: List[str]):
        """
        Handle override molecules during main assignment loop.
        If a molecule is overridden and was already in detected list,
        add it additional times based on add_counts.
        Ensures count never drops below historical maximum.
        """
        # Get current override molecules (≥3 occurrences)
        override_mols = self._check_overrides_without_adding(min_occurrences=2)

        for formula in override_mols:
            if formula not in self._formula_to_smiles:
                continue
            
            smiles = self._formula_to_smiles[formula]
            
            # Update highest_intensities
            if formula not in self.context.highest_intensities:
                first_idx = self.override_indices[formula][0]
                self.context.highest_intensities[formula] = self.lines[first_idx].intensity
        
            if smiles not in self.context.highest_smiles:
                first_idx = self.override_indices[formula][0]
                self.context.highest_smiles[smiles] = self.lines[first_idx].intensity
            
            # Check if this molecule was already in the OLD detected list
            if smiles in old_detected_smiles:
                # Already detected - add more copies
                num_add = self.add_counts[formula] + 1
                self.add_counts[formula] = num_add
                
                for _ in range(num_add):
                    self.context.detected_smiles.append(smiles)
                    self._molecules_since_last_calc += 1
            else:
                # First time override - add it once
                self.context.detected_smiles.append(smiles)
                self._molecules_since_last_calc += 1
                self.add_counts[formula] = 1
            
            # Update the maximum count for this SMILES
            current_count = self.context.detected_smiles.count(smiles)
            if current_count > self.max_detected_counts[smiles]:
                self.max_detected_counts[smiles] = current_count
            
            # Ensure count meets the historical maximum
            max_count = self.max_detected_counts[smiles]
            current_count = self.context.detected_smiles.count(smiles)
            
            if current_count < max_count:
                supplement_needed = max_count - current_count
                #print(f"Supplementing {smiles} ({formula}): adding {supplement_needed} more to reach max of {max_count}")
                for _ in range(supplement_needed):
                    self.context.detected_smiles.append(smiles)
                    self._molecules_since_last_calc += 1
        
    def _check_overrides_without_adding(self, min_occurrences: int = 2) -> List[str]:
        """
        Check for molecules that meet override criteria.
        Returns list of formula strings that have ≥min_occurrences lines
        with only structural relevance penalty.
        Does NOT modify detected list.
        """
        # Recalculate override counter based on current assignments
        self.override_counter.clear()
        self.override_indices.clear()
        
        for line in self.lines:
            override_mols = line.get_override_candidates()
            for mol in override_mols:
                self.override_counter[mol] += 1
                self.override_indices[mol].append(line.line_index)
        
        # Return molecules meeting threshold
        override_mols = []
        for mol, count in self.override_counter.items():
            if count >= min_occurrences:
                override_mols.append(mol)
        #print('override mols')
        #print(override_mols)
        return override_mols
    
    def assign_all_iteratively(self):
        """
        Main assignment loop with iterative refinement.
        """
        from ..utils.astro_utils import printProgressBar
        
        # Perform static checks once
        self.perform_all_static_checks()
        
        # If we have known molecules, calculate initial structural scores
        # This ensures structural relevance is considered from the very first line
        if len(self.context.detected_smiles) > 0 and len(self.context.structural_percentiles) == 0:
            print(f"Calculating initial structural scores based on {len(self.context.detected_smiles)} known molecules "
                  f"({len(set(self.context.detected_smiles))} unique)...")
            self.recalculate_structural_scores()
    
        print("Starting iterative line assignment...")
        printProgressBar(0, len(self.lines), prefix='Progress:', 
                        suffix='Complete', length=50)
        
        for i in range(len(self.lines)):
            #print('current detected')
            #print(self.context.detected_smiles)
            # Score and assign current line
            assigned, added_to_detected = self.score_and_assign_line(i)
            
            # Check if we need to recalculate structural scores
            if added_to_detected and self.should_recalculate_structural_scores():
                #print(f"\nRecalculating structural scores at line {i+1} "
                      #f"({len(set(self.context.detected_smiles))} unique molecules detected, "
                      #f"{len(self.context.detected_smiles)} total including duplicates)...")
                
                # Store old detected list for override handling
                old_detected = self.context.detected_smiles.copy()
                
                # Recalculate structural relevance
                self.recalculate_structural_scores()
                
                # Rescore ALL previous lines with new structural scores
                #print(f"Rescoring previous {i} lines...")
                for j in range(i):
                    prev_line = self.lines[j]
                    
                    # Rescore candidates (only dynamic penalties recalculated)
                    prev_line.rescore_candidates(self.context)
                    prev_line.calculate_softmax_and_combined()
                    
                    # Re-assign
                    old_assignment = prev_line.assigned_smiles
                    prev_line.assign(self.context.local_threshold)
                    
                    for candidate in prev_line.candidates:
                        if candidate.global_score >= 93.5:
                            if candidate.formula not in self.context.highest_intensities:
                                self.context.highest_intensities[candidate.formula] = prev_line.intensity
                            else:
                                self.context.highest_intensities[candidate.formula] = max(
                                    self.context.highest_intensities[candidate.formula],
                                    prev_line.intensity
                                )
                            
                            if candidate.smiles not in self.context.highest_smiles:
                                self.context.highest_smiles[candidate.smiles] = prev_line.intensity
                            else:
                                self.context.highest_smiles[candidate.smiles] = max(
                                    self.context.highest_smiles[candidate.smiles],
                                    prev_line.intensity
                                )
                    # Check if assignment changed
                    if prev_line.assignment_status == AssignmentStatus.ASSIGNED:
                        if prev_line.assigned_smiles != old_assignment:
                            # Assignment changed
                            if prev_line.assigned_smiles not in self.context.detected_smiles:
                                # Determine if should add based on iteration
                                if j < 100:
                                    should_add = prev_line.best_candidate.global_score >= 93.5
                                else:
                                    should_add = prev_line.best_candidate.global_score >= 97.0
                                
                                if should_add:
                                    self.context.add_detected_molecule(
                                        prev_line.assigned_smiles,
                                        prev_line.assigned_molecule,
                                        prev_line.intensity
                                    )
                                    self._molecules_since_last_calc += 1
                
                # Handle override molecules with multiple additions
                self._handle_override_in_main_loop(old_detected)
                
                # Ensure known molecules remain present after rescoring
                self.context.ensure_known_molecules_present()
            
            # Update override counter for tracking (always do this)
            override_mols = self.lines[i].get_override_candidates()
            #print('override_counter')
            #print(self.override_counter)
            for mol in override_mols:
                self.override_counter[mol] += 1
                self.override_indices[mol].append(i)
                #print(self.override_counter)
            
            # Update progress
            printProgressBar(i + 1, len(self.lines), prefix='Progress:', 
                           suffix='Complete', length=50)
            #print('updated detected')
            #print(self.context.detected_smiles)
        print("\nPerforming final convergence check...")
        self._final_convergence()
    
    def _final_convergence(self):
        """
        Final iterative refinement until convergence.
        Each iteration:
        1. Recalculate structural scores
        2. Rescore and reassign all lines
        3. Check for override molecules
        4. For override molecules already in detected list, add them one more time
        5. Repeat until no changes occur
        """
        converged = False
        iteration_count = 0
        max_iterations = 50

        minNeededFinal = {}
        
        while not converged and iteration_count < max_iterations:
            iteration_count += 1
            print(f"Final iteration {iteration_count}...")
            # Store state before this iteration
            old_detected = self.context.detected_smiles.copy()
            #print('old detected')
            #print(old_detected)
            old_detected_counter = Counter(old_detected)
            
            # Recalculate structural scores if we have detected molecules
            if len(self.context.detected_smiles) > 0:
                self.recalculate_structural_scores()
            
            # Reset detected list (will rebuild it)
            self.context.detected_smiles = []
            
            # Rescore and reassign all lines
            for idx, line in enumerate(self.lines):
                self.context.current_line_index = idx
                
                line.rescore_candidates(self.context)
                line.calculate_softmax_and_combined()
                line.assign(self.context.local_threshold)
                
                # Updating highest intensity lists
                for candidate in line.candidates:
                    if candidate.global_score >= 93.5:
                        if candidate.formula not in self.context.highest_intensities:
                            self.context.highest_intensities[candidate.formula] = line.intensity
                        else:
                            self.context.highest_intensities[candidate.formula] = max(
                                self.context.highest_intensities[candidate.formula],
                                line.intensity
                            )
                        
                        if candidate.smiles not in self.context.highest_smiles:
                            self.context.highest_smiles[candidate.smiles] = line.intensity
                        else:
                            self.context.highest_smiles[candidate.smiles] = max(
                                self.context.highest_smiles[candidate.smiles], line.intensity)
                                

                # Add to detected if assigned with high enough score
                if line.assignment_status == AssignmentStatus.ASSIGNED:
                    if line.assigned_smiles not in self.context.detected_smiles:
                        # Use 97.0 threshold for all lines in final convergence
                        if line.best_candidate.global_score >= 97.0:
                            self.context.add_detected_molecule(
                                line.assigned_smiles,
                                line.assigned_molecule,
                                line.intensity
                            )
            
            # Ensure known molecules are always present after rebuilding
            self.context.ensure_known_molecules_present()
            
            # Check for override molecules
            override_mols = self._check_overrides_without_adding(min_occurrences=2)
            #print('override mols')
            #print(override_mols)
            # Handle each override molecule
            for formula in override_mols:
                if formula not in self._formula_to_smiles:
                    continue
                
                smiles = self._formula_to_smiles[formula]
                
                # Update highest_intensities from first override occurrence
                if formula not in self.context.highest_intensities:
                    first_idx = self.override_indices[formula][0]
                    self.context.highest_intensities[formula] = self.lines[first_idx].intensity
                
                if smiles not in self.context.highest_smiles:
                    first_idx = self.override_indices[formula][0]
                    self.context.highest_smiles[smiles] = self.lines[first_idx].intensity
                #ensuring sufficienct entries for override function
                if smiles not in minNeededFinal:
                    minNeededFinal[smiles] = 1                            
                # Check if this molecule was in the OLD detected list
                if smiles in old_detected:
                    # It was detected before but still shows up in override
                    # This means it's STILL not structurally relevant enough
                    # Count how many times it appeared in old list
                    old_count = old_detected.count(smiles)
                    # Count how many times it appears in new list (from assignments)
                    new_count = self.context.detected_smiles.count(smiles)
                    
                    # Add it one MORE time than before
                    # Formula: add (1 + old_count - new_count) copies
                    # This ensures total count increases by 1 each override cycle
                    count_diff = 1 + old_count - new_count
                    minNeededFinal[smiles] = minNeededFinal[smiles] + count_diff
                    for _ in range(count_diff):
                        self.context.detected_smiles.append(smiles)
                else:
                    # First time in override - add once
                    if smiles not in self.context.detected_smiles:
                        self.context.detected_smiles.append(smiles)   
                # Update highest intensities
                if formula not in self.context.highest_intensities:
                    first_idx = self.override_indices[formula][0]
                    self.context.highest_intensities[formula] = self.lines[first_idx].intensity
                
                if smiles not in self.context.highest_smiles:
                    first_idx = self.override_indices[formula][0]
                    self.context.highest_smiles[smiles] = self.lines[first_idx].intensity
            
            for m in minNeededFinal:
                co = self.context.detected_smiles.count(m)
                if co < minNeededFinal[m]:
                    newAdd = minNeededFinal[m] - co
                    #print('adding',m,newAdd,'times')
                    for _ in range(newAdd):
                        self.context.detected_smiles.append(m)


            #print('new detected smiles')
            #print(self.context.detected_smiles)
            for m in minNeededFinal:
                co = self.context.detected_smiles.count(m)
                if co > 1:
                    minNeededFinal[m] = co
            
            # Ensure known molecules are always present before checking convergence
            self.context.ensure_known_molecules_present()
            
            #print('min needed')
            #print(minNeededFinal)    
            # Check convergence: detected list is stable AND no override molecules
            new_detected_counter = Counter(self.context.detected_smiles)
            num_override = len(override_mols)
            
            #if new_detected_counter == old_detected_counter and num_override == 0:
            if num_override == 0:
                converged = True
                print("Converged!")
            else:
                unique_added = len(set(self.context.detected_smiles) - set(old_detected))
                total_added = len(self.context.detected_smiles) - len(old_detected)
                #print(f"  Unique molecules: {len(set(self.context.detected_smiles))}")
                #print(f"  Total in detected list: {len(self.context.detected_smiles)}")
                #print(f"  Override molecules: {num_override}")
        
        if not converged:
            print(f"Warning: Did not converge after {max_iterations} iterations")
        
        # Final summary
        unique_detected = len(set(self.context.detected_smiles))
        total_detected = len(self.context.detected_smiles)
        #print(f"\nFinal: {unique_detected} unique molecules detected "
        #      f"({total_detected} total including duplicates for structural weighting)")
    
    def statistics(self) -> Dict:
        """Generate assignment statistics."""
        assigned = sum(1 for line in self.lines 
                      if line.assignment_status == AssignmentStatus.ASSIGNED)
        multiple = sum(1 for line in self.lines 
                      if line.assignment_status == AssignmentStatus.MULTIPLE_CARRIERS)
        unidentified = sum(1 for line in self.lines 
                          if line.assignment_status == AssignmentStatus.UNIDENTIFIED)
        
        unique_detected = len(set(self.context.detected_smiles))
        total_detected = len(self.context.detected_smiles)
        

        return {
            'total_lines': len(self.lines),
            'assigned': assigned,
            'multiple_carriers': multiple,
            'unidentified': unidentified,
            'unique_detected_molecules': unique_detected,
            'total_detected_molecules': total_detected,
            'structural_calculations': self.context.iteration,
            'override_molecules': sum(1 for c in self.override_counter.values() 
                                     if c >= 3)
        }
    
    def get_assigned_molecules(self) -> List[Tuple[str, str]]:
        """Get list of (molecule, database) tuples for assigned molecules."""
        assigned = []
        seen = set()
        
        for line in self.lines:
            if (line.assignment_status == AssignmentStatus.ASSIGNED and 
                line.best_candidate):
                key = (line.best_candidate.formula, line.best_candidate.linelist)
                if key not in seen:
                    assigned.append(key)
                    seen.add(key)
        
        return assigned
    
    def get_detected_molecules_summary(self) -> Dict[str, Dict]:
        """
        Get summary of detected molecules including how many times each was added.
        
        Returns:
            Dict mapping SMILES to info dict with:
                - formula: molecular formula
                - count: number of times in detected list
                - max_intensity: highest observed intensity
                - override_added: whether added via override mechanism
        """
        summary = {}
        detected_counter = Counter(self.context.detected_smiles)
        
        for smiles, count in detected_counter.items():
            # Find formula and other info
            formula = None
            for line in self.lines:
                for candidate in line.candidates:
                    if candidate.smiles == smiles:
                        formula = candidate.formula
                        break
                if formula:
                    break
            
            override_added = (formula in self.override_counter and 
                            self.override_counter[formula] >= 3)
            
            summary[smiles] = {
                'formula': formula,
                'count': count,
                'max_intensity': self.context.highest_smiles.get(smiles, 0.0),
                'override_added': override_added,
                'was_multiply_added': count > 1
            }
        
        return summary