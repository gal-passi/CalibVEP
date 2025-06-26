import pickle
import numpy as np
import argparse
from data_classes import MutationVariantSet
from definitions import DEVICE, ESM1B_MODEL, ESM_MAX_LENGTH, PATH_TO_BIN_DICT, REP_LAYERS, SCORE_MAPPING
from utils import has_homologs, is_disordered
import utils


class Predictor:
    """
    Predictor class for entropy-based variant effect prediction (VEP) with calibration.

    Parameters
    ----------
    sequence : str
        The amino acid sequence of the protein.
    variant : str
        The variant in HGVS or similar notation (e.g., "p.G12A").
    offset: int
        variant location is shift by offset, default is 1 (count starts from 0) 

    """
    
    def __init__(self, sequence: str, variant: str, offset=1):
        """
        Initialize the Predictor object and validate the input.

        Parameters
        ----------
        sequence : str
            Protein sequence.
        variant : str
            Variant description.

        Raises
        ------
        ValueError
            If the input validation fails.
        """
        self.sequence = sequence
        self.aa_mut = utils.process_mutation_name(variant, offset=offset)
        assert sequence[self.aa_mut.mut_idx] == self.aa_mut.wt_aa, "variant does not match provided sequence, check offset"
    
    def _traverse(self) -> tuple[bool, bool, bool]:
        """
        Traverse the input data to infer mutation subgroup properties.

        Returns
        -------
        tuple of bool
            (is_long, has_homologs, is_disordered)
        """
        is_long = len(self.sequence) > ESM_MAX_LENGTH
        has_homologs_seq = has_homologs(self.sequence)
        is_disordered_seq = is_disordered(self.sequence, self.aa_mut.mut_idx)
        return is_long, has_homologs_seq, is_disordered_seq

    def _get_tree_path_key(self) -> str:
        """
        Helper function to classify a mutation to its tree path based on its characteristics.
        
        Args:
            homolog_count: Number of same sequence in cluster
            is_disordered: Boolean indicating if mutation is in disordered region
            sequence_length: Length of the protein sequence
            
        Returns:
            Tree path string that can be used as a key
        """
        is_long, has_homologs, is_disordered_seq = self._traverse()
        # Homolog classification
        if has_homologs:
            homolog_part = "homologs_450_plus"
        else:
            homolog_part = "homologs_0_to_450"
        
        # Disorder classification
        if is_disordered_seq:
            disorder_part = "disordered"
        else:
            disorder_part = "ordered"
        
        # Length classification
        if is_long:
            length_part = "longer_than_1022"
        else:
            length_part = "shorter_than_1022"
        
        return f"all/{homolog_part}/{disorder_part}/{length_part}"
        
    
    # def _calibration_histogram(self, is_long: bool, has_homologs: bool, is_disordered: bool) -> np.ndarray:
    #     """
    #     Returns a calibration histogram based on mutation subgroup characteristics.

    #     Parameters
    #     ----------
    #     is_long : bool
    #         Whether the sequence is considered long.
    #     has_homologs : bool
    #         Whether homologous sequences are available.
    #     is_disordered : bool
    #         Whether the mutated region is predicted to be disordered.

    #     Returns
    #     -------
    #     np.ndarray
    #         A 1D array representing the calibration histogram with shape (n_buckets,).
    #     """
    #     raise NotImplementedError()
    
    def _compute_raw_score(self) -> tuple[str, float]:
        """
        Compute the raw pathogenicity score using ESM model.
        
        Returns
        -------
        tuple of str, float
            (bin_key, raw_score)
        """
        tree_path_key = self._get_tree_path_key()
        model, alphabet = utils.esm_setup(ESM1B_MODEL, device=DEVICE)
        
        mutation_variant_set: MutationVariantSet = utils.run_esm(model, alphabet, self.sequence, self.aa_mut)
        
        score_config = SCORE_MAPPING[tree_path_key]
        bin_key = score_config['method']
        raw_score = score_config['score_attr'](mutation_variant_set)
        
        return bin_key, raw_score


    def _calibrate_score(self, tree_path_key: str, bin_key: str, raw_score: float) -> float:
        """
        Calibrate the raw score using binned statistics.
        
        Parameters
        ----------
        tree_path_key : str
            The tree path identifier
        bin_key : str
            The bin method key
        raw_score : float
            The raw score to calibrate
            
        Returns
        -------
        float
            The calibrated score
        """
        with open(PATH_TO_BIN_DICT, "rb") as f:
            bins_dict = pickle.load(f)
        
        bin_info = bins_dict[tree_path_key][bin_key]
        bin_edges = np.array(bin_info['bin_edges'])
        bin_stats = bin_info['bin_stats']
        
        # Find which bin the score falls into
        bin_index = np.digitize(raw_score, bin_edges) - 1
        
        # Handle edge cases - clamp to valid range
        bin_index = max(0, min(bin_index, len(bin_edges) - 2))
        
        calibrated_score = bin_stats[bin_index]['path_pct']
        return calibrated_score


    def score(self) -> tuple[float, float]:
        """
        Compute raw and calibrated pathogenicity scores.

        Returns
        -------
        tuple of float
            (raw_score, calibrated_score)
        """
        bin_key, raw_score = self._compute_raw_score()
        tree_path_key = self._get_tree_path_key()
        calibrated_score = self._calibrate_score(tree_path_key, bin_key, raw_score)
        
        return raw_score, calibrated_score


def create_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline interface for protein-level analysis of missense variants in familial data"
    )

    parser.add_argument(
            "-seq", "--sequence",
            type=str,
            required=True,
            help="protein sequence",
        )

    parser.add_argument(
                "-mut", "--variant",
                type=str,
                required=True,
                help="mutation must be in format [wt_aa][index][alt_aa] i.e A167M | p.A167M",
            )

    parser.add_argument(
            "--offset",
            type=int,
            default=1,
            help="offset for mutation index default is 1: count start from zero i.e 1-->0 if offset=1",
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    predictor = Predictor(sequence = args.sequence, variant = args.variant, offset = args.offset)
    score_llr, calibrate_score = predictor.score()
    print(f"Tree Path Key: {predictor._get_tree_path_key()}")
    print(f"Score LLR: {score_llr}, Pathogenic Percentage: {calibrate_score:.2f}%")
