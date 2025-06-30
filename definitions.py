import torch
from os.path import join as pjoin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#  DIRECTORIES 

HOMOLOGS_PATH = pjoin('data', 'homologs_dict.pickle')
CALIBRATION_HISTOGRAMS = pjoin('data', 'alibration_histograms')

#  VARIANT PROCESSING 

VALID_AA = "ACDEFGHIKLMNPQRSTVWY"
STOP_AA = '_'
N_AA = len(VALID_AA)
AA_SYN = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
          "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
          "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}
AA_SYN_REV = dict((v, k) for k, v in AA_SYN.items())
AA_TO_INDEX_ESM = {'K': 0, 'R': 1, 'H': 2, 'E': 3, 'D': 4, 'N': 5, 'Q': 6, 'T': 7, 'S': 8, 'C': 9, 'G': 10,
                   'A': 11, 'V': 12, 'L': 13, 'I': 14, 'M': 15, 'P': 16, 'Y': 17, 'F': 18, 'W': 19}
MUTATION_REGEX =  rf'p\.(?P<symbol>(?P<orig>[{VALID_AA}]){{1}}(?P<location>[\d]+)(?P<change>[{VALID_AA}]){{1}})'

ESM_AA_ORDER = 'LAGVSERTIDPKQNFYMHWC'
ESM_AA_LOC = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}

#  VARIANT PREDICTION & ESM 

ESM1B_MODEL = 'esm1b_t33_650M_UR50S'
REP_LAYERS = [33]
# ESM1B_MODEL = 'esm1_t6_43M_UR50S'
# REP_LAYERS = [6]
ESM_MAX_LENGTH = 1020
MASK_TOKEN = '<mask>'

HOMOLOGS_THRESHIOLD = 450
# TODO load homologs dict here 

DISORDERED_THRESHOLD = 0.7
OVERLAP_SIZE_LONG_PROTEIN = 250
PATH_TO_BIN_DICT = "data/bins_dict.pkl"

SCORE_MAPPING = {
    'all/homologs_0_to_450/disordered/longer_than_1022': {
        'method': 'wt_marginals_base_wt_score_126',
        'score_attr': lambda mvs: mvs.wt.nadav_base_wt_score.item()
    },
    'all/homologs_0_to_450/disordered/shorter_than_1022': {
        'method': 'masked_marginals_entropy_weighted_llr_score_142',
        'score_attr': lambda mvs: mvs.masked.entropy_weighted_llr_score.item()
    },
    'all/homologs_0_to_450/ordered/longer_than_1022': {
        'method': 'wt_marginals_base_wt_score_200',
        'score_attr': lambda mvs: mvs.wt.nadav_base_wt_score.item()
    },
    'all/homologs_0_to_450/ordered/shorter_than_1022': {
        'method': 'mutant_marginals_entropy_weighted_llr_score_200',
        'score_attr': lambda mvs: mvs.mutante.entropy_weighted_llr_score.item()
    },
    'all/homologs_450_plus/disordered/longer_than_1022': {
        'method': 'wt_not_nadav_marginals_base_wt_score_203',
        'score_attr': lambda mvs: mvs.wt.llr_base_score.item()
    },
    'all/homologs_450_plus/disordered/shorter_than_1022': {
        'method': 'wt_not_nadav_marginals_base_wt_score_77',
        'score_attr': lambda mvs: mvs.wt.llr_base_score.item()
    },
    'all/homologs_450_plus/ordered/longer_than_1022': {
        'method': 'mutant_marginals_entropy_weighted_llr_score_200',
        'score_attr': lambda mvs: mvs.mutante.entropy_weighted_llr_score.item()
    },
    'all/homologs_450_plus/ordered/shorter_than_1022': {
        'method': 'masked_marginals_entropy_weighted_llr_score_200',
        'score_attr': lambda mvs: mvs.masked.entropy_weighted_llr_score.item()
    },
}
