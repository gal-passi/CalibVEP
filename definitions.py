import torch
from os.path inmport join as pjoin

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

#  VARIANT PREDICTION & ESM 

ESM1B_MODEL = 'esm1b_t33_650M_UR50S'
ESM_MAX_LENGTH = 1020

# TODO load homologs dict here 
