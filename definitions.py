import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#  VARIANT PROCESSING 
VALID_AA = "ACDEFGHIKLMNPQRSTVWY"
MUTATION_REGEX =  rf'p\.(?P<symbol>(?P<orig>[{VALID_AA}]){{1}}(?P<location>[\d]+)(?P<change>[{VALID_AA}]){{1}})'
