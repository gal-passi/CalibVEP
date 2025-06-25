from dataclasses import dataclass
from enum import Enum
from mutation_record import MutationRecord


@dataclass
class AAMut:
    wt_aa: str
    mut_idx: int
    change_aa: str


@dataclass
class MutationVariantSet:
    """This class holds the mutation states for a protein sequence.
    It contains three MutationRecord objects: wt, mutante, and masked.
    """
    wt: MutationRecord = None
    mutante: MutationRecord = None
    masked: MutationRecord = None

    def add_mutation_record(self, mutation_record: MutationRecord, method: 'METHODS_TO_ESM'):
        """Adds a MutationRecord to the appropriate field based on the method.
        This method allows you to add a mutation record for either the wild type (wt), 
        the mutant (mutante), or the masked version (masked) of the protein sequence.

        Args:
            mutation_record (MutationRecord): The mutation record to add.
            method (METHODS_TO_ESM): The method indicating which mutation record to add.

        Raises:
            ValueError: If the method is not one of the defined METHODS_TO_ESM.
        """
        if method == METHODS_TO_ESM.MUTANTE:
            self.mutante = mutation_record
        elif method == METHODS_TO_ESM.MASKED:
            self.masked = mutation_record
        elif method == METHODS_TO_ESM.WT:
            self.wt = mutation_record
        else:
            raise ValueError(f"Unknown method: {method}")


class METHODS_TO_ESM(Enum):
    """
    Enum for mutation names
    """
    MUTANTE = "mutant_marginals"
    MASKED = "masked_marginals"
    WT = "wt_marginals"

    def __eq__(self, value):
        return value == self.value

    @classmethod
    def get_methods(cls):
        """Returns a list of all methods in the enum."""
        return [method for method in cls]
    


class DetailedLabel(Enum):
    BENIGN = 'Benign'
    PATHOGENIC = 'Pathogenic'
    LIKELY_BENIGN = 'Likely_benign'
    LIKELY_PATHOGENIC = 'Likely_pathogenic'
    BENIGN_LIKELY_BENIGN = 'Benign_Likely_benign'
    PATHOGENIC_LIKELY_PATHOGENIC = 'Pathogenic_Likely_pathogenic'
    UNLABELED = 'Unlabeled'

    def is_benign_category(self) -> bool:
        """Returns True if the label is benign or likely benign """
        return self in [DetailedLabel.BENIGN, DetailedLabel.LIKELY_BENIGN, DetailedLabel.BENIGN_LIKELY_BENIGN]
    
    def is_pathogenic_category(self) -> bool:
        """Returns True if the label is pathogenic or likely pathogenic """
        return self in [DetailedLabel.PATHOGENIC, DetailedLabel.LIKELY_PATHOGENIC, DetailedLabel.PATHOGENIC_LIKELY_PATHOGENIC]

