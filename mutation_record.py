from functools import cached_property
from typing import Optional, List, TYPE_CHECKING
import numpy as np
import torch
from definitions import ESM_AA_LOC, ESM_AA_ORDER
import math

# if TYPE_CHECKING:
#     from data_classes import AAMut

class MutationRecord:

    def __init__(self, protein_seq: str, aa_mut, truncated_logits: torch.Tensor):
        self.protein_seq = protein_seq
        self.aa_mut = aa_mut
        self.truncated_logits = truncated_logits

    @staticmethod
    def static_relevant_logits(truncated_logits, aa_mut):
        return truncated_logits[aa_mut.mut_idx] - truncated_logits[aa_mut.mut_idx][ESM_AA_LOC[aa_mut.wt_aa]]

    @cached_property
    def relevant_logits(self):
        # return MutationRecord.static_relevant_logits(self.truncated_logits, self.aa_mut)
        return self.truncated_logits[self.aa_mut.mut_idx] - self.truncated_logits[self.aa_mut.mut_idx][ESM_AA_LOC[self.aa_mut.wt_aa]]

    @staticmethod
    def static_calculate_entropy(relevant_logits):
        probs = torch.softmax(relevant_logits, dim=-1)
        entropy_seqs = torch.distributions.Categorical(probs=probs).entropy()
        return entropy_seqs

    @cached_property
    def entropy_tensor(self):
        probs = torch.softmax(self.relevant_logits, dim=-1)
        entropy_seqs = torch.distributions.Categorical(probs=probs).entropy()
        return entropy_seqs
        # return MutationRecord.static_calculate_entropy(self.relevant_logits)

    @cached_property
    def llr_logits(self):
        return torch.log_softmax(self.relevant_logits, dim=-1)

    @cached_property
    def llr_base_score(self):
        """
        The log-likelihood ratio of the mutant in the position of the mutation
        """
        return self.llr_logits[ESM_AA_LOC[self.aa_mut.change_aa]]

    @cached_property
    def nadav_base_wt_score(self):
        """
        The score of the mutant in the position of the mutation (from the logits)
        """
        log_softmax_truncated = torch.log_softmax(self.truncated_logits[self.aa_mut.mut_idx], dim=-1)
        base_wt_llr = log_softmax_truncated - log_softmax_truncated[ESM_AA_LOC[self.aa_mut.wt_aa]]
        return base_wt_llr[ESM_AA_LOC[self.aa_mut.change_aa]]

    # @cached_property
    # def nadav_base_wt_score(self):
    #     # log_softmax_truncated = torch.log_softmax(self.truncated_logits[self.aa_mut.mut_idx], dim=-1)
    #     # base_wt_llr = log_softmax_truncated - log_softmax_truncated[ESM_AA_LOC[self.aa_mut.wt_aa]]
    #     log_softmax_truncated = torch.log_softmax(self.truncated_logits[self.aa_mut.mut_idx], dim=-1)
    #     base_wt_llr = log_softmax_truncated[ESM_AA_LOC[self.aa_mut.change_aa]] - log_softmax_truncated[ESM_AA_LOC[self.aa_mut.wt_aa]]
    #     return base_wt_llr.item()

        # log_softmax_seq = torch.log_softmax(
        #     self.truncated_logits,
        #     dim=-1
        # )
        #
        # # Extract AA logits and calculate LLR in one go
        # aa_logits = log_softmax_seq[:, aa_indices]
        # wt_diag = aa_logits[torch.arange(len(wt_sequence)), wt_aa_indices]
        # LLR = aa_logits - wt_diag.unsqueeze(-1)

        # results_ = torch.log_softmax(model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)['logits'],
        #                              dim=-1)
        # results = torch.log_softmax(self.truncated_logits, dim=-1)

        # WTlogits = pd.DataFrame(results[0, :, :].cpu().numpy()[1:-1, :], columns=alphabet.all_toks,
        #                         index=list(input_df[input_df.id == gname].seq.values[0])).T.iloc[4:24].loc[AAorder]
        # WTlogits.columns = [j.split('.')[0] + ' ' + str(i + 1) for i, j in enumerate(WTlogits.columns)]
        # wt_norm = np.diag(WTlogits.loc[[i.split(' ')[0] for i in WTlogits.columns]])
        # LLR = WTlogits - wt_norm

    @cached_property
    def normilized_entropy(self):
        """
        The normalized entropy of the mutant in the position of the mutation.
        The normalized entropy is defined as 1 - (entropy / log(20)), what makes it range from 0 to 1.
        1 means that the model is very confident about the prediction, while 0 means that the model is very uncertain.
        """
        return 1 - (self.entropy_tensor / np.log(len(ESM_AA_ORDER)))

    @cached_property
    def entropy_weighted_llr_score(self):
        """
        The entropy weighted log-likelihood ratio of the mutant in the position of the mutation
        """
        return self.llr_base_score * self.normilized_entropy

    @cached_property
    def entropy_weighted_base_wt_score(self):  # base_wt's score
        """
        The entropy weighted score of the mutant in the position of the mutation
        """
        return self.nadav_base_wt_score * self.normilized_entropy

    @cached_property
    def sqrt_normalized_entropy(self):
        """
        Alternative normalized entropy using square root: 1 - sqrt(entropy / log(20))
        """
        return 1 - torch.sqrt(self.entropy_tensor / np.log(len(ESM_AA_ORDER)))

    @cached_property
    def squared_sqrt_normalized_entropy(self):
        """
        Squared version of sqrt_normalized_entropy: (1 - sqrt(entropy / log(20)))²
        """
        sqrt_norm = self.sqrt_normalized_entropy
        return sqrt_norm * sqrt_norm

    @cached_property
    def log_ratio_entropy(self):
        """
        Log ratio entropy function: (log(log(20) + 1) - log(entropy + 1)) / log(log(20) + 1)
        """
        log_term = math.log(math.log(len(ESM_AA_ORDER)) + 1)
        return (log_term - torch.log(self.entropy_tensor + 1)) / log_term

    @cached_property
    def sigmoid_entropy(self):
        """
        Sigmoid function on entropy: 1 / (1 + e^(5(entropy - log(5))))
        """
        return 1 / (1 + torch.exp(5 * (self.entropy_tensor - math.log(5))))

    @cached_property
    def sqrt_norm_entropy_weighted_llr_score(self):
        """
        LLR score weighted by sqrt_normalized_entropy
        """
        return self.llr_base_score * self.sqrt_normalized_entropy

    @cached_property
    def squared_sqrt_norm_entropy_weighted_llr_score(self):
        """
        LLR score weighted by squared_sqrt_normalized_entropy
        """
        return self.llr_base_score * self.squared_sqrt_normalized_entropy

    @cached_property
    def log_ratio_entropy_weighted_llr_score(self):
        """
        LLR score weighted by log_ratio_entropy
        """
        return self.llr_base_score * self.log_ratio_entropy

    @cached_property
    def sigmoid_entropy_weighted_llr_score(self):
        """
        LLR score weighted by sigmoid_entropy
        """
        return self.llr_base_score * self.sigmoid_entropy

    @cached_property
    def sqrt_norm_entropy_weighted_base_wt_score(self):
        """
        base_wt score weighted by sqrt_normalized_entropy
        """
        return self.nadav_base_wt_score * self.sqrt_normalized_entropy

    @cached_property
    def squared_sqrt_norm_entropy_weighted_base_wt_score(self):
        """
        base_wt score weighted by squared_sqrt_normalized_entropy
        """
        return self.nadav_base_wt_score * self.squared_sqrt_normalized_entropy

    @cached_property
    def log_ratio_entropy_weighted_base_wt_score(self):
        """
        base_wt score weighted by log_ratio_entropy
        """
        return self.nadav_base_wt_score * self.log_ratio_entropy

    @cached_property
    def sigmoid_entropy_weighted_base_wt_score(self):
        """
        base_wt score weighted by sigmoid_entropy
        """
        return self.nadav_base_wt_score * self.sigmoid_entropy
