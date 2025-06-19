from definitions import *
import re

def process_raw_variant(description: str): 
        """
         validates variant description compatibility
        :param description: str format p.[AA][INDEX][AA]
        :return: re.search obj if successful else raise ValueError
        """
        if not description.startswith('p.'):
            description = 'p.' + description
        res = re.search(MUTATION_REGEX, description)
        if not res:
            raise ValueError("Invalid input valid format of form p.{AA}{location}{AA}")
        return res.group('orig'), res.group('change'), int(res.group('location'))

def esm_setup(model_name=ESM1B_MODEL):
    """
    :param model_name: str model name
    :return: model, alphabet api
    """
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    model = model.to(DEVICE)
    print(f"model loaded on {DEVICE}")
    return model, alphabet

def esm_process_long_sequences(seq, loc):
    """
    trims seq to len < 1024 using mut Object. to fit for bert model
    :return: offset from orig location, trimmed sequence
    """
    thr = ESM_MAX_LENGTH // 2
    left_bound = 0 if loc-thr < 0 else loc - thr
    right_bound = len(seq) if loc + thr >len(seq) else loc+thr
    left_excess = 0 if loc - thr > 0 else abs(loc-thr)
    right_excess = 0 if loc + thr <= len(seq) else loc+thr-len(seq)
    if (left_excess == 0) and (right_excess == 0):
        return left_bound, seq[left_bound:right_bound]
    if left_excess > 0:
        return 0, seq[left_bound:right_bound+left_excess]
    if right_excess > 0:
        return left_bound-right_excess, seq[left_bound-right_excess:right_bound]


def esm_seq_logits(model, tokens, log=True, softmax=True, return_device='cpu', esm_version=1):
    """
    :param model: ESM3 initiated model
    :param tokens: tokenized sequence
    :param log: bool return log of logits
    :param softmax: bool return softmax of logits
    :param return_device: 'cpu' | 'cuda should logits be returned on cpu or cuda
    :return: numpy array - log_s
    """

    if return_device == 'cuda':
        assert return_device == DEVICE, 'return type cuda but no GPU detected'
    model.eval()
    with torch.no_grad():
        if esm_version == 3:
            assert tokens.device == model.device, 'tokens and model must be on the same device'
            output = model.forward(sequence_tokens=tokens)
            sequence_logits = output.sequence_logits
        else:  #  esm 1,2
            assert tokens.device == next(model.parameters()).device, 'tokens and model must be on the same device'
            sequence_logits = model(tokens, repr_layers=REP_LAYERS, return_contacts=False)['logits']
        aa_logits = sequence_logits[0, 1:-1, 4:24]
        if log and softmax:
            token_probs = torch.log_softmax(aa_logits, dim=-1)
        elif log:
            token_probs = torch.log(aa_logits)
        elif softmax:
            token_probs = torch.softmax(aa_logits, dim=-1)
        token_probs = token_probs.cpu().numpy()
        return token_probs

    def score_mutation_esm_inference(model, alphabet, sequence: str, wt_aa: str, alt_aa: str, mut_idx: int, method='masked_marginals'):
        """
        :param model: esm initiated model
        :param alphabet: alphabet api
        :param sequence: protein sequence
        :param mut_idx: int index of mutation in sequence
        :param wt_aa: str wild-type amino-acid
        :param alt_aa: str altered amino-acid 
        :param method: scoring method str: wt_marginals | mutant_marginals | masked_marginals
        :return: tuple (float | None ESM3 masked marginal, str log)
        """
        if len(sequence) > ESM_MAX_LENGTH:
            new_offset, sequence = esm_process_long_sequences(sequence, mut_idx)
            mut_idx = mut_idx - new_offset
        assert sequence[mut_idx] == wt_aa, 'sequence does not match wild-type AA - check offset'
        if method == 'mutant_marginals':
            input_seq = sequence[:mut_idx] + alt_aa + sequence[mut_idx + 1:]
        elif method == 'masked_marginals':
            input_seq = sequence[:mut_idx] + MASK_TOKEN + sequence[mut_idx + 1:]
        else:
            input_seq = sequence
        tokenizer = alphabet.get_batch_converter()
        # tokens = torch.tensor(tokenized, dtype=torch.int64).unsqueeze(0).to(DEVICE)
        _, _, batch_tokens = tokenizer([(f"p.{wt_aa}{mut_idx}{alt_aa}, input_seq)])
        batch_tokens = batch_tokens.to(DEVICE)
        logits = esm_seq_logits(model=model, tokens=batch_tokens, log=True, softmax=True, return_device='cpu',
                                esm_version=1)
        return float(logits[mut_idx][AA_ESM_LOC[alt_aa]] - logits[mut_idx][AA_ESM_LOC[wt_aa]]), log
