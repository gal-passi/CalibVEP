from data_classes import METHODS_TO_ESM, AAMut, MutationVariantSet
import definitions as d
import torch
import re
import metapredict as meta

from mutation_record import MutationRecord

def process_raw_variant(description: str): 
    """
        validates variant description compatibility
    :param description: str format p.[AA][INDEX][AA]
    :return: re.search obj if successful else raise ValueError
    """
    if not description.startswith('p.'):
        description = 'p.' + description
    res = re.search(d.MUTATION_REGEX, description)
    if not res:
        raise ValueError("Invalid input valid format of form p.{AA}{location}{AA}")
    return res.group('orig'), res.group('change'), int(res.group('location'))

def esm_setup(model_name=d.ESM1B_MODEL, device=d.DEVICE):
    """
    :param model_name: str model name
    :return: model, alphabet api
    """
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    model = model.to(device)
    print(f"model loaded on {device}")
    return model, alphabet


def process_mutation_name(mutation, offset):
    """
    :param mutation: str in format R29L
    :return: wt_AA, loc, change_AA
    """
    return AAMut(
        wt_aa=mutation[0],
        mut_idx=int(mutation[1:-1]) - offset,
        change_aa=mutation[-1],
    )


def identify_idrs(disorder_values, threshold, min_length=3):
    """
    Identify Intrinsically Disordered Regions (IDRs) manually.

    Parameters:
    -----------
    disorder_values : list
        List of disorder scores
    threshold : float
        Threshold for considering a residue as disordered
    min_length : int
        Minimum length of an IDR

    Returns:
    --------
    list
        List of (start, end) tuples for IDRs (1-indexed)
    """
    idrs = []
    in_idr = False
    start = 0

    for i, score in enumerate(disorder_values):
        if score >= threshold and not in_idr:
            # Start of a new IDR
            in_idr = True
            start = i + 1  # 1-indexed position
        elif score < threshold and in_idr:
            # End of current IDR
            end = i  # End position is the last disordered residue
            if end - start + 1 >= min_length:
                idrs.append((start, end))
            in_idr = False

    # Check if we ended while still in an IDR
    if in_idr:
        end = len(disorder_values)
        if end - start + 1 >= min_length:
            idrs.append((start, end))

    return idrs


def is_position_in_idr(position, idrs):
    """
    Check if a position is within any IDR.

    Parameters:
    -----------
    position : int
        Position to check (1-indexed)
    idrs : list
        List of (start, end) tuples for IDRs (1-indexed)

    Returns:
    --------
    bool
        True if position is in an IDR, False otherwise
    """
    for start, end in idrs:
        if start <= position <= end:
            return True
    return False


def is_disordered(seq: str, idx: int) -> bool:
    """ Check if a residue at a given index in a sequence is disordered.
    This function uses the MetaPredict library to predict disorder scores for the sequence

    Args:
        seq (str): The protein sequence to analyze.   
        idx (int): The index of the residue to check.

    Returns:
        bool: True if the residue at the given index is disordered, False otherwise.
    """
    disorder_values = meta.predict_disorder(seq)
    idrs = identify_idrs(disorder_values, threshold=d.DISORDERED_THRESHOLD, min_length=3)
    is_disordered = is_position_in_idr(idx + 1, idrs)  # TODO check if we should use idx + 1 or idx
    return is_disordered


def find_mask_positions(protein_seq, mask_token):
    """
    Find all mask token positions in the protein sequence.
    
    Args:
        protein_seq: The protein sequence string
        mask_token: The mask token to search for
        
    Returns:
        List of tuples (start, end) for each mask token position
    """
    mask_positions = []
    start_pos = 0
    
    while True:
        mask_pos = protein_seq.find(mask_token, start_pos)
        if mask_pos == -1:
            break
        mask_positions.append((mask_pos, mask_pos + len(mask_token)))
        start_pos = mask_pos + len(mask_token)
    
    return mask_positions


def would_split_mask(chunk_start, chunk_end, mask_positions):
    """
    Check if chunk boundaries would split any mask token.
    
    Args:
        chunk_start: Start position of the chunk
        chunk_end: End position of the chunk
        mask_positions: List of (start, end) tuples for mask positions
        
    Returns:
        Tuple of (would_split, mask_start, mask_end)
    """
    for mask_start, mask_end in mask_positions:
        # Check if mask is partially inside the chunk (would be split)
        if (chunk_start < mask_end and chunk_end > mask_start and
                not (chunk_start <= mask_start and chunk_end >= mask_end)):
            return True, mask_start, mask_end
    return False, None, None


def calculate_chunk_positions(seq_length, chunk_size, overlap_size, mask_positions):
    """
    Calculate optimal chunk positions that don't split mask tokens.
    
    Args:
        seq_length: Total length of the sequence
        chunk_size: Maximum size of each chunk
        overlap_size: Size of overlap between chunks
        mask_positions: List of mask token positions
        
    Returns:
        List of dictionaries with chunk position information
    """
    chunk_positions = []
    start_pos = 0
    
    while start_pos < seq_length:
        end_pos = min(start_pos + chunk_size, seq_length)
        
        # Check if this chunk would split a mask
        would_split, mask_start, mask_end = would_split_mask(start_pos, end_pos, mask_positions)
        
        if would_split:
            # Adjust chunk_end to not split the mask
            if mask_start >= start_pos:
                # Mask starts within or after chunk start - end chunk before mask
                end_pos = mask_start
            else:
                # Mask starts before chunk start but ends within chunk
                # Include the entire mask
                end_pos = mask_end
        
        # If the adjusted chunk is too small, include the mask
        if end_pos - start_pos < overlap_size and would_split:
            for mask_start, mask_end in mask_positions:
                if mask_start < start_pos + chunk_size and mask_end > end_pos:
                    end_pos = min(mask_end, seq_length)
                    break
        
        # Skip empty chunks
        if end_pos <= start_pos:
            start_pos += 1
            continue
        
        # Define the valid region (excluding padding/overlap)
        valid_start = overlap_size if start_pos > 0 else 0
        valid_end = (end_pos - start_pos) - overlap_size if end_pos < seq_length else (end_pos - start_pos)
        
        # Ensure valid_end doesn't exceed chunk size and we have a valid region
        valid_end = min(valid_end, end_pos - start_pos)
        if valid_end <= valid_start:
            valid_end = end_pos - start_pos
        
        chunk_positions.append({
            'chunk_start': start_pos,
            'chunk_end': end_pos,
            'valid_start': valid_start,
            'valid_end': valid_end
        })
        
        # Break if we've reached the end
        if end_pos >= seq_length:
            break
        
        # Calculate next start position with overlap
        next_start = start_pos + chunk_size - 2 * overlap_size
        
        # Ensure we don't start in the middle of a mask token
        for mask_start, mask_end in mask_positions:
            if mask_start < next_start < mask_end:
                next_start = mask_end
                break
        
        start_pos = next_start
    
    return chunk_positions


def process_chunks_and_combine_logits(protein_seq, chunk_positions, alphabet, seq_name, model):
    """
    Process each chunk through the model and combine the logits.
    
    Args:
        protein_seq: The full protein sequence
        chunk_positions: List of chunk position dictionaries
        alphabet: The ESM alphabet
        seq_name: Name of the sequence
        model: The ESM model
        
    Returns:
        Combined logits tensor for the entire sequence
    """
    final_logits = None
    
    for i, pos in enumerate(chunk_positions):
        # Extract the chunk
        chunk_seq = protein_seq[pos['chunk_start']:pos['chunk_end']]
        
        # Process the chunk through the model
        batch_tokens = get_batch_token(alphabet, seq_name, chunk_seq)
        chunk_logits = get_trunctad_logits(True, batch_tokens, model)
        chunk_logits = chunk_logits.squeeze(0)
        
        # Extract the valid region (excluding overlaps)
        valid_logits = chunk_logits[pos['valid_start']:pos['valid_end']]
        
        # Combine with previous logits
        if final_logits is None:
            final_logits = valid_logits
        else:
            final_logits = torch.cat([final_logits, valid_logits], dim=0)
    
    return final_logits


def process_long_sequence_chunking_with_overlapping_regions(alphabet, seq_name, protein_seq, model):
    """
    Process long protein sequences by chunking with overlapping regions.
    Ensures that mask tokens are never split across chunks.

    Args:
        alphabet: The ESM alphabet
        seq_name: Name of the sequence
        protein_seq: The full protein sequence
        model: The ESM model

    Returns:
        Combined logits for the entire sequence
    """
    seq_length = len(protein_seq)
    
    mask_positions = find_mask_positions(protein_seq, d.MASK_TOKEN)
    
    chunk_positions = calculate_chunk_positions(
        seq_length, 
        d.ESM_MAX_LENGTH, 
        d.OVERLAP_SIZE_LONG_PROTEIN, 
        mask_positions
    )
    
    final_logits = process_chunks_and_combine_logits(
        protein_seq, 
        chunk_positions, 
        alphabet, 
        seq_name, 
        model
    )
    
    return final_logits


def get_batch_token(alphabet, example_name, sequence):
    tokenizer = alphabet.get_batch_converter()
    input = [(example_name, sequence)]
    _, _, batch_tokens = tokenizer(input)
    batch_tokens = batch_tokens.to(d.DEVICE)
    return batch_tokens


def get_trunctad_logits(aa_only, batch_tokens, model):
    chunk_logits = model(batch_tokens, repr_layers=d.REP_LAYERS, return_contacts=False)['logits']
    logit_parts = []
    if d.ESM1B_MODEL == 'esm1_t6_43M_UR50S':
        logit_parts.append(chunk_logits[0, 1:, 4:24] if aa_only else chunk_logits[0, 1:, :])
    else:
        logit_parts.append(chunk_logits[0, 1:-1, 4:24] if aa_only else chunk_logits[0, 1:-1, :])
    return torch.stack(logit_parts).to(d.DEVICE)


def get_mutant_dest_and_seq(method_mutant, sequence, aa_mut: AAMut):
    if method_mutant == METHODS_TO_ESM.MUTANTE:
        mutant_seq = sequence[:aa_mut.mut_idx] + aa_mut.change_aa + sequence[aa_mut.mut_idx + 1:]
    elif method_mutant == METHODS_TO_ESM.MASKED:
        mutant_seq = sequence[:aa_mut.mut_idx] + "<mask>" + sequence[aa_mut.mut_idx + 1:]
    elif method_mutant == METHODS_TO_ESM.WT:
        mutant_seq = sequence
    else:
        raise ValueError(f"Unknown method_mutant: {method_mutant}")
    return mutant_seq


def run_esm(model, alphabet, protein_seq: str, aa_mut: AAMut):
    mutation_variant_set = MutationVariantSet()

    model.eval()
    with torch.no_grad():
        for method_mutant in METHODS_TO_ESM.get_methods():
            mutant_seq = get_mutant_dest_and_seq(method_mutant, protein_seq, aa_mut)
            seq_name = f"{method_mutant.value}_{aa_mut.wt_aa}{aa_mut.mut_idx}{aa_mut.mut_idx}"
            truncated_logits = process_long_sequence_chunking_with_overlapping_regions(alphabet, seq_name, mutant_seq, model)

            mutant_record = MutationRecord(
                protein_seq=protein_seq,
                aa_mut=aa_mut,
                truncated_logits=truncated_logits,
            )
            mutation_variant_set.add_mutation_record(mutant_record, method_mutant)

    return mutation_variant_set



def has_homologs(seq: str):
    raise NotImplementedError()
