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
