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
