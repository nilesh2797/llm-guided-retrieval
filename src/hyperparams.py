import re, os
import argparse

def abbreviate_key(key: str) -> str:
    """
    Systematically abbreviate a key by:
    - Splitting on underscores
    - Taking first 3 letters of each part
    - Keeping numbers intact
    """
    parts = re.split(r'[_\-]', key)
    abbrev_parts = []
    for p in parts:
        if p.isdigit() or len(p) < 4:
            abbrev_parts.append(p.capitalize())  # keep numbers and short parts as is
        else:
            abbrev_parts.append(p[:1].capitalize())  # take first 3 letters
    return "".join(abbrev_parts)

def compress_hparam_string(hparam_str: str) -> str:
    """
    Compress a hyperparameter string into a shorter form
    without hardcoded replacement rules.
    """
    parts = hparam_str.split("--")
    compressed_parts = []

    for part in parts:
        if "=" in part:
            key, val = part.split("=", 1)
            if val and val.lower() not in ['false', 'none']:
                val = val.replace('/', '__')
                compressed_parts.append(f"{abbreviate_key(key)}={val}")
        else:
            # Handle flag-only parameters
            compressed_parts.append(abbreviate_key(part))

    final_str = "-".join(compressed_parts)
    final_str = re.sub(r"\.log$", "", final_str)

    return final_str

class HyperParams(argparse.Namespace):
    NO_SAVE_VARS = set(['dataset', 'rerank', 'load_existing', 'llm_max_concurrent_calls', 'num_threads', 'search_with_path_relevance', 'llm_api_timeout', 'llm_api_max_retries', 'llm_api_staggering_delay'])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __str__(self):
        return compress_hparam_string('--'.join(f'{k.lower()}={v}' for k, v in vars(self).items() if k.lower() not in self.NO_SAVE_VARS))

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, k):
        return vars(self).get(k.lower())
    
    @classmethod
    def from_args(cls, args=None):
        """Parse command line arguments and return HyperParams instance"""
        parser = argparse.ArgumentParser(description='Hyperparameters')
        
        # Add common hyperparameters here
        parser.add_argument('--dataset', type=str, default='BRIGHT')
        parser.add_argument('--subset', type=str, required=True, help='Subset of data to use')
        parser.add_argument('--tree_version', type=str, required=True, help='Version of the tree structure to use')
        parser.add_argument('--traversal_prompt_version', type=int, default=5)
        parser.add_argument('--reasoning_in_traversal_prompt', type=int, default=-1)
        parser.add_argument('--max_query_char_len', type=int, default=None)
        parser.add_argument('--max_doc_desc_char_len', type=int, default=None)
        parser.add_argument('--max_prompt_proto_size', type=int, default=None)
        parser.add_argument('--search_with_path_relevance', type=bool, default=True)
        parser.add_argument('--num_leaf_calib', type=int, default=10)
        parser.add_argument('--pl_tau', type=float, default=5.0)
        parser.add_argument('--relevance_chain_factor', type=float, default=0.5)
        parser.add_argument('--llm_api_backend', type=str, default='genai')
        parser.add_argument('--llm', type=str, default='gemini-2.5-flash')
        parser.add_argument('--llm_max_concurrent_calls', type=int, default=20)
        parser.add_argument('--llm_api_timeout', type=int, default=120)
        parser.add_argument('--llm_api_max_retries', type=int, default=4)
        parser.add_argument('--llm_api_staggering_delay', type=float, default=0.1)
        parser.add_argument('--num_iters', type=int, default=20)
        parser.add_argument('--num_eval_samples', type=int, default=1_000)
        parser.add_argument('--max_beam_size', type=int, default=2)
        parser.add_argument('--rerank', default=False, action='store_true')
        parser.add_argument('--load_existing', default=False, action='store_true') 
        parser.add_argument('--num_threads', type=int, default=os.cpu_count())
        parser.add_argument('--suffix', type=str, default='')
        
        # Parse arguments
        parsed_args = parser.parse_args(args.split() if args else None)
        
        # Create HyperParams instance from parsed arguments
        return cls(**vars(parsed_args))
    
    def add_param(self, key, value):
        """Add a parameter dynamically"""
        setattr(self, key, value)