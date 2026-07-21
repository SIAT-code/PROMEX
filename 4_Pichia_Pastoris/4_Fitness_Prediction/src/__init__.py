import os
import json
import pandas as pd
from collections import namedtuple



with open(f'{os.path.dirname(__file__)}/config_round2_single_mutant.json') as f:  # set config_round1_single_mutant.json, config_round2_single_mutant.json, config_round3_combinated_mutant.json
    config = json.load(f)
config = namedtuple('config', config.keys())(**config)

pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)
