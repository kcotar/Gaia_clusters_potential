import os
import numpy as np


def string_bool(str):
    return str in ['T', 't', 'True', 'true', '1']


def get_abundance_cols3(col_names):
    abund_col_names = [col for col in col_names if '_abund_cannon' in col and 'e_' not in col and 'flag_' not in col and len(col.split('_'))<4]
    return list(np.sort(abund_col_names))


def get_element_names(col_names):
    names = [val.split('_')[0].capitalize() for val in col_names]
    return names


def move_to_dir(path):
    if not(os.path.isdir(path)):
        os.mkdir(path)
    os.chdir(path)


def create_dir(path):
    if not(os.path.isdir(path)):
        os.mkdir(path)


