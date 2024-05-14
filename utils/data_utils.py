# --------------------------------------------------
# Imports
# --------------------------------------------------

import os
import dill
from pandas import read_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------
# Load
# --------------------------------------------------

def full_loader(file_name, root=ROOT):
    """
    Loads a database
    
    :param file_name: Name of file
    :param root: Root directory (optional)
    :return: Datafile
    """
    file_path = root + '/data/' + str(file_name)
    data = read_csv(file_path)
    return data

# --------------------------------------------------

def load_result(file_name):
    """
    Loads a result file
    
    :param file_name: Name of file
    :return: Catalogue
    """
    with open(ROOT+'/results/'+str(file_name), 'rb') as f:
        new_catalogue = dill.load(f)
    return new_catalogue

# --------------------------------------------------
# Save
# --------------------------------------------------

def save_result(data, file_name):
    """
    Saves a result file
    
    :param file_name: Name of file
    """
    with open(ROOT+'/results/'+str(file_name), 'wb') as f:
        dill.dump(data, f)