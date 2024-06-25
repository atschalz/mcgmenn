'''
Miscellaneous utilities for expanding paths, getting datestamps, etc.
'''
import os
import datetime
from .settings import RESULTSDIR, DATADIR

def get_datestamp(with_time: bool=False):
    """Get datestamp with optional timestamp in format YYYY-MM-DD.

    Args:
        with_time (bool, optional): Include timestamp in format HH-mm-SS.
        Defaults to False.

    Returns:
        str: datestamp
    """    
    
    now = datetime.datetime.now()
    strDate = now.strftime('%Y-%m-%d')
    
    if with_time:
        strDate += '_' + now.strftime('%H-%M-%S')
        
    return strDate

def _expand_path(path: str, root: str, make: bool=False):
    """Concatenate root + base paths and make directory if needed.

    Args:
        path (str): base path
        root (str): root path
        make (bool, optional): Make directory if needed. 
            Defaults to False.

    Raises:
        FileNotFoundError: If make == False and concatenated 
            path does not exist.
    """    
    if path.startswith(os.path.sep):
        # Already an absolute path
        newpath = path
    else:
        newpath = os.path.join(root, path)
        
    if make:
        if os.path.exists(newpath):
            print('Warning: output path already exists')
        else:
            os.makedirs(newpath)
    else:
        if not os.path.exists(newpath):
            raise FileNotFoundError(newpath)

def expand_results_path(path: str, make: bool=False):
    """Concatenate directory path to RESULTSDIR in settings.py.

    Args:
        path (str): base path
        make (bool, optional): Make directory if it 
            does not exist. Defaults to False.

    Returns:
        str: expanded path
    """    
    return _expand_path(path, RESULTSDIR, make=make)

def expand_data_path(path: str, make: bool=False):
    """Concatenate path to DATADIR in settings.py.

    Args:
        path (str): base path
        make (bool, optional): Make directory if it 
            does not exist. Defaults to False.

    Returns:
        str: expanded path
    """ 
    return _expand_path(path, DATADIR, make=make)    

def make_random_onehot(n_rows: int, n_cols: int):
    """Create a random one-hot encoding matrix

    Args:
        n_rows (int): number of rows
        n_cols (int): number of columns
        
    Returns:
        array: one-hot matrix
    """    
    import numpy as np

    arrEye = np.eye(n_cols)
    arrRandomCols = np.random.choice(n_cols, size=(n_rows,))
    arrOnehot = arrEye[arrRandomCols,]
    return arrOnehot
    