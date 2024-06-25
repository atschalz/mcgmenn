'''
Utilities for Tensorflow sessions
'''

import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def set_gpu(gpu: int, mem_frac: float=1.0):
    """Assign GPU to Tensorflow session and limit memory usage

    Args:
        gpu (int): GPU index
        mem_frac (float, optional): memory limit. Defaults to 1.0.
    """    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(gpu))
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = mem_frac
    session = InteractiveSession(config=config)
