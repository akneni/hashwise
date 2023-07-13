import numpy as _np
import ctypes as _ctypes
import hashlib as _hashlib
import os as _os
from pathlib import Path as _Path

_current_dir = _Path(_os.path.dirname(_os.path.realpath(__file__)))
_cudalib_path = _current_dir / "cuda-libraries" / "cudalib.dll"
cudalib = _ctypes.cdll.LoadLibrary(str(_cudalib_path))

_current_dir = _Path(_os.path.dirname(_os.path.realpath(__file__)))
_clib_path = _current_dir / "c-libraries" / "clib.dll"
clib = _ctypes.cdll.LoadLibrary(str(_clib_path))



def sha256(payload:bytes):
    return _hashlib.sha256(payload).hexdigest()





def num_gpu():
    return max(cudalib.cuda_enabled(), 0)

def __partition(num, num_cores):
    dif = num // num_cores
    partitions = []
    curr = 0
    for i in range (num_cores):
        if i != num_cores-1:
            partitions.append((curr, curr+dif))
            curr += dif
        else:
            partitions.append((curr, num))
    
    return partitions
