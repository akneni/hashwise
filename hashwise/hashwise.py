from device_status import DeviceStatus
from exceptions import *
import ctypes as _ctypes
from time import perf_counter as _perf_counter
import hashlib as _hashlib
from tqdm import tqdm as _tqdm
import os as _os
from pathlib import Path as _Path

# Load cuda dependencies
try:
    _cudalib_path = _Path(_os.path.dirname(_os.path.realpath(__file__))) / "cuda-libraries" / "cudalib-windows-64bit.dll"
    _cudalib = _ctypes.cdll.LoadLibrary(str(_cudalib_path))
except FileNotFoundError:
    raise DependencyNotFoundError(f"Cannot load cuda dependencies. Double check that {_cudalib_path} exists.")

# Declare cuda function argument and return types
_cudalib.sha256_brute_force.argtypes = [_ctypes.c_int, _ctypes.c_char_p, _ctypes.c_int, _ctypes.c_char_p]
_cudalib.sha256_brute_force.restype = _ctypes.c_char_p

def blake2b(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.blake2b(payload).hexdigest()

def blake2s(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.blake2s(payload).hexdigest()

def md5(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.md5(payload).hexdigest()

def sha1(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha1(payload).hexdigest()

def sha224(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha224(payload).hexdigest()

def sha256(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha256(payload).hexdigest()

def sha384(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha384(payload).hexdigest()

def sha512(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha512(payload).hexdigest()

def sha3_224(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha3_224(payload).hexdigest()

def sha3_256(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha3_256(payload).hexdigest()

def sha3_384(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha3_384(payload).hexdigest()

def sha3_512(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha3_512(payload).hexdigest()

def shake_128(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.shake_128(payload).hexdigest()

def shake_256(payload:bytes):
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.shake_256(payload).hexdigest()

def brute_force_hash(hash_algorithm, possible_elements, target:str, len_permutation:int=None, string_encoding:str='utf-8', use_gpu=None, show_progress_bar=False):
    if hash_algorithm not in all_algorithms:
        raise ValueError("Argument 'hash_algorithm' must be a hashing algorithm from hashwise library. See 'hashwise.all_algorithms' for complete list.")
    
    if not isinstance(target, str):
        raise TypeError("Argument 'target' must be a string.")
    elif len(target) != hash_algo_res_length[hash_algorithm]:
        raise ValueError("Argument 'target' mush be a hash (in base 16).")

    # checks and formats the possible_elements argument
    gen_type = str
    if isinstance(possible_elements, str) or all([isinstance(i, str) for i in possible_elements]):
        if len(possible_elements) != len(set(possible_elements)):
            raise ValueError("Argument 'possible_elements' cannot have repeat values.")
        elif not all([len(i) == 1 for i in possible_elements]):
            raise ValueError("Argument 'possible_elements' must be a collection of chars. No element can have multiple caracters.")
        if not (isinstance(possible_elements, str)):
            possible_elements = ''.join(list(possible_elements))
        if len_permutation is None:
            raise TypeError("Argument 'len_permutation' must be defined when passing colections of chars to 'possible_elements'")
        permutation_generator = __perm_gen_str(possible_elements, len_permutation)
        permutation_length = len(possible_elements)**len_permutation

    elif 1 <= len(possible_elements) <= 3 and all([isinstance(i, (int)) for i in possible_elements]):
        if len(possible_elements) == 1:
            possible_elements = [0, possible_elements[0], 1]
        elif len(possible_elements) == 2:
            possible_elements = [possible_elements[0], possible_elements[1], 1]
        permutation_generator = __perm_gen_num(*possible_elements)
        permutation_length = (possible_elements[1] - possible_elements[0]) // possible_elements[2]
        gen_type = int
    elif isinstance(possible_elements, range):
        permutation_generator = possible_elements
        permutation_length = (permutation_generator.stop - permutation_generator.start) // permutation_generator.step
        gen_type = int
    else:
        raise ValueError("Argument 'possible_elements' must be a collection of strings/chars or a range (start:int, end:int, step:int) as a range object or a list/tuple.")

    if use_gpu is not None:
        if use_gpu:
            if not DeviceStatus.device_available():
                raise GPUNotAccessibleError("GPU not found. If a Nvidia graphics card is available on your system, follow the instructions at https://developer.nvidia.com/cuda-downloads to download CUDA.")
    else:
        use_gpu = 4_000_000 < permutation_length
        use_gpu = False # until gpu portion is implemented

    if use_gpu and show_progress_bar:
        print("Warning: progress bar cannot be shown when computing on graphics card.")

    if use_gpu:
        res:bytes = _cudalib.sha256_brute_force(len_permutation, possible_elements.encode('utf-8'), len(possible_elements), target.encode('utf-8'))
        res = res.decode(string_encoding)
        if all([i=='z' for i in res]):
            return None
        if len(res) != len_permutation: raise UnknownGPUError("An unknown error occured when computing on the GPU. Please set use_gpu=False and we will get all bugs fixed soon!")
        return res
    else:        
        if show_progress_bar:
            permutation_generator = _tqdm(permutation_generator, total=permutation_length)

        if gen_type == str:
            for i in permutation_generator:
                byte_obj = bytes(i, encoding=string_encoding)
                if hash_algorithm(byte_obj) == target:
                    return i
        elif gen_type == int:
            for i in permutation_generator:
                byte_obj = bytes(i)
                if hash_algorithm(byte_obj) == target:
                    return i
            
    return None

def brute_force_time_estimate(hash_algorithm, possible_elements, length:int=None, string_encoding:str='utf-8', units='seconds', num_trials=None):
    if hash_algorithm not in all_algorithms:
        raise ValueError("Argument 'hash_algorithm' must be a hashing algorithm from hashwise library. See 'hashwise.all_algorithms' for complete list.")
    
    unit_lst = {
        'units': ['seconds', 'minutes', 'hours', 'days', 'weeks', 'years', 'decades', 'centuries', 'millennium'],
        'modifiers': [1, 60, 60, 24, 7, 52.14, 10, 10, 10]
    }

    units = units.lower()
    if units not in unit_lst['units']:
        raise ValueError ("Argument 'units' must one of", unit_lst["units"])

    # checks and formats the possible_elements argument
    gen_type = str
    if isinstance(possible_elements, str) or all([isinstance(i, str) for i in possible_elements]):
        permutation_generator = __perm_gen_str(possible_elements, length)
        permutation_length = len(possible_elements)**length
        if length is None:
            raise TypeError("Argument 'length' must be defined when lassing colections of chars to 'possible_elements'")
    elif 1 <= len(possible_elements) <= 3 and all([isinstance(i, (int)) for i in possible_elements]):
        if len(possible_elements) == 1:
            possible_elements = [0, possible_elements[0], 1]
        elif len(possible_elements) == 2:
            possible_elements = [possible_elements[0], possible_elements[1], 1]
        permutation_generator = __perm_gen_num(*possible_elements)
        permutation_length = (possible_elements[1] - possible_elements[0]) // possible_elements[2]
        gen_type = int
    elif isinstance(possible_elements, range):
        permutation_generator = possible_elements
        permutation_length = (permutation_generator.stop - permutation_generator.start) // permutation_generator.step
        gen_type = int
    else:
        raise ValueError("Argument 'possible_elements' must be a collection of strings/chars or a range (start:int, end:int, step:int) as a range object or a list/tuple.")
    
    if num_trials is None:
        num_trials = 10_000 if (gen_type == str) else 500

    start  =_perf_counter()
    counter = 0
    loop_broken = False
    if gen_type == str:
        for i in permutation_generator:
            byte_obj = bytes(i, encoding=string_encoding)
            hash_algorithm(byte_obj)
            counter += 1
            if counter >= num_trials:
                loop_broken = True
                break
    elif gen_type == int:
        for i in permutation_generator:
            byte_obj = bytes(i)
            hash_algorithm(byte_obj)
            counter += 1
            if counter >= num_trials:
                loop_broken = True
                break
    end = _perf_counter()

    modifier = 1
    for un, mod in zip(unit_lst["units"], unit_lst["modifiers"]):
        modifier *= mod
        if units == un:
            break            

    return (((end-start) / num_trials) * (permutation_length) if loop_broken else (end-start)) / modifier

all_algorithms = [
    blake2b,
    blake2s,
    md5,
    sha1,
    sha224,
    sha256,
    sha384,
    sha512,
    sha3_224,
    sha3_256,
    sha3_384,
    sha3_512,
    shake_128,
    shake_256,
]

hash_algo_res_length = {
    blake2b: 0,
    blake2s: 0,
    md5: 0,
    sha1: 0,
    sha224: 0,
    sha256: 64,
    sha384: 0,
    sha512: 0,
    sha3_224: 0,
    sha3_256: 0,
    sha3_384: 0,
    sha3_512: 0,
    shake_128: 0,
    shake_256: 0,
}

def __perm_gen_str(chars, length):
    if length == 0:
        yield ''
    else:
        for char in chars:
            for perm in __perm_gen_str(chars, length - 1):
                yield char + perm

def __perm_gen_num(start, end, step):
    # In order to avoid floating point errors
    num_deci = max([len(str(i).split('.')[1]) if '.' in str(i) else 0 for i in (start, end, step)])

    counter = start
    while counter < end:
        yield round(counter, num_deci)
        counter += step
