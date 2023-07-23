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
_cudalib.sha256_brute_force.argtypes = [_ctypes.c_int, _ctypes.c_char_p, _ctypes.c_int, _ctypes.c_char_p, _ctypes.c_int, _ctypes.c_int]
_cudalib.sha256_brute_force.restype = _ctypes.c_char_p
_cudalib.md5_brute_force.argtypes = [_ctypes.c_int, _ctypes.c_char_p, _ctypes.c_int, _ctypes.c_char_p, _ctypes.c_int, _ctypes.c_int]
_cudalib.md5_brute_force.restype = _ctypes.c_char_p
_cudalib.sha1_brute_force.argtypes = [_ctypes.c_int, _ctypes.c_char_p, _ctypes.c_int, _ctypes.c_char_p, _ctypes.c_int, _ctypes.c_int]
_cudalib.sha1_brute_force.restype = _ctypes.c_char_p

def blake2b(payload:bytes):
    """
    This function generates a BLAKE2b hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the BLAKE2b hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.blake2b(payload).hexdigest()

def blake2s(payload:bytes):
    """
    This function generates a BLAKE2s hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the BLAKE2s hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.blake2s(payload).hexdigest()

def md5(payload:bytes):
    """
    This function generates a MD-5 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the MD-5 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.md5(payload).hexdigest()

def sha1(payload:bytes):
    """
    This function generates a SHA-1 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the SHA-1 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha1(payload).hexdigest()

def sha224(payload:bytes):
    """
    This function generates a SHA-224 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the SHA-224 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha224(payload).hexdigest()

def sha256(payload:bytes):
    """
    This function generates a SHA-256 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the SHA-256 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha256(payload).hexdigest()

def sha384(payload:bytes):
    """
    This function generates a SHA-384 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the SHA-384 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha384(payload).hexdigest()

def sha512(payload:bytes):
    """
    This function generates a SHA-512 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the SHA-512 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha512(payload).hexdigest()

def sha3_224(payload:bytes):
    """
    This function generates a SHA-3-224 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the SHA-3-224 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha3_224(payload).hexdigest()

def sha3_256(payload:bytes):
    """
    This function generates a SHA-3-256 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the SHA-3-256 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha3_256(payload).hexdigest()

def sha3_384(payload:bytes):
    """
    This function generates a SHA-3-384 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the SHA-3-384 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha3_384(payload).hexdigest()

def sha3_512(payload:bytes):
    """
    This function generates a SHA-3-512 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.

    Returns:
    A hexadecimal string representing the SHA-3-512 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.sha3_512(payload).hexdigest()

def shake_128(payload:bytes, length:int):
    """
    This function generates a SHAKE-128 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.
    length (int): The length of the hash to be generated.

    Returns:
    A hexadecimal string representing the SHAKE-128 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.shake_128(payload).hexdigest(length)

def shake_256(payload:bytes, length:int):
    """
    This function generates a SHAKE-256 hash of a given payload.

    Parameters:
    payload (bytes): The data to be hashed. Must be of type 'bytes'.
    length (int): The length of the hash to be generated.

    Returns:
    A hexadecimal string representing the SHAKE-256 hash of the payload.

    Raises:
    TypeError: If the payload is not of type 'bytes'.
    """
    if not isinstance(payload, bytes):
        raise TypeError("Argument 'payload' must be of type 'bytes'")
    return _hashlib.shake_256(payload).hexdigest(length)

def brute_force_hash(hash_algorithm, possible_elements, target:str, len_permutation:int=None, string_encoding:str='utf-8', use_gpu=None, numBlocks=32, numThreadsPerBlock=32, show_progress_bar=False):
    """
    This function attempts to find the original value of a hashed string by brute force.

    Parameters:
    hash_algorithm: A hashing algorithm from the hashwise library.
    possible_elements: A string or collection of characters to be used for generating permutations, or a range object or list/tuple specifying a range of integers.
    target (str): The hashed string that the function is trying to find the original value of.
    len_permutation (int, optional): The length of the permutations to be generated. Required if possible_elements is a collection of characters.
    string_encoding (str, optional): The encoding to be used when converting strings to bytes. Default is 'utf-8'.
    use_gpu (bool, optional): Whether to use the GPU for computation. If not specified, the function will decide based on the number of permutations.
    numBlocks (int, optional): The number of blocks to be used when computing on the GPU. Default is 32.
    numThreadsPerBlock (int, optional): The number of threads per block to be used when computing on the GPU. Default is 32.
    show_progress_bar (bool, optional): Whether to show a progress bar. Default is False.

    Returns:
    The original value of the hashed string if found, otherwise None.

    Raises:
    ValueError: If hash_algorithm is not a valid hashing algorithm from the hashwise library, or if possible_elements contains repeat values or elements with multiple characters, or if target is not a hexadecimal hash of the correct length for the specified hash algorithm.
    TypeError: If target is not a string, or if len_permutation is not specified when possible_elements is a collection of characters.
    GPUNotAccessibleError: If use_gpu is True but no GPU is found.
    NotImplementedError: If use_gpu is True but GPU dependencies for the specified hash algorithm have not been implemented.
    UnknownGPUError: If an unknown error occurs when computing on the GPU.
    """
    if hash_algorithm not in _hash_algo_info.keys():
        raise ValueError("Argument 'hash_algorithm' must be a hashing algorithm from hashwise library. See 'hashwise.all_algorithms' for complete list.")
    
    if not isinstance(target, str):
        raise TypeError("Argument 'target' must be a string.")
    elif len(target) != _hash_algo_info[hash_algorithm]['hash_len']:
        raise ValueError("Argument 'target' mush be a hexadecimal hash of length {} for hash algorithm {}".format(_hash_algo_info[hash_algorithm]['hash_len'], _hash_algo_info[hash_algorithm]['name']) )

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
        if _hash_algo_info[hash_algorithm]['gpu_func'] is None:
            raise NotImplementedError("GPU dependencies for {} have yet to be implemented.".format(_hash_algo_info[hash_algorithm]['name']))
        res:bytes = _hash_algo_info[hash_algorithm]['gpu_func'](
            len_permutation, 
            possible_elements.encode('utf-8'), 
            len(possible_elements), 
            target.encode('utf-8'),
            numBlocks,
            numThreadsPerBlock
        )
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
                byte_obj = i.encode(string_encoding)
                if hash_algorithm(byte_obj) == target:
                    return i
        elif gen_type == int:
            for i in permutation_generator:
                byte_obj = bytes(i)
                if hash_algorithm(byte_obj) == target:
                    return i
            
    return None

def brute_force_time_estimate(hash_algorithm, possible_elements, length:int=None, string_encoding:str='utf-8', units='seconds', num_trials=None):
    """
    This function estimates the time it would take to brute force a hashed string using a specified hash algorithm and possible elements.

    Parameters:
    hash_algorithm: A hashing algorithm from the hashwise library.
    possible_elements: A string or collection of characters to be used for generating permutations, or a range object or list/tuple specifying a range of integers.
    length (int, optional): The length of the permutations to be generated. Required if possible_elements is a collection of characters.
    string_encoding (str, optional): The encoding to be used when converting strings to bytes. Default is 'utf-8'.
    units (str, optional): The units in which to return the estimated time. Default is 'seconds'.
    num_trials (int, optional): The number of trials to run when estimating the time. If not specified, the function will decide based on the type of possible_elements.

    Returns:
    The estimated time to brute force the hashed string, in the specified units.

    Raises:
    ValueError: If hash_algorithm is not a valid hashing algorithm from the hashwise library, or if possible_elements contains repeat values or elements with multiple characters, or if units is not a valid unit of time.
    TypeError: If length is not specified when possible_elements is a collection of characters.
    """
    if hash_algorithm not in _hash_algo_info.keys():
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

def get_all_hash_algorithms():
    """
    This function returns a list of all available hash algorithms.

    Returns:
    A list of all the hash functions avilable in hashwise.
    """
    return list(_hash_algo_info.keys())

_hash_algo_info = {
    blake2b: {'name':'blake2b', 'hash_len':128, "gpu_func":None},
    blake2s: {'name':'blake2s', 'hash_len':64, "gpu_func":None},
    md5: {'name':'md5', 'hash_len':32, "gpu_func":_cudalib.md5_brute_force},
    sha1: {'name':'sha1', 'hash_len':40, "gpu_func":_cudalib.sha1_brute_force},
    sha224: {'name':'sha224', 'hash_len':56, "gpu_func":None},
    sha256: {'name':'sha256', 'hash_len':64, "gpu_func":_cudalib.sha256_brute_force},
    sha384: {'name':'sha384', 'hash_len':96, "gpu_func":None},
    sha512: {'name':'sha512', 'hash_len':128, "gpu_func":None},
    sha3_224: {'name':'sha3_224', 'hash_len':56, "gpu_func":None},
    sha3_256: {'name':'sha3_256', 'hash_len':64, "gpu_func":None},
    sha3_384: {'name':'sha3_384', 'hash_len':96, "gpu_func":None},
    sha3_512: {'name':'sha3_512', 'hash_len':128, "gpu_func":None},
    shake_128: {'name':'shake_128', 'hash_len':128, "gpu_func":None},
    shake_256: {'name':'shake_256', 'hash_len':256, "gpu_func":None},
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
