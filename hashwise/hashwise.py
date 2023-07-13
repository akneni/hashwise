import ctypes as _ctypes
from time import perf_counter as _perf_counter
import hashlib as _hashlib
from tqdm import tqdm as _tqdm
import os as _os
from pathlib import Path as _Path

# _current_dir = _Path(_os.path.dirname(_os.path.realpath(__file__)))
# _cudalib_path = _current_dir / "cuda-libraries" / "cudalib.dll"
# cudalib = _ctypes.cdll.LoadLibrary(str(_cudalib_path))

# _current_dir = _Path(_os.path.dirname(_os.path.realpath(__file__)))
# _clib_path = _current_dir / "c-libraries" / "clib.dll"
# clib = _ctypes.cdll.LoadLibrary(str(_clib_path))

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

def brute_force_hash(hash_algorithm, possible_chars, length:int, target:str, string_encoding:str='utf-8', show_progress_bar=False):
    if hash_algorithm not in all_algorithms:
        raise ValueError("Argument 'hash_algorithm' must be a hashing algorithm from hashwise library. See 'hashwise.all_algorithms' for complete list.")

    if show_progress_bar:
        for i in _tqdm(__perm_gen(possible_chars, length), total=(len(possible_chars)**length)):
            byte_obj = bytes(i, encoding=string_encoding)
            if hash_algorithm(byte_obj) == target:
                return i
    else:
        for i in __perm_gen(possible_chars, length):
            byte_obj = bytes(i, encoding=string_encoding)
            if hash_algorithm(byte_obj) == target:
                return i
        
    return None

def brute_force_time_estimate(hash_algorithm, possible_chars, length:int, string_encoding:str='utf-8', units='seconds'):
    if hash_algorithm not in all_algorithms:
        raise ValueError("Argument 'hash_algorithm' must be a hashing algorithm from hashwise library. See 'hashwise.all_algorithms' for complete list.")
    
    unit_lst = {
        'units': ['seconds', 'minutes', 'hours', 'days', 'weeks', 'years', 'decades', 'centuries', 'millennium'],
        'modifiers': [1, 60, 60, 24, 7, 52.14, 10, 10, 10]
    }

    units = units.lower()
    if units not in unit_lst['units']:
        raise ValueError ("Argument 'units' must one of", unit_lst["units"])

    start  =_perf_counter()
    counter = 0
    loop_broken = False
    for i in __perm_gen(possible_chars, length):
        byte_obj = bytes(i, encoding=string_encoding)
        hash_algorithm(byte_obj)
        counter += 1
        if counter >= 10_000:
            loop_broken = True
            break
    
    end = _perf_counter()

    modifier = 1
    for un, mod in zip(unit_lst["units"], unit_lst["modifiers"]):
        modifier *= mod
        if units == un:
            break            

    return (((end-start) / 10_000) * (len(possible_chars) ** length) if loop_broken else (end-start)) / modifier


# def num_gpu():
#     return max(cudalib.cuda_enabled(), 0)


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

def __perm_gen(chars, length):
    if length == 0:
        yield ''
    else:
        for char in chars:
            for perm in __perm_gen(chars, length - 1):
                yield char + perm

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
