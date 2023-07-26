# Hashwise Python Library

This library provides a set of functions for hashing and brute forcing hashes using various algorithms. It leverages CUDA for GPU computation to speed up the process.

## Instilation
`pip install hashwise`

## Dependencies

The library depends on the following Python packages:

- `ctypes`
- `time`
- `hashlib`
- `tqdm`
- `os`
- `pathlib`

It also requires CUDA libraries for GPU computation. The required DLL is expected to be located in the "cuda-libraries" directory relative to the location of the library file.

## Functions

The library provides the following functions:

- `blake2b(payload:bytes)`
- `blake2s(payload:bytes)`
- `md5(payload:bytes)`
- `sha1(payload:bytes)`
- `sha224(payload:bytes)`
- `sha256(payload:bytes)`
- `sha384(payload:bytes)`
- `sha512(payload:bytes)`
- `sha3_224(payload:bytes)`
- `sha3_256(payload:bytes)`
- `sha3_384(payload:bytes)`
- `sha3_512(payload:bytes)`
- `shake_128(payload:bytes, length:int)`
- `shake_256(payload:bytes, length:int)`

These functions take a byte string as input and return the hashed value of the input.

The library also provides the following functions for brute forcing hashes:

- `brute_force_hash(hash_algorithm, possible_elements, target:str, len_permutation:int=None, string_encoding:str='utf-8', use_gpu=None, numBlocks=32, numThreadsPerBlock=32, show_progress_bar=False)`
- `brute_force_time_estimate(hash_algorithm, possible_elements, length:int=None, string_encoding:str='utf-8', units='seconds', num_trials=None)`

The `brute_force_hash` function attempts to find the original value of a hashed string by brute force. The `brute_force_time_estimate` function estimates the time it would take to brute force a hashed string using a specified hash algorithm and possible elements.

## Exceptions

The library defines the following exceptions:

- `DependencyNotFoundError`
- `GPUNotAccessibleError`
- `NotImplementedError`
- `UnknownGPUError`

These exceptions are raised when there are issues with dependencies, GPU access, or unknown errors during computation.

## Usage

In order to see if a CUDA enables GPU is available, call the following:

```python
import hashwise

# Returns a list of all the GPU devices available. An empty list means that no GPUs were found.
hashwise.DeviceStatus.devices()

# Alternatively, this method will return true if a GPU is available and false if otherwise.
hashwise.DeviceStatus.device_available()
```

To use the library, import the required functions and call them with the appropriate arguments. For example:

```python
hashed_value = hashwise.sha256(b'hello world')
original_value = hashwise.brute_force_hash(hashwise.sha256, 'abcdefghijklmnopqrstuvwxyz', hashed_value)
```
This will hash the string 'hello world' using the SHA256 algorithm, and then attempt to find the original value of the hashed string by brute force.

Additionally, we can specify the following parameters to enable GPU acceleration specify the number of blocks and threads to be allocated on the GPU. If these numbers aren't defined, an estimate of the optimal configuration will be used.
```python
original_value = hashwise.brute_force_hash(
    hash_algorithm=hashwise.sha256,
    possible_elements="abcdefghijklmnopqrstuvwxyz",
    target=hashwise.sha256(b'hello world'),
    use_gpu=True,
    numBlocks=512,
    numThreadsPerBlock=64
)
```

