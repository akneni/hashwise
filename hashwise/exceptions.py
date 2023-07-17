class GPUNotAccessibleError(Exception):
    """Exception when gpu is not found"""
    pass

class UnknownGPUError(Exception):
    """
    Exception when an unknown gpu error occured. Note, this is temporary an will be
    removed after more testing.
    
    """
    pass

class DependencyNotFoundError(Exception):
    """Exception when a dependency, such as a .dll file, cannot be found"""
    pass