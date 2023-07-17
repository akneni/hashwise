import ctypes

class DeviceStatus():
    __status = []
    __initialized = False
    __device_info_clib = None

    @classmethod
    def __init(cls):
        cls.__initialized = True
        cls.__device_info_clib = ctypes.cdll.LoadLibrary('./device-info/device-info.dll')

        cls.__device_info_clib.num_devices.restype = ctypes.c_int
        cls.__device_info_clib.get_device_compute_capability.restype = ctypes.c_double
        cls.__device_info_clib.get_device_name.restype = ctypes.c_char_p
        cls.__device_info_clib.get_num_blocks.restype = ctypes.c_int
        cls.__device_info_clib.get_max_threads_per_block.restype = ctypes.c_int
    
        if cls.__device_info_clib.num_devices() <= 0: return 

        cls.__status.append({
            "name": ctypes.string_at(cls.__device_info_clib.get_device_name()).decode('utf-8'),
            "compute_capability":cls.__device_info_clib.get_device_compute_capability(),
            'num_blocks':cls.__device_info_clib.get_num_blocks(),
            'threads_per_block':cls.__device_info_clib.get_max_threads_per_block(),
        })

    @classmethod
    def num_devices(cls):
        if not cls.__initialized: cls.__init()
        return max(cls.__device_info_clib.num_devices(), 0)

    @classmethod
    def device_available(cls):
        if not cls.__initialized: cls.__init()
        return len(cls.__status) > 0

    @classmethod
    def device_name(cls, index=0):
        if not cls.__initialized: cls.__init()
        if len(cls.__status) == 0:
            return None
        return cls.__status[index]['name']

    @classmethod
    def device_compute_capability(cls, index=0):
        if not cls.__initialized: cls.__init()
        if len(cls.__status) == 0:
            return None
        return cls.__status[index]['compute_capability']

    @classmethod
    def device_num_blocks(cls, index=0):
        if not cls.__initialized: cls.__init()
        if len(cls.__status) == 0:
            return None
        return cls.__status[index]['num_blocks']

    @classmethod
    def device_threads_per_block(cls, index=0):
        if not cls.__initialized: cls.__init()
        if len(cls.__status) == 0:
            return None
        return cls.__status[index]['threads_per_block']

    @classmethod
    def devices(cls, verbose=0):
        if not cls.__initialized: cls.__init()
        if verbose == 0:
            return ["<{} : {}>".format(device['name'], device['compute_capability']) for device in cls.__status if (device['name'] is not None)]
        elif verbose >= 1:
            return cls.__status