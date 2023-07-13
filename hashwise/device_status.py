class DeviceStatus():
    status = []

    @classmethod
    def add_device(cls, name=None, compute_capability=None, threads_per_block=None, num_blocks=None):
        cls.status.append({
            "name":name,
            "compute_capability":compute_capability,
            'threads_per_block':threads_per_block,
            'num_blocks':num_blocks,
        })

    @classmethod
    def device_available(cls):
        return len(cls.status) != 0

    @classmethod
    def device_name(cls, index=0):
        if len(cls.status) == 0:
            return None
        return cls.status[index]['name']

    @classmethod
    def device_compute_capability(cls, index=0):
        if len(cls.status) == 0:
            return None
        return cls.status[index]['compute_capability']

    @classmethod
    def device_threads_per_block(cls, index=0):
        if len(cls.status) == 0:
            return None
        return cls.status[index]['threads_per_block']

    @classmethod
    def device_num_blocks(cls, index=0):
        if len(cls.status) == 0:
            return None
        return cls.status[index]['num_blocks']

    @classmethod
    def devices(cls):
        return str(["<{} : {}>".format(device['name'], device['compute_capability']) for device in cls.status if (device['name'] is not None)])