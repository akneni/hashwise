#include <stdlib.h>
#include <cuda_runtime.h>

#define DEBUG_MODE 0


extern "C" {
    __declspec(dllexport) int num_devices(){
        int ph = -1;
        int* res = &ph;
        cudaGetDeviceCount(res);
        return *res;
    }

    __declspec(dllexport) double get_device_compute_capability() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            return deviceProp.major +  (double) deviceProp.minor * 0.1;
        }
        return 0;
    }

    __declspec(dllexport) char* get_device_name() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            char* res = (char*) calloc(100,  sizeof(char));
            strcpy(res, deviceProp.name);
            return res;
        }
        return NULL;
    }

    __declspec(dllexport) int get_num_blocks() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            return deviceProp.multiProcessorCount;
        }
        return 0;
    }

    __declspec(dllexport) int get_max_threads_per_block() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            return deviceProp.maxThreadsPerBlock;
        }
        return 0;
    }
}