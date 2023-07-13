#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

#define DEBUG_MODE 0


extern "C" {
    __declspec(dllexport) char* sha256_bulk(char* chars, int len_chars, int length){

        // To be implemented

        return "NoHashFound";
    }



    __declspec(dllexport) int cuda_enabled(){
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

    __declspec(dllexport) void get_info() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            std::cout << "Device " << i << " has compute capability " << deviceProp.major << "." << deviceProp.minor << "\n";
            std::cout << "Device name: " << deviceProp.name << "\n";
            std::cout << "Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << "\n";
            std::cout << "Number of multiprocessors: " << deviceProp.multiProcessorCount << "\n";
            std::cout << "Total global memory: " << deviceProp.totalGlobalMem << "\n";
            std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock << "\n";
        }
    }
}