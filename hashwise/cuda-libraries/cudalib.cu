#include <stdlib.h>

#define DEBUG_MODE 0


extern "C" {
    __declspec(dllexport) int cuda_enabled(){
        int ph = -1;
        int* res = &ph;
        cudaGetDeviceCount(res);
        return *res;
    }
}




