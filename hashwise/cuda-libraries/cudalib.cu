#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>


#include "gen-cuda-func.cu"
#include "sha256.cu"
#include "md5.cu"
#include "sha1.cu"

#define DEBUG_MODE 0

__device__ int update_str(char* target, int len_target, char* elements, int len_elements) {
    // Returns 1 if updated sucessfully, returns 0 if the end of the permutation has been reached or the function was executed incorrectly
    for (int i = len_target-1; i>=0; i--){
        for (int j = 0; j < len_elements; j++){
            if (target[i] == elements[j]) { 
                if (j == len_elements-1) {
                    if (i == 0){
                        return 0;
                    }
                    target[i] = elements[0];
                }
                else{
                    target[i] = elements[j+1];
                    return 1;
                }
            }
        }
    }
    return 0;
}

__device__ int update_str_multi(char* target, int len_target, char* elements, int len_elements, int num_update) {
    // Checks to ensure the length of the element or target does not include the null character
    if (elements[len_elements-1] == '\0'){
        len_elements--;
    }
    if (target[len_target-1] == '\0'){
        len_target--;
    }

    int num_inc;
    for (int i = 0; i < len_target; i++){
        if ((len_target-i-1) == 0){
            num_inc = num_update;
        }
        else{
            num_inc = (int)(num_update / pow(len_elements , (len_target-i-1)));
        }


        if (num_inc != 0){
            for (int j = 0; j < len_elements; j++){
                if (elements[j] == target[i]){
                    if (j + num_inc >= len_elements){
                        num_inc = len_elements-1-j;
                    }
                    target[i] = elements[j + num_inc];
                    num_update -= (int) num_inc * pow(len_elements , (len_target-i-1));
                    break;
                }
            }
        }
    }

    return (int) (num_update == 0);
}

__global__ void global_sha1_brute_force(int* len_chars_ptr, char* elements, int* len_elements_ptr, ull* part, const char* target, char* res){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	int len_chars = *len_chars_ptr;
	int len_elements = *len_elements_ptr;
    
	if (threadID == 0){
		res[len_chars] = '\0';
	}

	long long start = part[threadID];
	long long end = part[threadID+1];

	char* current = (char*) malloc(sizeof(char) * (len_chars+1));

	for (unsigned int i = 0; i < len_chars; i++){
		current[i] = elements[0];
	}
	current[len_chars] = '\0';
	update_str_multi(current, len_chars, elements, len_elements, start);

	if (DEBUG_MODE >= 2){
		printf("Finished updating string for thread <%d>\n", threadID);
	}

    char* checker = (char*) malloc(41 * sizeof(char));
	checker[40] = '\0';
    while (start < end){
        SHA1(current, len_chars, checker);
        if (cudaStrcmp(checker, target, 40) == 0){
            cudaStrcpy(res, current, len_chars);
            break;
        }
        update_str(current, len_chars, elements, len_elements);
        start++;
    }
	if (DEBUG_MODE >= 2){
		printf("Finished execution for thread <%d>, Current = %s\n", threadID, current);
	}

    free(current);
    free(checker);
}


__global__ void global_sha256_brute_force(int* len_chars_ptr, char* elements, int* len_elements_ptr, ull* part, const char* target, char* res){
	int threadID = (blockIdx.x * blockDim.x) + threadIdx.x;

	int len_chars = *len_chars_ptr;
	int len_elements = *len_elements_ptr;

	if (threadID == 0){
		res[len_chars] = '\0';
	}

	unsigned long long start = part[threadID];
	unsigned long long end = part[threadID+1];

	char* current = (char*) malloc(sizeof(char) * (len_chars+1));

	for (int i = 0; i < len_chars; i++){
		current[i] = elements[0];
	}
	current[len_chars] = '\0';
	update_str_multi(current, len_chars, elements, len_elements, start);

	if (DEBUG_MODE >= 2){
		printf("Finished updating string for thread <%d>\n", threadID);
	}
    int update_checker;
    char* checker = (char*) malloc(65 * sizeof(char));
	checker[64] = '\0';
    while (start < end){
        SHA256(current, len_chars, checker);
        if (cudaStrcmp(checker, target, 64) == 0){
            cudaStrcpy(res, current, len_chars);
            break;
        }
        update_checker = update_str(current, len_chars, elements, len_elements);
        if (update_checker != 1){
            break;
        }
        start++;
    }
	if (DEBUG_MODE >= 2){
		printf("Finished execution for thread <%d>, Current = %s\n", threadID, current);
	}

    free(current);
    free(checker);
}

__global__ void global_md5_brute_force(int* len_chars_ptr, char* elements, int* len_elements_ptr, ull* part, const char* target, char* res){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID == 0){
		res[32] = '\0';
	}

	int len_chars = *len_chars_ptr;
	int len_elements = *len_elements_ptr;
	long long start = part[threadID];
	long long end = part[threadID+1];

	char* current = (char*) malloc(sizeof(char) * (len_chars+1));

	for (int i = 0; i < len_chars; i++){
		current[i] = elements[0];
	}
	current[len_chars] = '\0';
	update_str_multi(current, len_chars, elements, len_elements, start);

	if (DEBUG_MODE >= 2){
		printf("Finished updating string for thread <%d>\n", threadID);
	}

    char* checker = (char*) malloc(33 * sizeof(char));
	checker[32] = '\0';
    while (start < end){
        MD5(current, len_chars, checker);
        if (cudaStrcmp(checker, target, 32) == 0){
            cudaStrcpy(res, current, len_chars);
            break;
        }
        update_str(current, len_chars, elements, len_elements);
        start++;
    }
	if (DEBUG_MODE >= 2){
		printf("Finished execution for thread <%d>, Current = %s\n", threadID, current);
	}

    free(current);
    free(checker);
}

extern "C" {
	__declspec(dllexport)  ull* partition(ull length, int num_threads){
		ull threads = (ull) num_threads;
		ull* res = (ull*) calloc(1+num_threads, sizeof(long long));

		ull dif = length / threads;
		ull rem = length % threads;

		res[0] = 0;

		for (int i = 1; i <= num_threads; i++){
			res[i] = res[i-1] + dif + (i <= rem ? 1 : 0);
		}
		res[num_threads] = length;

		return res;
	}

    __declspec(dllexport) char* sha1_brute_force(int len_chars, char* elements, int len_elements, char* target, int numBlocks, int threadsPerBlock){  
        int totalThreads = numBlocks * threadsPerBlock;

        int* cuda_len_chars;
        cudaMalloc(&cuda_len_chars, sizeof(int));
        cudaMemcpy(cuda_len_chars, &len_chars, sizeof(int), cudaMemcpyHostToDevice);

        char* cuda_elements;
        cudaMalloc(&cuda_elements, sizeof(char)*len_elements);
        cudaMemcpy(cuda_elements, elements, sizeof(char)*len_elements, cudaMemcpyHostToDevice);

        int* cuda_len_elements;
        cudaMalloc(&cuda_len_elements, sizeof(int));
        cudaMemcpy(cuda_len_elements, &len_elements, sizeof(int), cudaMemcpyHostToDevice);

        ull* part = partition((ull) pow(len_elements, len_chars), totalThreads);
        ull* cuda_part;
        cudaMalloc(&cuda_part, sizeof(ull)*(totalThreads+1));
        cudaMemcpy(cuda_part, part, sizeof(ull)*(totalThreads+1), cudaMemcpyHostToDevice);

        if (DEBUG_MODE >= 1){
            printf("partition: {");
            for (int i = 0; i < totalThreads; i++){
                printf("%d, ", (int) part[i]);
            }
            printf("%d}\n", (int) part[totalThreads]);
        }
        free(part);

        char* cuda_target;
        cudaMalloc(&cuda_target, sizeof(char)*41);
        cudaMemcpy(cuda_target, target, sizeof(char)*41, cudaMemcpyHostToDevice);


        char* res_init = (char*) malloc((len_chars + 1) * sizeof(char));
        if (res_init == NULL && DEBUG_MODE >= 1){printf("failed to allocate memory for res_init.");}
        for (int i = 0; i < len_chars; i++){
            res_init[i] = 'z';
        }
        res_init[len_chars] = '\0';
        if (DEBUG_MODE >= 1){printf("res_init: %s\n", res_init);}
        char* cuda_res;
        cudaMalloc((void**) &cuda_res, sizeof(char) * (len_chars+1));
        cudaMemcpy(cuda_res, res_init, sizeof(char) * (len_chars+1), cudaMemcpyHostToDevice);
        free(res_init);

        if (DEBUG_MODE >= 1){printf("Starting gpu hashing\n");}

        global_sha1_brute_force<<<numBlocks, threadsPerBlock>>>(cuda_len_chars, cuda_elements, cuda_len_elements, cuda_part, cuda_target, cuda_res);
        cudaDeviceSynchronize();

        cudaFree(cuda_len_chars);
        cudaFree(cuda_elements);
        cudaFree(cuda_len_elements);
        cudaFree(cuda_part);
        cudaFree(cuda_target);

        char* res = (char*) malloc((len_chars+1)*sizeof(char));
        cudaMemcpy(res, cuda_res, sizeof(char)*len_chars, cudaMemcpyDeviceToHost);
        cudaFree(cuda_res);
        res[len_chars] = '\0';

        return res;
    }

    __declspec(dllexport) char* sha256_brute_force(int len_chars, char* elements, int len_elements, char* target, int numBlocks, int threadsPerBlock){ 

        sha256_initialize_cuda_global_vars();

        int totalThreads = numBlocks * threadsPerBlock;

        int* cuda_len_chars;
        cudaMalloc(&cuda_len_chars, sizeof(int));
        cudaMemcpy(cuda_len_chars, &len_chars, sizeof(int), cudaMemcpyHostToDevice);

        char* cuda_elements;
        cudaMalloc(&cuda_elements, sizeof(char)*len_elements);
        cudaMemcpy(cuda_elements, elements, sizeof(char)*len_elements, cudaMemcpyHostToDevice);

        int* cuda_len_elements;
        cudaMalloc(&cuda_len_elements, sizeof(int));
        cudaMemcpy(cuda_len_elements, &len_elements, sizeof(int), cudaMemcpyHostToDevice);

        ull* part = partition((ull) pow(len_elements, len_chars), totalThreads);
        ull* cuda_part;
        cudaMalloc(&cuda_part, sizeof(ull)*(totalThreads+1));
        cudaMemcpy(cuda_part, part, sizeof(ull)*(totalThreads+1), cudaMemcpyHostToDevice);

        if (DEBUG_MODE >= 1){
            printf("partition: {");
            for (int i = 0; i < totalThreads; i++){
                printf("%d, ", (int) part[i]);
            }
            printf("%d}\n", (int) part[totalThreads]);
        }
        free(part);

        char* cuda_target;
        cudaMalloc(&cuda_target, sizeof(char)*65);
        cudaMemcpy(cuda_target, target, sizeof(char)*65, cudaMemcpyHostToDevice);


        char* res_init = (char*) malloc((len_chars + 1) * sizeof(char));
        if (res_init == NULL && DEBUG_MODE >= 1){printf("failed to allocate memory for res_init.");}
        for (int i = 0; i < len_chars; i++){
            res_init[i] = '\0';
        }
        res_init[len_chars] = '\0';
        if (DEBUG_MODE >= 1){printf("res_init: %s\n", res_init);}
        char* cuda_res;
        cudaMalloc((void**) &cuda_res, sizeof(char) * (len_chars+1));
        cudaMemcpy(cuda_res, res_init, sizeof(char) * (len_chars+1), cudaMemcpyHostToDevice);
        free(res_init);

        if (DEBUG_MODE >= 1){printf("Starting gpu hashing\n");}

        global_sha256_brute_force<<<numBlocks, threadsPerBlock>>>(cuda_len_chars, cuda_elements, cuda_len_elements, cuda_part, cuda_target, cuda_res);
        cudaDeviceSynchronize();

        cudaFree(cuda_len_chars);
        cudaFree(cuda_elements);
        cudaFree(cuda_len_elements);
        cudaFree(cuda_part);
        cudaFree(cuda_target);

        char* res = (char*) malloc((len_chars+1)*sizeof(char));
        cudaMemcpy(res, cuda_res, sizeof(char)*len_chars, cudaMemcpyDeviceToHost);
        cudaFree(cuda_res);
        res[len_chars] = '\0';

        return res;
    }

    __declspec(dllexport) char* md5_brute_force(int len_chars, char* elements, int len_elements, char* target, int numBlocks, int threadsPerBlock){  
        int totalThreads = numBlocks * threadsPerBlock;

        md5_initialize_cuda_global_vars();

        int* cuda_len_chars;
        cudaMalloc(&cuda_len_chars, sizeof(int));
        cudaMemcpy(cuda_len_chars, &len_chars, sizeof(int), cudaMemcpyHostToDevice);

        char* cuda_elements;
        cudaMalloc(&cuda_elements, sizeof(char)*len_elements);
        cudaMemcpy(cuda_elements, elements, sizeof(char)*len_elements, cudaMemcpyHostToDevice);

        int* cuda_len_elements;
        cudaMalloc(&cuda_len_elements, sizeof(int));
        cudaMemcpy(cuda_len_elements, &len_elements, sizeof(int), cudaMemcpyHostToDevice);

        ull* part = partition((ull) pow(len_elements, len_chars), totalThreads);
        ull* cuda_part;
        cudaMalloc(&cuda_part, sizeof(ull)*(totalThreads+1));
        cudaMemcpy(cuda_part, part, sizeof(ull)*(totalThreads+1), cudaMemcpyHostToDevice);

        if (DEBUG_MODE >= 1){
            printf("partition: {");
            for (int i = 0; i < totalThreads; i++){
                printf("%d, ", (int) part[i]);
            }
            printf("%d}\n", (int) part[totalThreads]);
        }
        free(part);

        char* cuda_target;
        cudaMalloc(&cuda_target, sizeof(char)*33);
        cudaMemcpy(cuda_target, target, sizeof(char)*33, cudaMemcpyHostToDevice);


        char* res_init = (char*) malloc((len_chars + 1) * sizeof(char));
        if (res_init == NULL && DEBUG_MODE >= 1){printf("failed to allocate memory for res_init.");}
        for (int i = 0; i < len_chars; i++){
            res_init[i] = '\0';
        }
        res_init[len_chars] = '\0';
        if (DEBUG_MODE >= 1){printf("res_init: %s\n", res_init);}

        char* cuda_res;
        cudaMalloc((void**) &cuda_res, sizeof(char) * (len_chars+1));
        cudaMemcpy(cuda_res, res_init, sizeof(char) * (len_chars+1), cudaMemcpyHostToDevice);
        free(res_init);

        if (DEBUG_MODE >= 1){printf("Starting gpu hashing\n");}

        global_md5_brute_force<<<numBlocks, threadsPerBlock>>>(cuda_len_chars, cuda_elements, cuda_len_elements, cuda_part, cuda_target, cuda_res);
        cudaDeviceSynchronize();

        cudaFree(cuda_len_chars);
        cudaFree(cuda_elements);
        cudaFree(cuda_len_elements);
        cudaFree(cuda_part);
        cudaFree(cuda_target);

        char* res = (char*) malloc((len_chars+1)*sizeof(char));
        cudaMemcpy(res, cuda_res, sizeof(char)*len_chars, cudaMemcpyDeviceToHost);
        cudaFree(cuda_res);
        res[len_chars] = '\0';

        return res;
    }
}

#include <iostream>
#include <time.h>
int main(){

    char* hash_sha256 = "406e409a273a55a1d66938548120ebad2cfc5600e43bf8f3a1f0294e4ce4bf26";
    char* hash_sha1 = "cb5b2b5c44385482720385ef4d7be4842fb9b588";
    char* hash_md5 = "7368173d5f2c16cfbbc9e7af8e3ccb8e";

    char* elements = "0123456789abcdefghijlmnopqrstuvwxyz";
    int num_blocks;
    std::cout << "Enter the number of blocks: ";
    std::cin >> num_blocks;


    time_t start = clock();
    // char* res = sha256_brute_force(6, elements, 14, hash_sha256, num_blocks, 32);
    // char* res = sha1_brute_force(6, elements, 25, hash_sha1, num_blocks, 32);
    char* res = md5_brute_force(6, elements, 10, hash_md5, num_blocks, 32);
    time_t end = clock();

    std::cout << res << "\n";
    std::cout << "Time slapsed: " << (double) (end-start) / CLOCKS_PER_SEC << " seconds.\n";


    
    
    
    
    
    
    
    
    
    // cudaDeviceProp prop;
    // int count;

    // cudaGetDeviceCount(&count);

    // for (int i = 0; i < count; i++) {
    //     cudaGetDeviceProperties(&prop, i);
    //     std::cout << "Device Number: " << i << std::endl;
    //     std::cout << "  Device name: " << prop.name << std::endl;
    //     std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
    //     std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    //     std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
    //     std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    //     std::cout << "  Max threads dim: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    //     std::cout << "  Max grid size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    //     std::cout << "  Total global mem: " << prop.totalGlobalMem << std::endl;
    //     std::cout << "  Warp size: " << prop.warpSize << std::endl;
    //     std::cout << "  Max pitch: " << prop.memPitch << std::endl;
    //     std::cout << "  Clock rate: " << prop.clockRate << std::endl;
    //     std::cout << "  Total const mem: " << prop.totalConstMem << std::endl;
    //     std::cout << "  Device overlap: " << prop.deviceOverlap << std::endl;
    //     std::cout << "  Multi-processor count: " << prop.multiProcessorCount << std::endl;
    //     std::cout << "  Kernel execution timeout: " << prop.kernelExecTimeoutEnabled << std::endl;
    //     std::cout << "  Integrated: " << prop.integrated << std::endl;
    //     std::cout << "  Can map host memory: " << prop.canMapHostMemory << std::endl;
    //     std::cout << "  Compute mode: " << prop.computeMode << std::endl;
    //     std::cout << "  Concurrent kernels: " << prop.concurrentKernels << std::endl;
    //     std::cout << "  ECC enabled: " << prop.ECCEnabled << std::endl;
    //     std::cout << "  PCI bus ID: " << prop.pciBusID << std::endl;
    //     std::cout << "  PCI device ID: " << prop.pciDeviceID << std::endl;
    //     std::cout << "  PCI domain ID: " << prop.pciDomainID << std::endl;
    //     std::cout << "  TCC driver: " << prop.tccDriver << std::endl;
    //     std::cout << "  Async engine count: " << prop.asyncEngineCount << std::endl;
    //     std::cout << "  Unified addressing: " << prop.unifiedAddressing << std::endl;
    //     std::cout << "  Memory clock rate: " << prop.memoryClockRate << std::endl;
    //     std::cout << "  Memory bus width: " << prop.memoryBusWidth << std::endl;
    //     std::cout << "  L2 cache size: " << prop.l2CacheSize << std::endl;
    //     std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    //     std::cout << "  Stream priorities supported: " << prop.streamPrioritiesSupported << std::endl;
    //     std::cout << "  Global L1 cache supported: " << prop.globalL1CacheSupported << std::endl;
    //     std::cout << "  Local L1 cache supported: " << prop.localL1CacheSupported << std::endl;
    //     std::cout << "  Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
    //     std::cout << "  Registers per multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    //     std::cout << "  Managed memory: " << prop.managedMemory << std::endl;
    //     std::cout << "  Is multi GPU board: " << prop.isMultiGpuBoard << std::endl;
    //     std::cout << "  Multi GPU board group ID: " << prop.multiGpuBoardGroupID << std::endl;
    //     std::cout << std::endl;
    // }
    return 0;
}

