#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "sha256-device.cu"
#include "sha256-host.cu"

__device__ int update_str_device(char* target, int len_target, char* elements, int len_elements) {
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
    return -1;
}

__device__ int update_str_multi_device(char* target, int len_target, char* elements, int len_elements, int num_update) {
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
                    if (j + num_inc > len_elements-1){
                        num_inc = len_elements-1-j;
                    }
                    target[i] = elements[j + num_inc];
                    num_update -= (int) num_inc * pow(len_elements , (len_target-i-1));
                    break;
                }
            }
        }
    }
    if (num_update > 0){
        return 0;
    }
    else if (num_update == 0){
        return 1;
    }

    return num_update;
}

__global__ void global_sha256_brute_force(int* len_chars_ptr, char* elements, int* len_elements_ptr, ull* part, const char* target, char* res){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID == 0){
		res[64] = '\0';
	}

	int len_chars = *len_chars_ptr;
	int len_elements = *len_elements_ptr;
	long long start = part[threadID];
	long long end = part[threadID+1];

	char* current = (char*) malloc(sizeof(char) * (len_chars+1));

	for (unsigned int i = 0; i < len_chars; i++){
		current[i] = elements[0];
	}
	current[len_chars] = '\0';
	update_str_multi_device(current, len_chars, elements, len_elements, start);

	if (DEBUG_MODE >= 2){
		printf("Finished updating string for thread <%d>\n", threadID);
	}

    char* checker = (char*) malloc(65 * sizeof(char));
	checker[64] = '\0';
    while (start < end){
        SHA256_device(current, len_chars, checker);
        if (cudaStrcmp(checker, target, 64) == 0){
            cudaStrcpy(res, current, len_chars);
            break;
        }
        update_str_device(current, len_chars, elements, len_elements);
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
		ull* res = (ull*) calloc(1+threads, sizeof(long long));

		ull dif = length / threads;
		ull rem = length % threads;

		res[0] = 0;

		for (int i = 1; i <= num_threads; i++){
			res[i] = res[i-1] + dif + (i <= rem ? 1 : 0);
		}
		res[num_threads] = length;

		return res;
	}

    __declspec(dllexport) char* sha256_brute_force(int len_chars, char* elements, int len_elements, char* target){
        int numBlocks = 32;
        int threadsPerBlock = 32;   
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
        free(part);

        if (DEBUG_MODE >= 1){
            printf("partition: {");
            for (int i = 0; i < totalThreads; i++){
                printf("%d, ", (int) part[i]);
            }
            printf("%d}\n", (int) part[totalThreads]);
        }

        char* cuda_target;
        cudaMalloc(&cuda_target, sizeof(char)*65);
        cudaMemcpy(cuda_target, target, sizeof(char)*65, cudaMemcpyHostToDevice);


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
}

