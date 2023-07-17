#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#define uchar unsigned char
#define uint unsigned int
#define ull unsigned long long

#define DEBUG_MODE 0


#define DBL_INT_ADD(a,b,c) if (a > 0xffffffff - (c)) ++b; a += c;
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

typedef struct {
	unsigned char data[64];
	unsigned int datalen;
	unsigned int bitlen[2];
	unsigned int state[8];
} SHA256_CTX;

__device__ void SHA256Transform(SHA256_CTX *ctx, unsigned char data[]) {
	const unsigned int k[64] = {
		0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
		0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
		0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
		0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
		0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
		0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
		0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
		0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
	};
	unsigned int a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for (; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e, f, g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a, b, c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}

__device__ void SHA256Init(SHA256_CTX *ctx) {
	ctx->datalen = 0;
	ctx->bitlen[0] = 0;
	ctx->bitlen[1] = 0;
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

__device__ void SHA256Update(SHA256_CTX *ctx, unsigned char data[], unsigned int len) {
	for (unsigned int i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			SHA256Transform(ctx, ctx->data);
			DBL_INT_ADD(ctx->bitlen[0], ctx->bitlen[1], 512);
			ctx->datalen = 0;
		}
	}
}

__device__ void SHA256Final(SHA256_CTX *ctx, unsigned char hash[]) {
	unsigned int i = ctx->datalen;

	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		SHA256Transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	DBL_INT_ADD(ctx->bitlen[0], ctx->bitlen[1], ctx->datalen * 8);
	ctx->data[63] = ctx->bitlen[0];
	ctx->data[62] = ctx->bitlen[0] >> 8;
	ctx->data[61] = ctx->bitlen[0] >> 16;
	ctx->data[60] = ctx->bitlen[0] >> 24;
	ctx->data[59] = ctx->bitlen[1];
	ctx->data[58] = ctx->bitlen[1] >> 8;
	ctx->data[57] = ctx->bitlen[1] >> 16;
	ctx->data[56] = ctx->bitlen[1] >> 24;
	SHA256Transform(ctx, ctx->data);

	for (i = 0; i < 4; ++i) {
		hash[i] = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}
}

__device__ void cudaStrcpy(char* dst, const char* src, int length) {
	if (length == NULL){
		while ((*dst++ = *src++) != '\0');
	}
	else{
		for (unsigned int i = 0; i < length; i++){dst[i] = src[i];}
	}
}

__device__ void cudaStrcat(char* dst, const char* src) {
    while (*dst) dst++; // find end of dst
    while ((*dst++ = *src++)); // copy src to end of dst
}

__device__ void cudaSprintf(char* dst, unsigned char value) {
    const char hex_map[] = "0123456789abcdef";
    dst[0] = hex_map[value >> 4]; // upper nibble
    dst[1] = hex_map[value & 0x0F]; // lower nibble
    dst[2] = '\0'; // null-terminate the string
}

__device__ int cudaStrcmp(const char *str1, const char *str2, unsigned int length) {
	if (length == 0){
		while (*str1 && (*str1 == *str2)) {
			str1++;
			str2++;
		}
		return *(unsigned char *) str1 - *(unsigned char *) str2;
	}
	else{
		for(unsigned int i = 0; i < length; i++){
			if(*str1 != *str2){
				return *(unsigned char *) str1 - *(unsigned char *) str2;
			}
			str1++;
			str2++;
		}
		return 0;
	}
}
__device__ void SHA256(char* arg, int length, char* res) {
	SHA256_CTX ctx;
	unsigned char hash[32];
	char* hashStr = (char*) malloc(sizeof(char) * 65);
	if (!hashStr) {
		// return NULL; // check for failed malloc
	}
	cudaStrcpy(hashStr, "", NULL);

	SHA256Init(&ctx);
	SHA256Update(&ctx, (unsigned char*) arg, length);
	SHA256Final(&ctx, hash);

	char s[3];
	for (int i = 0; i < 32; i++) {
		cudaSprintf(s, hash[i]);
		cudaStrcat(hashStr, s);
	}

	cudaStrcpy(res, hashStr, 64);
	free(hashStr);
}


__device__ int update_str(char* target, int len_target, char* elements, int len_elements) {
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

__device__ int update_str_multi(char* target, int len_target, char* elements, int len_elements, int num_update) {
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

ull* partition(ull length, int num_threads){
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

__global__ void global_brute_force_sha256(int* len_chars_ptr, char* elements, int* len_elements_ptr, ull* part, const char* target, char* res){
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
	update_str_multi(current, len_chars, elements, len_elements, start);

	if (DEBUG_MODE >= 2){
		printf("Finished updating string for thread <%d>\n", threadID);
	}

    char* checker = (char*) malloc(65 * sizeof(char));
	checker[64] = '\0';
    while (start < end){
        SHA256(current, len_chars, checker);
        if (cudaStrcmp(checker, target, 64) == 0){
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
    __declspec(dllexport) char* sha256_brute_force(int len_chars, char* elements, int len_elements, char* target){
        int numBlocks = 512;
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

        global_brute_force_sha256<<<numBlocks, threadsPerBlock>>>(cuda_len_chars, cuda_elements, cuda_len_elements, cuda_part, cuda_target, cuda_res);
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

