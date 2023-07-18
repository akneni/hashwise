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
