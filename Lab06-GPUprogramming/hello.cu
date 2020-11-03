#include <stdio.h>

// kernel function
__global__ void cuda_hello() {}

int main()
{
        // calling the kernel
        cuda_hello<<<1,1>>>();
        return 0;
}
