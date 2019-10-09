#include <stdio.h>

// kernel function
__global__ void cuda_hello()
{
        // nothing prints as such functionalities
        // are not provided by device
        printf("Hello world from GPU\n");
}

int main()
{
        // calling the kernel
        cuda_hello<<<1,1>>>();
        return 0;
}
