#include <stdio.h>

#define N 10000  // vector size


// kernel function for device
__global__ void compute( float * a, float * b, float * c )
{
        // index by block ID
        int i = blockIdx.x;
        if (i < N)  // check if thread is valid
                c[i] = b[i] / a[i];  // compute tangent
}

int main(void)
{
        // initialize and define host vectors
        float a[N], b[N], c[N];
        
        for (int i=0; i<N; i++)
        {
                a[i] = cos(i);
                b[i] = sin(i);
        }   
        // initialize and allocate device pointers
        float * dev_a, * dev_b, * dev_c;

        cudaMalloc((void **)&dev_a, N*sizeof(float));
        cudaMalloc((void **)&dev_b, N*sizeof(float));
        cudaMalloc((void **)&dev_c, N*sizeof(float));

        // copy data to host memory
        cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

        // call kernel for N bloack, 1 thread each
        compute<<<N,1>>>(dev_a, dev_b, dev_c);
      
        //check for successful thread execution
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
                printf("Error\n");
                return -1;
        }

        // copy results back to host memory
        cudaMemcpy(c, dev_c, N*sizeof(float), cudaMemcpyDeviceToHost);

        // print a few results for observation
        for (int i=0; i<10; i++)
                printf(">> for i %d ocomputes %f.\n", i, c[i]);

        // free device memory
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

        return 0;
}
