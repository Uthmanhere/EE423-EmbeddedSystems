#include <stdio.h>


#define SIZE 64  // 64-by-64 square matrix

__global__ void matMult(int * matProd, int * matA, int * matB)
{
        int row = blockIdx.x;
        int col = threadIdx.x;

        int tmpSum = 0;;
        if (row < SIZE && col < SIZE)
        {
                for (int i=0; i<SIZE; ++i)
                        tmpSum += matA[row*SIZE + i] * matB[i*SIZE + col];
                matProd[row*SIZE + col] = tmpSum;
        }
}

int main()
{
        // initialize, aalocate and define host memory
        int matA[SIZE*SIZE] = { 0 };
        int matB[SIZE*SIZE] = { 0 };
        int matProd[SIZE*SIZE] = { 0 };
        for (int i=0; i<SIZE; ++i)
        {
                for (int j=0; j<SIZE; ++j)
                {
                        matA[i*SIZE + j] = i+j;
                        matB[i*SIZE + j] = i-j;
                }
        }

        // initialize and allocate device memory
        int * dev_matProd, * dev_matA, * dev_matB;
        cudaMalloc((void **)&dev_matA, SIZE*SIZE*sizeof(int));
        cudaMalloc((void **)&dev_matB, SIZE*SIZE*sizeof(int));
        cudaMalloc((void **)&dev_matProd, SIZE*SIZE*sizeof(int));


        // copy data to device memory
        cudaMemcpy((void *)dev_matA, (void *)matA, SIZE*SIZE*sizeof(int),
                        cudaMemcpyHostToDevice);
        cudaMemcpy((void *)dev_matB, (void *)matB, SIZE*SIZE*sizeof(int),
                        cudaMemcpyHostToDevice);

        matMult<<<SIZE,SIZE>>>(dev_matProd, dev_matA, dev_matB);
        
        // check for successful thread execution
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
                printf("Error\n");
                return -1;
        }

        // copy results from device to host memory
        cudaMemcpy(matProd, dev_matProd, SIZE*SIZE*sizeof(int),
                        cudaMemcpyDeviceToHost);


        for (int i=0; i<SIZE/2; ++i)  // inspecting first few diagnols
                printf(" > Diagonal %d of prudect is %d.\n",
                                i, matProd[i*SIZE+i]);

        return 0;
}
