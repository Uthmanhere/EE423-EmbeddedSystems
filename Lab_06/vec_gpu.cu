#define N 10000

__global__ void compute( float * a, float * b, float * c )
{
        int tid = blockIdx.x;
        if (tid < N)
                c[tid] = b[tid] / a[tid];
}

int main(void)
{
        float a[N], b[N], c[N];
        float * dev_a, * dev_b, * dev_c;

        cudaMalloc((void **)&dev_a, N*sizeof(int));
        cudaMalloc((void **)&dev_b, N*sizeof(int));
        cudaMalloc((void **)&dev_c, N*sizeof(int));

        for (int i=0; i<N; i++)
        {
                a[i] = cos(i);
                b[i] = sin(i);
        }

        cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

        compute<<<N,1>>>(dev_a, dev_b, dev_c);

        cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

        for (int i=0; i<10; i++)
                printf(">> for i %d ocomputers %f.\n", i, c[i]);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

        return 0;
}
