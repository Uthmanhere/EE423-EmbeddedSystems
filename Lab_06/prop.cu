#include <stdio.h>

int main()
{
        cudaDeviceProp prop;
        int count;
        
        // count devices found.
        cudaGetDeviceCount( &count );
        
        // print properties of each device.
        for (int i=0; i<count; i++)
        {
                cudaGetDeviceProperties( &prop, i );
                printf("---General device information---\n");
                printf(" > Name: %s\n", prop.name);
                printf(" > Computer capability: %d.%d\n", prop.major, prop.minor);
                printf(" > Clock rate: %d\n", prop.clockRate);
                printf(" > Device copy overlap: %s\n", prop.deviceOverlap ? "Enabled" : "Disabled");
                printf("---Memory Information---\n");
                printf(" > Total global memory: %ld\n", prop.totalGlobalMem);
                printf(" > Total constant memory: %ld\n", prop.totalConstMem);
                printf(" > Max memory pitch: %ld\n", prop.memPitch);
                printf(" > Texture Alignment: %ld\n", prop.textureAlignment);
                printf("---MP Information for device---\n");
                printf(" > Multiprocessor count: %d\n", prop.multiProcessorCount);
                printf(" > Shared memory per MP: %ld\n", prop.sharedMemPerBlock);
                printf(" > Registers per MP: %d\n", prop.regsPerBlock);
                printf(" > Threads in warp: %d\n", prop.warpSize);
                printf(" > Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
                printf(" > Maximum thread dimensions: (%d, %d, %d)\n",
                                prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
                printf(" > Maximum grid dimentions: (%d, %d, %d)\n\n\n",
                                prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        }
}
