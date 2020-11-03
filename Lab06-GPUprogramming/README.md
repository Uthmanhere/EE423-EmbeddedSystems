# Introduction
## CUDA
Nvidia CUDA is just an extension to C programming language. The primitives particularly introduced by Cuda are discussed with detail in this document.
## CUDA Programming Model
Programming in Cuda itself is heterogeneous in nature. The parallelization offered by GPU is exploited from a CPU. Hence we need to switch in between one and the other for programming GPUs. Following terminology prevails for Cuda in this regard:
- CPU and it's memory as Host
- GPU and it's memory as Device

## Kernel
In CUDA, kernel code is the one that runs on GPU. A kernel function is defined by \_\_global\_\_ keyword in C. Refer to task \ref{sec:task1} for programming your first Cuda kernel.
## Program Flow
Sequential and parallel codes complement each other, prior begin synchronous and latter asynchronous in nature. Refer to figure \ref{fig:progFlow} for a visual understanding.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab06-GPUprogramming/img/progFlow.png" width="400">

Swithcing in between host and device is through memory.
- Host to Device: copy data in host memory to device memory.
- Device to Host: copy data in device memory to host memory.

Modern GPU architectures also offer a unified memory for device and host. Refer to figure \ref{fig:progModel} for the block diagram of programmer's model for GPU programming.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab06-GPUprogramming/img/progModel.png" width="600">

## Memory
### Management
The function definitions for memory management for device are same as that of a CPU code in C except names. Such functions are preceded by CUDA as following:
- cudaMalloc
- cudaMemcpy
- cudaMemset
- cudaFree

### Data Transfer}Data is exchanged by device and host using following CUDA utilities:
- cudaMemcpyHostToDevice
- cudaMemcpyDeviceToHost

The definition of each being equivalent to the following listing:
```C
cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind);
```
Returns _cudaSuccess_ for successful allocation, _cudaErrorMemoryAllocation_ otherwise.
## Layers of Parallelism}
GPU parallelism is exploited at two different layers. These layers along-with their constituents are:
- Grid
    - All threads spawned by a single kernel are collectively called a grid.
    - Grids consists of three-dimensional blocks. Usually only two dimensions are defined.
- Block
    - A group of threads that can cooperate with each other.
    - Blocks consist of three-dimensional threads.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab06-GPUprogramming/img/gridBlock.png" width="800">

## Thread Organization}
When kernels are launched in device, they are assigned to several threads. Organizing threads is the core of CUDA programming. The threads are divided among grids and block.
### Coordinates
A thread is defined by using it's block ID and thread ID. Each of these coordinates is three-dimensional in nature. These IDs are defined within kernel by CUDA itself and can be accessed through structures as:
- blockIdx
    - blockIdx.x
    - blockIdx.y
    - blockIdx.z
- threadIdx
    - threadIdx.x
    - threadIdx.y
    - threadIdx.z
### Dimensoins
- blockDim: measured in threads.
- gridDim: measured in blocks.
## Calling Kernel
Kernels calls use three ankle brackets to define grid and dimension size. The following listing emphasizes the syntax for figure \ref{fig:gridBlock}:
```C
dim3 threadsPerBlock(4, 4); //equivalent to (4, 4, 1)
dim3 numBlocks(2,4);        //equivalent to (2, 4, 1)
kernelFunction<<<numBlocks, threadsPerBlock>>>(arguments ...);
```
For simpler use cases as just one-dimensional grids and block, following syntax shall suffice:
```C
#define numBlocks       10
#define threadsPerBlock 16
kernelFunction<<<numBlocks, threadsPerBlock>>>(arguments ...);
```

## Indexing
Indexing is the concept that helps programmer code a generic kernel for all the threads. We learn this art throughout the lab tasks.

# Lab Tasks
## Task A: The First Cuda Program <a name="task1"></a>
### Syntax}
Cuda launches kernels in GPU. These kernels are defined by \_\_\_global\_\_\_ keyword. Sunch functions can only run on GPU. These are called in angle brackets _<<<x,y>>>_. Significance of this syntax will be explained later.
### Description
The given routine [hello.cu](./hello.cu) is simple enough to elaborate what it does itself. But it does not print anything as this particular functionality is not available in kernels. It is just supposed to elaborate basic Cuda syntax.
### Task
Compile and run the [hello.cu](./hello.cu) and check if it compiles and runs.
## Task B: GPUs & their Properties
### Description
GPU routines are often sensitive to GPU architectures. Such architectural details can be observed in _cudaDeviceProp_ structure. The following routine elaborates the same.
### Task
Compile and run the [prop.cu](./prop.cu) and observe the number of GPUs and their specifications in your system.
## Task C: Parallelizing a Vector Computation
### Vector Compute on CPU} <a name="vec_cpu"></a>
#### Description
Given is a simple vector computation code executing in a sequential manner over a CPU.
### Vector Compute on GPU <a name="vec_gpu"></a>
#### Description
The same functionality as in code of [Vector Comput on CPU](#vec_cpu) has been implemented here. The only difference in [CUDA-enabled code](./vec_gpu.cu) is that it has been parallelized through Cuda-define extensions. Syntax _<<<N,1>>>_ launches as many kernels as elements in the array. _blockIdx.x_ keeps track of thread IDs. Test for same thread ID has been expressed as a good programming practice for debugging.
### Task C-I: Block-level Parallelism
Compile and run the [vec_cpu.cu](./vec_cpu.cu) and [vec_gpu.cu](./vec_gpu.cu); observe the resluts.
### Task C-II Thread-level Parallelism
Change _blockIdx.x_ to _threadIdx.x_ in line _9_ of code in [GPU vector snippet](#vec_gpu). Replace _<<<N,1>>>_ with _<<<1,N>>>_ in line _38_ as well. Compile and execute the code.

Observing the maximum thread dimensions allowed for GPU in properties, are you prompted by the expected result? If not, what reason could have made it possible?

Recommend the maximum thread and block dimensions for optimum parallel processing in GPUs.
### Task C-III: Threads & Blocks Combined
#### Elaboration
Assume a kernel has been called for 3 blocks containing 8 threads each. Such a kernel call is expressed as _<<< 3 , 8 >>>_. The most simplistic indexing technique for this kernel is expressed in the following figure. It can be expressed as equation:
```
	Index = Block ID x Block Size + Thread ID
```
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab06-GPUprogramming/img/threadBlock.png" width="800">

#### Task
Call the kernel with following snippet now:
```C
#define threadsPerBlock 64
compute<<<ceil(N/threadsPerBlock), threadsPerBlcok>>>(dev_a, dev_b, dev_c);
```
and change the indexing technique to
```C
// indexing with block ID and thread Id combined
int i = blockId.x*blockDim.x + threadIdx.x;
```

## Task D: Matrix Multiplication on GPU <a name="sqMat"></a>
### Code Snippet
Refer to code snippet [multSq.cu](./multSq.cu).
### Elaboration
The kernel call is a combination of threads and blocks. Blocks are responsible for indexing rows while threads index columns. Block index and thread index are utilized such that each kernel compute a product matrix constituent. The following figure elaborates phenomenon in quite an elegant manner.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab06-GPUprogramming/img/matMult.png" width="1000">

### Task D-I
Compile the code in [multSq.cu](./multSq.cu) and observe the output. Verify using MATLAB or any other tool possible.

### Task D-II
Evolve the code snippet in [multSq.cu](./multSq.cu) for rectangular matrix multiplication.

