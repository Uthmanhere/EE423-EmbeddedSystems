
# Introduction to Cache Mapping Techniques
## Cache Mapping Techniques
- Direct-Mapped
- Fully Associative
- N-Way Associative
## Direct-Mapped Cache
In a direct-mapped cache, each memory address is associated with only one block in cache. Locations are referred to, by tag, index and offset bits.
- Tag checks if block is correct.
- Index selects the block.
- Word offset within the block.
Whenever an access is issued by core, tag bits are compared for the particular index to know if data exists in the cache.
### Mapping Bits
#### Offset Calculation
Word offset size is related to size of a block. if $N$ bits represent offset, it can be calculated using equation:
```
    N = log2 S_Block
```
where S\_Block is the block size.
#### Index Calculation
Address index size is related to number of cache blocks. if $I$ bits represent offset, it can be calculated using equation:
```
    I = log2 N_Block
```
where N\_Block is the number of cache blocks.
#### Tag Calculation
Number of bits in tag are those of remaining ones and number of $T$ tag bits can be mathematically calculated as:
```
    T = 32 - (I + N)
```
assuming 32 bit architecture.
### Access Patterns
Understanding effects of access patterns on hit \& miss rates is not a trivial task. Please refer to [Task A](#dm_task) for a theoretical discourse and hands-on tutorial.
## Fully Associative Cache
Fully associative cache stores memory blocks required in an access. It refers to address using tag and offset bits.
### Mapping Bits
#### Offset Calculation
Word offset size is related to size of a block. if $N$ bits represent offset, it can be calculated using equation:
```
    N = lg2 S_Block
```
where S\_Block is the block size.
#### Tag Calculation
Number of bits in tag are those of remaining ones and number of $T$ tag bits can be mathematically calculated as:
```
    T = 32 - N
```
assuming 32 bit architecture.
### Access Patterns
Understanding effects of access patterns on hit & miss rates is not a trivial task. Please refer to section [Task B](#fa_task) for a theoretical discourse and hands-on tutorial.
## N-Way Set Associative Cache
Set associative cache helps map $N$ blocks in a set. It is in fact a sweet spot between direct-mapped and fully associative caches.
### Mapping Bits
#### Offset Calculation
Word offset size is related to size of a block. if _O_ bits represent offset, it can be calculated using equation:
```
    O = lg2 S_Block
```
where S\_Block is the block size.
#### Index Calculation
Address index size is related to number of cache blocks. if $I$ bits represent offset, it can be calculated using equation:
```
    I = lg2 N_Block / N
```
where there are $N$ ways and N\_Block is the number of cache blocks.
#### Tag Calculation
Number of bits in tag are those of remaining ones and number of $T$ tag bits can be mathematically calculated as:
```
    T = 32 - (O + N)
```
assuming 32 bit architecture.
### Access Patterns
Understanding effects of access patterns on hit \& miss rates is not a trivial task. Please refer to [TAsk C](#sa_task) for a theoretical discourse and hands-on tutorial.


## Venus RISC-V Simulator
[Venus](https://venus.cs61c.org/) is a RISC-V simulator. You can observe it's features by copying the following code in editor and running it in the simulator. Try to explore Editor and Simulator particularly Registers and Cache sections of the simulator.
### Code Snippet
Refer to code snippet in [cache.s](./cache.s)
### Flowchart
Refer to flowchart representation of code snippet in the following figure.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab07-CacheAndPipelineArch/img/flow.png" width="800">

# Introduction to Pipeline Hazards
Pipepline hazards can be categorized as following:
- Structural Hazards
- Data Hazards
    - RAW
    - WAR
    - WAW
- Control Hazards
## Ripes RISC-V Simulator
Ripes is an open source pipelined RISC-V simulator. The releases can be obtained from [Git Releases](https://github.com/mortbopet/Ripes/releases) page.

Download and double click _AppImage_ to run the simulator. Paste the code snippet in section [Task D](#pipeR1). Simulate it step-by-step in the processor window.

# Lab Tasks
## Task A: Direct-Mapped Cache <a name=dm_task></a>
We shall try to understand cache architecture and it's effects on hit \& miss rates by setting up a reference example.
### Cache Architecture
- Direct-mapped.
- Block size: 16 bytes.
- 4 blocks.
### Mapping Bits
#### Offset Bits
```
    N = lg2 S_Block = lg2 16= 4 bits
```
#### Index bits
```
    I = lg2 N_Block = lg2 4 = 2 bits
```
#### Tag Bits
```
    T = 32 - (I + N) = 32 - (4+2) = 26 bits
```
### Access Pattern I  <a name=dm_a1></a>
Assume the access pattern:
```
    Address = 0, 2, 4, 6, 8, A, C, E, 10, 12, 14, 16, 18, 1A, 1C, 1E, ...
```
#### Access Log in a Cache Perspective
Observe the thorough log of accesses in the following table and try to infer how hits and misses have been induced.

_Hint: Observe index bits._
Address | Tag | Index | Offset | Status
------- | --- | ----- | ------ | ------
0x00 | 0 | 0 | 0 | M
0x02 | 0 | 0 | 2 | H
0x04 | 0 | 0 | 4 | H
0x06 | 0 | 0 | 6 | H
0x08 | 0 | 0 | 8 | H
0x0A | 0 | 0 | A | H
0x0C | 0 | 0 | C | H
0x0E | 0 | 0 | E | H
0x10 | 0 | 1 | 0 | M
0x12 | 0 | 1 | 2 | H
0x14 | 0 | 1 | 4 | H
0x16 | 0 | 1 | 6 | H
0x18 | 0 | 1 | 8 | H
0x1A | 0 | 1 | A | H
0x1C | 0 | 1 | C | H
0x1E | 0 | 1 | E | H
#### Explanation
Access pattern is such that 8 consecutive accesses can be observed before index changes (or a new block is accessed in other words.)
#### Hit Rate
Since we can observe a series of 1 compulsory miss and 7 hits, the hit rate can be conveniently calculated to be:
```
    Rate_Hit = Hits / Accesses = 7/8 = 0.875
```
#### Simulation
Get the code from [cache.s](./cache.s) and copy in Venus. Set step size of 2 bytes to match our access pattern and array size of 256 or 512 bytes. Adjust the cache parameters in simulator:
- Cache levels: 1
- Block size: 16 bytes
- Number of blocks: 4
- Enabled? Green
- Direct-mapped
- LRU, L1
Now run and observe the hit rate. Also add breakpoint (by clicking) over instruction at address _0x18_ to observe each instruction access in cache.
### Access Pattern II <a name="dm_a2"></a>
Assume the access pattern:
```
    Address = 0, 4, 8, C, 10, 14, 18, 1C, 20, 24, 28, 2C, 30, 34, 38, 3C, ...
```
#### Access Log in a Cache Perspective
Observe the thorough log of accesses in the following table and try to infer how hits and misses have been induced.
_Hint: Observe index bits_.
Address | Tag | Index | Offset | Status
------- | --- | ----- | ------ | ------
0x00 | 0 | 0 | 0 | M
0x04 | 0 | 0 | 4 | H
0x08 | 0 | 0 | 8 | H
0x0C | 0 | 0 | C | H
0x10 | 0 | 1 | 0 | M
0x14 | 0 | 1 | 4 | H
0x18 | 0 | 1 | 8 | H
0x1C | 0 | 1 | C | H
0x20 | 0 | 2 | 0 | M
0x24 | 0 | 2 | 4 | H
0x28 | 0 | 2 | 8 | H
0x2C | 0 | 2 | C | H
0x30 | 0 | 3 | 0 | M
0x34 | 0 | 3 | 4 | H
0x38 | 0 | 3 | 8 | H
0x3C | 0 | 3 | C | H
#### Explanation
Access pattern is such that 4 consecutive accesses can be observed before index changes (or a new block is accessed in other words.)
#### Hit Rate
Since we can observe a series of 1 compulsory miss and 3 hits, the hit rate can be conveniently calculated to be:
```
    Rate_Hit = Hits / Accesses = 3/4 = 0.7
```
#### Simulation
Get the code from [cache.s](./cache.s) and copy in Venus. Set step size of 4 bytes to match our access pattern and array size of 256 or 512 bytes. Adjust the cache parameters in simulator:
- Cache levels: 1
- Block size: 16 bytes
- Number of blocks: 4
- Enabled? Green
- Direct-mapped
- LRU, L1
Now run and observe the hit rate. Also add breakpoint (by clicking) over instruction at address _0x18_ to observe each instruction access in cache.
### Access Pattern III <a name="dm_a3"></a>
Assume the access pattern:
```
    Address = 0, 4, 8, C, 10, 14, 18, 1C, 20, 24, 28, 2C, 30, 34, 38, 3C, ...
```
#### Access Log in a Cache Perspective
Observe the thorough log of accesses in the following table and try to infer how hits and misses have been induced.
_Hint: Observe tag bits_.
Address | Tag | Index | Offset | Status
------- | --- | ----- | ------ | ------
0x00 | 0 | 0 | 0 | M
0x10 | 0 | 1 | 0 | M
0x20 | 0 | 2 | 8 | M
0x30 | 0 | 3 | C | M
0x40 | 1 | 0 | 0 | M
0x50 | 1 | 1 | 4 | M
0x60 | 1 | 2 | 8 | M
0x70 | 1 | 3 | C | M
0x80 | 2 | 0 | 0 | M
0x90 | 2 | 1 | 4 | M
0xA0 | 2 | 2 | 8 | M
0xB0 | 2 | 3 | C | M
0xC0 | 3 | 0 | 0 | M
0xD0 | 3 | 1 | 4 | M
0xE0 | 3 | 2 | 8 | M
0xF0 | 3 | 3 | C | M
#### Explanation
Access pattern is such that no accesses can be observed before index changes (or a new block is accessed in other words.)
#### Hit Rate
Since we can observe a series of compulsory misses and no hits, the hit rate can be conveniently calculated to be:
```
    Rate_Hit = Hits / Accesses = 0/n = 0
```
#### Simulation
Get the code from [cache.s](./cache.s) and copy in Venus. Set step size of 16 bytes to match our access pattern and array size of 256 or 512 bytes. Adjust the cache parameters in simulator:
- Cache levels: 1
- Block size: 16 bytes
- Number of blocks: 4
- Enabled? Green
- Direct-mapped
- LRU, L1
Now run and observe the hit rate. Also add breakpoint (by clicking) over instruction at address _0x18_ to observe each instruction access in cache.
### Task
Consider the cache architecture as follows:
- Direct-mapped.
- Block size: 32 bytes.
- 2 blocks.
Calculate the size of mapping bits, evaluate access log in a cache perspective, estimate the hit, simulate and compare for all three access patterns we observed in [Access Pattern II](#dm_a2) and [Access Pattern III](#dm_a3).

## Task B: Fully Associative Cache <a name=fa_task></a>
We shall try to understand cache architecture and it's effects on hit \& miss rates by setting up a reference example.
### Cache Architecture <a name="tb_arch"></a>
- Direct-mapped or fully associative.
- Block size: 16 bytes.
- 4 blocks.
### Task
#### Access Pattern <a name="tb_ap"></a>
Assume the following access pattern in hexadecimals.
```
    Address = 0, 20, 40, 60, 1, 21, 41, 61, 2, 22, 42, 62, 3, 23, 43, 63, ...
```
#### Access Log and Hit Rate <a name="tb_ana"></a>
Compose an access log for misses and hits for each access and predict the hit rate for:
- Directly mapped cache
- Fully associative cache
#### Simulation
Set the appropriate parameters in Venus as:
- Accesses: 4
- Step size: 32
- Repeat count: 4
- Offset: 1
Run the simulation for both architectures and observe the validity of analysis of [Access Pattern](#tb_ana). Increase the repeat count to 8 and 16 to observe how hit rate varies for both architectures.
#### Insight
Try to appreciate significance of cache architectures for different access patterns.

## Task C: N-Way Set Associative Cache <a name="sa_task"></a>
We shall try to understand cache architecture and it's effects on hit \& miss rates by setting up a reference example.
### Cache Architecture <a name="tc_arch"></a>
- Direct-mapped or set associative.
- Block size: 16 bytes.
- 4 blocks.
### Task
#### Access Pattern <a name="tc_ap>"></a>
Assume the following access pattern in hexadecimals.
```
    Address = 0, 40, 2, 42, 4, 44, 6, 46, 8, 48, A, 4A, C, 4C, ...
```
#### Access Log and Hit Rate <a name="tc_ana"></a>
Compose an access log for misses and hits for each access and predict the hit rate for:
- Directly mapped cache
- Fully associative cache
- 2-Way set associative cache
#### Simulation
Set the appropriate parameters in Venus as:
- Accesses: 2
- Step size: 64
- Repeat count: 16
- Offset: 2
Run the simulation for both architectures and observe the validity of analysis of [Access Pattern](#tc_ana). Then change the access pattern to:
```
    Address = 0, 40, 80, C0, 2, 42, 82, C2, 4, 44, 84, C4, 6, 46, 86, C6, ...
```
using parameters:
- Accesses: 4
- Step size: 64
- Repeat count: 16
- Offset: 2
observe the hit and misses along with hit rate of the three architectures.
#### Insight
Try to appreciate significance of access patterns for different cache architectures.

## Task D: Routine I
### Routine <a name=pipeR1></a>
```
  li    t0,10
loop:
  addi  t0,t0,-1
  bne   t0,x0,loop
```
### Task
Analyze the routine for it's pipelined execution in Ripes. Try to populate the following table with instruction number being executed and track the value of _t0_. Locate and identify hazards to answer the following questions:
- Which hazards have been observed?
- How has it been resolved?

No. | IF | ID | EX | MEM | WB | t0 | Hazard
--- | -- | -- | -- | --- | -- | -- | ------
1 |  |  |  |  |  |  |
2 |  |  |  |  |  |  |
3 |  |  |  |  |  |  |
4 |  |  |  |  |  |  |
5 |  |  |  |  |  |  |
6 |  |  |  |  |  |  |
7 |  |  |  |  |  |  |
8 |  |  |  |  |  |  |
9 |  |  |  |  |  |  |
10 |  |  |  |  |  |  |
11 |  |  |  |  |  |  |
12 |  |  |  |  |  |  |
13 |  |  |  |  |  |  |
14 |  |  |  |  |  |  |
15 |  |  |  |  |  |  |
16 |  |  |  |  |  |  |
17 |  |  |  |  |  |  |
18 |  |  |  |  |  |  |
19 |  |  |  |  |  |  |
20 |  |  |  |  |  |  |

## Task E: Routine II
### Routine
```
  addi t0,x0,12
  addi t1,x0,8
  bne  t0,t1,jmp_label
loop:
  addi t0,t0,-1
  bne  t0,x0,loop
jmp_label:
  addi t0,t0,1
```
### Task
Analyze the routine for it's pipelined execution in Ripes. Locate and identify hazards to answer the following questions:
- Which hazards have been observed?
- How has it been resolved?

