# Introduction
## Direct Memory Access (DMA)
Direct memory access (DMA) is used in order to provide high-speed data transfer between peripherals and memory and between memory and memory. Data can be quickly moved by
DMA without any CPU action. This keeps CPU resources free for other operations.
The DMA controller combines a powerful dual AHB (Advanced High-performance Bus) master bus 
architecture with independent FIFO to optimize the bandwidth of the system, based on a complex
bus matrix architecture. 
The two DMA controllers have 16 streams in total (8 for each controller), each dedicated to
managing memory access requests from one or more peripherals. Each stream can have
up to 8 channels (requests) in total. And each has an arbiter for handling the priority
between DMA requests. 
 
# DMA Functional Description
## General Description
The DMA controller performs direct memory transfer: as an AHB master, it can take the
control of the AHB bus matrix to initiate AHB transactions. It can carry out the following transactions:
* peripheral-to-memory
* memory-to-peripheral
* memory-to-memory
The DMA controller provides two AHB master ports: the AHB memory port, intended to be connected to memories and the AHB peripheral port, intended to be connected to peripherals. However, to allow memory-to-memory transfers, the AHB peripheral port must also have access to the memories. The AHB slave port is used to program the DMA controller (it supports only 32-bit accesses).

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab04-DirectMemoryAccess/img/dma_block.PNG" width="600"/>

## DMA Transactions
A DMA transaction consists of a sequence of a given number of data transfers. The number
of data items to be transferred and their width (8-bit, 16-bit or 32-bit) are software programmable. Each DMA transfer consists of three operations:
* A loading from the peripheral data register or a location in memory, addressed through
the _DMA_SxPAR_ or _DMA_SxM0AR_ register.
* A storage of the data loaded to the peripheral data register or a location in memory, addressed through the _DMA_SxPAR_ or _DMA_SxM0AR_ register.
* A post-decrement of the _DMA_SxNDTR_ register that contains the number of transactions that still have to be performed.

## Channel Selection
Each stream is associated with a DMA request that can be selected out of 8 possible channel requests. The selection is controlled by the _CHSEL[2:0]_ bits in the _DMA_SxCR_ register. The 8 requests from the peripherals (TIM, ADC, SPI, I2C, etc.) are independently connected to each channel and their connection depends on the product implementation. The DMA request mappings are present in the reference manual (Tables 42 and 43). (For our purpose in this lab, ADC 1 is mapped to Stream 0, Channel 0 of DMA 2).

## Arbiter
An arbiter manages the 8 DMA stream requests based on their priority for each of the two AHB master ports (memory and peripheral ports) and launches the peripheral/memory access sequences. Priorities are managed in two stages:
* __Software__: Each stream priority can be configured in the _DMA_SxCR_ register. There are four levels:
  * Very high priority
  * High priority
  * Medium priority
  * Low priority
* __Hardware__: If two requests have the same software priority level, the stream with the lower number takes priority over the stream with the higher number. For example, Stream 2 takes priority over Stream 4.

## DMA Streams
Each of the 8 DMA controller streams provides a unidirectional transfer link between a
source and a destination. Each stream can be configured to perform:
* Regular type transactions: memory-to-peripherals, peripherals-to-memory or memory-to-memory transfers
* Double-buffer type transactions: double buffer transfers using two memory pointers for the memory (while the DMA is reading/writing from/to a buffer, the application can write/read to/from the other buffer).
The amount of data to be transferred (up to 65535) is programmable and related to the
source width of the peripheral that requests the DMA transfer connected to the peripheral
AHB port. The register that contains the amount of data items to be transferred is
decremented after each transaction.

## Source, Destination and Transfer modes
Both source and destination transfers can address peripherals and memories in the entire
4 GB area, at addresses comprised between 0x0000 0000 and 0xFFFF FFFF. The direction is configured using the DIR[1:0] bits in the _DMA_SxCR_ register and offers
three possibilities: memory-to-peripheral, peripheral-to-memory or memory-to-memory
transfers.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab04-DirectMemoryAccess/img/sd_addr.PNG" width="600"/>

When the data width (programmed in the PSIZE or MSIZE bits in the _DMA_SxCR_ register)
is a half-word or a word, respectively, the peripheral or memory address written into the
_DMA_SxPAR_ or _DMA_SxM0AR/M1AR_ registers has to be aligned on a word or half-word
address boundary, respectively.

## Pointer Incrementation
Peripheral and memory pointers can optionally be automatically post-incremented or kept
constant after each transfer depending on the PINC and MINC bits in the _DMA_SxCR_
register.
Disabling the Increment mode is useful when the peripheral source or destination data are accessed through a single register.
If the Increment mode is enabled, the address of the next transfer will be the address of the
previous one incremented by 1 (for bytes), 2 (for half-words) or 4 (for words) depending on
the data width programmed in the PSIZE or MSIZE bits in the _DMA_SxCR_ register.

## Circular Mode
The circular mode is available to handle circular buffers and continuous data flows (e.g.
ADC scan mode). This feature can be enabled using the CIRC bit in the _DMA_SxCR_
register.
When the circular mode is activated, the number of data items (in _DMA_SxNDTR_ register) to be transferred is automatically reloaded with the initial value programmed \\during the stream configuration phase, and the DMA requests continue to be served.
In the normal (non-circular) mode, once the DMA\_SxNDTR register reaches zero, the stream is disabled (the EN bit in the _DMA_SxCR_ register is then equal to 0).

## Double Buffer Mode
This mode is available for all the DMA1 and DMA2 streams. It is enabled by setting the DBM bit in the _DMA_SxCR_ register.
A double-buffer stream works as a regular (single buffer) stream with the difference that it
has two memory pointers. When the Double buffer mode is enabled, the Circular mode is
automatically enabled (CIRC bit in _DMA_SxCR_ is don’t care) and at each end of transaction,
the memory pointers are swapped.
In this mode, the DMA controller swaps from one memory target to another at each end of
transaction. This allows the software to process one memory area while the second memory
area is being filled/used by the DMA transfer. This mode of operation is known as __ping-pong__ mode. The double-buffer stream can work in both
directions (the memory can be either the source or the destination).

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab04-DirectMemoryAccess/img/sd_addr_db.PNG" widt="800"/>

## Programmable Data Width
The number of data items to be transferred has to be programmed into _DMA_SxNDTR_
(number of data items to transfer bit, NDT) before enabling the stream (except when the
flow controller is the peripheral, PFCTRL bit in _DMA_SxCR_ is set).
When using the internal FIFO, the data widths of the source and destination data are
programmable through the PSIZE and MSIZE bits in the DMA\_SxCR register (can be 8-,
16- or 32-bit).
PSIZE, MSIZE and NDT[15:0] have to be configured so as to ensure that the last transfer
will not be incomplete. This can occur when the data width of the peripheral port (PSIZE
bits) is lower than the data width of the memory port (MSIZE bits). This constraint is
summarized in the table below.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab04-DirectMemoryAccess/img/p_width.PNG" width="600"/>

## Single and Burst Transfers
The DMA controller can generate single transfers or incremental burst transfers of 4, 8 or 16
beats. The size of the burst is configured by software independently for the two AHB ports by using the MBURST[1:0] and PBURST[1:0] bits in the _DMA_SxCR_ register. The burst size indicates the number of beats in the burst, not the number of bytes
transferred.

## FIFO
### FIFO Structure
The FIFO is used to temporarily store data coming from the source before transmitting them
to the destination. Each stream has an independent 4-word FIFO and the threshold level is software configurable between 1/4, 1/2, 3/4 or full.
To enable the use of the FIFO threshold level, the direct mode must be disabled by \\ setting
the DMDIS bit in the DMA\_SxFCR register.
### FIFO threshold and burst configuration
Caution is required when choosing the FIFO threshold (bits FTH[1:0] of the _DMA_SxFCR_
register) and the size of the memory burst (MBURST[1:0] of the _DMA_SxCR_ register): The
content pointed by the FIFO threshold must exactly match to an integer number of memory
burst transfers. If this is not in the case, a FIFO error (flag FEIFx of the _DMA_HISR_ or
_DMA_LISR_ register) will be generated when the stream is enabled, then the stream will be
automatically disabled.
In all cases, the burst size multiplied by the data size must not exceed the FIFO size (data
size can be: 1 (byte), 2 (half-word) or 4 (word)).
### FIFO flush
The FIFO can be flushed when the stream is disabled by resetting the EN bit in the
_DMA_SxCR_ register and when the stream is configured to manage peripheral-to-memory or
memory-to-memory transfers: If some data are still present in the FIFO when the stream is
disabled, the DMA controller continues transferring the remaining data to the destination
(even though stream is effectively disabled). When this flush is completed, the transfer
complete status bit (_TCIFx_) in the _DMA_LISR_ or _DMA_HISR_ register is set.
### Direct mode
By default, the FIFO operates in direct mode (_DMDIS_ bit in the _DMA_SxFCR_ is reset) and
the FIFO threshold level is not used. This mode is useful when the system requires an
immediate and single transfer to or from the memory after each DMA request.
When the DMA is configured in direct mode (FIFO disabled), to transfer data in memory-to-peripheral mode, the DMA preloads one data from the memory to the internal FIFO to
ensure an immediate data transfer as soon as a DMA request is triggered by a peripheral.
Direct mode must not be used when implementing memory-to-memory transfers.

## DMA Transfer Completion
Different events can generate an end of transfer by setting the TCIFx bit in the DMA\_LISR
or _DMA_HISR_ status register:
* In DMA flow controller mode:
  - The _DMA_SxNDTR_ counter has reached zero in the memory-to-peripheral mode.
  - The stream is disabled before the end of transfer (by clearing the EN bit in the _DMA_SxCR_ register) and (when transfers are peripheral-to-memory or memory-to-memory) all the remaining data have been flushed from the FIFO into the memory.
* In Peripheral flow controller mode: 
  - The last external burst or single request has been generated from the peripheral and (when the DMA is operating in peripheral-to-memory mode) the remaining data have been transferred from the FIFO into the memory.
  - The stream is disabled by software, and (when the DMA is operating in peripheral-to-memory mode) the remaining data have been transferred from the FIFO into the memory.

## Flow Controller
The entity that controls the number of data to be transferred is known as the flow controller.
This flow controller is configured independently for each stream using the PFCTRL bit in the
_DMA_SxCR_ register. The flow controller can be:
* The DMA controller: in this case, the number of data items to be transferred is
programmed by software into the DMA\_SxNDTR register before the DMA stream is
enabled.
* The peripheral source or destination: this is the case when the number of data items to
be transferred is unknown. The peripheral indicates by hardware to the DMA controller
when the last data are being transferred. This feature is only supported for peripherals
which are able to signal the end of the transfer.

If the stream is configured in noncircular mode, after the end of the transfer (that is when the
number of data to be transferred reaches zero), the DMA is stopped (EN bit in _DMA_SxCR_
register is cleared by Hardware) and no DMA request is served unless the software
reprograms the stream and re-enables it (by setting the EN bit in the _DMA_SxCR_ register).

## DMA Interrupts
For each DMA stream, an interrupt can be produced on the following events:
* Half-transfer reached
* Transfer complete
* Transfer error
* Fifo error (overrun, underrun or FIFO level error)
* Direct mode error

Separate interrupt enable control bits are available for flexibility as shown in the table below.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab04-DirectMemoryAccess/img/dma_intr.PNG" width="600" />

## DMA Registers
DMA uses the following registers (details in section 10.5 of reference manual) that are accessed by 32-bit words.
- DMA low interrupt status register (DMA\_LISR)

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab04-DirectMemoryAccess/img/dma_lisr.PNG" width="800"/>

- DMA high interrupt status register (DMA\_HISR)
- DMA low interrupt flag clear register (DMA\_LIFCR): Writing 1 to a bit clears the corresponding flag in the DMA\_LISR register.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab04-DirectMemoryAccess/img/dma_licfr.PNG" width="800"/>

-DMA high interrupt flag clear register (DMA\_HIFCR)
- DMA stream x configuration register (DMA\_SxCR) (x = 0..7): This register is used to configure the concerned stream.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab04-DirectMemoryAccess/img/reg.PNG" width="800"/>

- DMA stream x number of data register (DMA\_SxNDTR) (x = 0..7)
- DMA stream x peripheral address register (DMA\_SxPAR) (x = 0..7)
- DMA stream x memory 0 address register (DMA\_SxM0AR) (x = 0..7)
- DMA stream x memory 1 address register (DMA\_SxM1AR) (x = 0..7)
- DMA stream x FIFO control register (DMA\_SxFCR) (x = 0..7)

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab04-DirectMemoryAccess/img/fifo.PNG" width="800"/>

# Lab Tasks

## Task A: Multi-Channel ADC using DMA
Copy the code in [dma.c](./dma.c) to operate an ADC with multiple analog inputs. An ADC has 16 input channels. However, each ADC has just one output data register. This means that the register can hold a value from just one channel at a time, and taking an input value from another channel will over-write the data register. One approach can be to use the ADC in single conversion mode for one channel, convert the input value, read the output data register and then start the ADC all over again for another channel. This approach is inefficient as it requires a lot of CPU cycles. Therefore, we will use DMA for the multi-channel ADC, so as to accommodate values from all channels simultaneously. \\ \\
Add variable _adcVal_ to the live watch in debug mode to monitor its values in real-time.
_Note: You may use potentiometers for both ADC channels on pins A0 and A1 as I have done. It is important to note that voltage input to these pins must not exceed 3V. You may use 3V and GND pins on the board for the potentiometers._

## Task B: DMA in Ping-Pong mode
Copy the code in [dma_pp.c](./dma_pp.c) to run DMA in Double Buffer mode. We will use a single ADC input channel at pin A0 to take an analog input. Ping-pong mode uses two memory buffers for DMA. When one of the buffers is filled, the memory pointer is shifted to the second buffer while the data in the first buffer is available for use. We will use on-board LEDs 13 (to indicate current target as memory 0) and 14 (to indicate current target as memory 1) to demonstrate shifts between memory buffers during ping-pong operation. The 'Transfer Complete' interrupt is used to indicate that a buffer has been filled.
Add variables _boundval_buf1_ and _boundval_buf2_ to the live watch in debug mode to monitor boundary values (first and last 5) of the two buffers respectively in real-time.

## Task C
An alternative to ping-pong mode is to use a single buffer, and employ __Half Transfer__ and __Transfer Complete__ interrupts to work on data in the filled half of the buffer while the other half is being filled. Using this approach, write a program that takes input from an ADC channel and downsamples the signal in the filled-half of the buffer by a factor of 25. Take average of the values of the downsampled signal and store in a variable. Repeat the process for each half of the buffer as it is filled, and monitor the real time average value in live watch. Use LEDs to indicate current half of the buffer being served by DMA (as in Task B).
_Hint: Use CR, LISR and LIFCR registers to employ 'Half Transfer' interrupt in the same way as 'Transfer Complete' interrupt has been employed in Task B. Also, use a large size for the buffer (preferably above 10,000) so that the LEDs are toggled at such a speed that is visible to the human eye._
