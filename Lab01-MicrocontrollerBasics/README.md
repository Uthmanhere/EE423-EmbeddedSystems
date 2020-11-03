# Introduction
## ARM Processing Cores
More than 70% of silicon is ARM-based for it's customized sets energy efficient designs. These are categorized as ARM Cortex A, R and M.
### ARM Cortex A Series
These are high speed processors commonly found in smart phones and single-board computers. Some common examples are:
* Samsung Galaxy Note 10: ARM Cortex A76 \& ARM Cortex A55
* Raspberry Pi: ARM Cortex A72
* Beaglebone Black: ARM Cortex A8
* Zynq Ultrascale+: ARM Cortex A53
### ARM Cortex R Series
These are rugged series used in safety-critical applications particularly industrial automation and automotive industry. Some common examples are:
* TI Hercules RM48: ARM Cortex R4F
* Zynq Ultrascale+: ARM Cortex R5
### ARM Cortex M Series
These are energy efficient microcontrollers used in general-purpose embedded systems as sensor nodes, ad-hoc communication and gateways etc. Some examples of these microcontrollers are:
* STM32F429ZI: Arm Cortex M4F
* TI Tiva TM4C12: ARM Cortex M4F
* EMF32 Gecko: ARM Cortex M3
* nRF52840: ARM Cortex M4

## Development Ecosystem
Development for embedded system engineers has never been about coming up with pieces but an end-to-end solution. For this purpose, all the parties at stake need to be considered and referred to throughout the development phase. One mode of communication with entities at each layer is documentation which is already well-versed for famous microcontrollers. In case of an embedded system design with STM32F4 microcontroller, following documentations need to be frequently referred to:
* STM32F429ZI Discovery Kit User Manual (UM1670).
* STM32F4xxxx Application Developer Reference Manual (RM0090).
* ARMv7-M Architecture Reference Manual.
Moreover, well-versed coders consider referring to documentation around IDE as well:
* IAR C/C++ Devlopment Guide for ARM Microprocessor Family
### Development Procedure
The available peripherals on kit and their mappings on board are observed from STM32F429ZI Discovery Kit User Manual. Then the relevant peripheral is implemented through STM32F4xxxx Application Developer Reference Manual, while sometimes referring to ARMv7-M Architecture Reference Manual for core-specific features (as Nested Vector Interrupt Controller.)
### Integrated Development Environment
Support for STM32F4 series microcontrollers is available in IDEs as following:
* GCC
* Keil MDK
* IAR Embedded
We choose IAR Embedded Workbench for development in these labs.

# Lab Tasks
## Task A: The Toolchain
Follow the IAR Embedded Workbench tutorial guide with this lab and program [blink.c](./blink.c) in microcontroller. Familiarize yourself with the debugging environment and try to understand the code as well.
### Code
Refer to [blink.c](./blink.c) for code.
### Flowchart
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab01-MicrocontrollerBasics/img/flow_0.png" alt="flowchart" width="160"/>

### Explanation
* _Line 18_: Enable clock for GPIO port G in AHB1 Enable Register\footnote{Refer to section \texttt{6.3.12} of RM0090 reference manual for RCC AHB1 peripheral clock enable (RCC\_AHB1ENR) register mappings.}. Clock is not provided to all peripherals by default for enabling energy efficiency. Hence clock needs to be enabled for each module in use.
* _Line 21 ,22_: Configure pins 13 \& 14 as general purpose output pins in GPIO Mode Enable Register.
* _Line 25, 26_: Set pins 13 \& 14 in Pull-up configuration through Push Pull Data Register. Other configurations can be Pull Down or Open Drain.
* _Line 29_: Set pin 13 high in Output Data Register.
* _Line 33_: Delay routine is defined in \texttt{line 39} which loops over the argument doing nothing ass \emph{NOP} assembly primitive.
* _Line 34_: Toggle LEDs using XOR operation.

## Task B: Button Polling
Program [button.c](./button.c) in microcontroller. Familiarize yourself with the debugging environment and try to understand the code as well.
### Code
Refer to file [button.c](./button.c) for code.
### Flowchart
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab01-MicrocontrollerBasics/img/flow_1.png" alt="flowchart" width="210"/>

### Explanation
* _Line 18_: Enable clock for GPIO port G and GPIO port A in AHB1 Enable Register.
* _Line 46_: Check port A from Input Data Register if input is high at pin 0. Then toggle the LED; otherwise do not.

## Task C: Button Interrupting
Program [exti.c](./exti.c) in microcontroller. Familiarize yourself with the debugging environment and try to understand the code as well.
### Code
Refer to [exti.c](./exti.c) for code.
### Flowchart for Main Routine
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab01-MicrocontrollerBasics/img/flow_2.png" alt="flowchart" width="160"/>

### Flowchart for Main Routine
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab01-MicrocontrollerBasics/img/flow_2i.png" alt="flowchart" width="280"/>

### Explanation
* _Line 43_: Enable system configuration clock in RCC APB2 peripheral clock enable register\footnote{Refer to section \texttt{6.3.18} of RM0090 reference manual for RCC APB2 peripheral clock enable (RCC\_APB2ENR) register mappings.}.
* _Line 46_: Select source input for external interrupt as pin 0 of GPIO port A in system configuration controller external interrupt configuration register\footnote{Refer to section \texttt{8.2.4} of RM0090 reference manual.}.
* _Line 47_: Enable trigger type in Rising Trigger Selection Register\footnote{Refer to section \texttt{10.3.3} of RM0090 reference manual.}.
* _Line 48_: Set interrupt mask register\footnote{Refer to section \texttt{10.3.1} of RM0090 reference manual.} for EXTI0.
* _Line 51_: Set priority of interrupt request in Interrupt Priority Register\footnote{Refer to section \texttt{B3.4.9} of ARMv7-M Architecture Reference Manual}.
    \item \texttt{Line 54}: Using interrupt set enable register\footnote{Refer to section \texttt{B3.4.4} of ARMv7-M Architecture Reference Manual} to enable EXI interrupt requests.
_ _Line 18_: Check if interrupt is from pin A0 from interrupt pending register\footnote{Refer to section \texttt{10.3.6} of RM0090 reference manual.}.
* _Line 19_: Clear interrupt in interrupt pending register.

## Task D: Timer
Program [tim2.c](./tim2.c) in microcontroller. Familiarize yourself with the debugging environment and try to understand the code as well.
### Code
Refer to [tim2.c](./tim2.c) for code.
### State Diagram
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab01-MicrocontrollerBasics/img/flow_3iii.png" alt="flowchart" width="512"/>

### Flowchart for Main Routine
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab01-MicrocontrollerBasics/img/flow_3.png" alt="flowchart" width="160"/>

### Flowchart for External Interrup Routine
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab01-MicrocontrollerBasics/img/flow_3i.png" alt="flowchart" width="280"/>

### Flowchart for Time Routine
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab01-MicrocontrollerBasics/img/flow_3ii.png" alt="flowchart" width="160"/>

### Explanation
* _Line 62_: Enable clock for timer 2 in RCC APB1 peripheral clock enable register\footnote{Refer to section \texttt{6.3.24} in RM0090 for RCC\_APB1ENR register mappings.}.
* _Line 65 \& 66_: Calibrate timer 2 event update frequency using clock prescalar and autorelaod values. A generic formula in this regard being: $$TIMx\_EVT_f = \frac{CLK_{INT}}{(PSC+1)\times(ARR+1)}$$
        * PLL clock \footnote{Refer to section \texttt{6.2} of RM0090 reference manual for a detailed discussion around clocks.} input frequency is configured in RCC PLL configuration register \footnote{Refer to section \texttt{6.3.2} of RM0090 reference manual RCC\_PLLCFGR register mappings.}. Which for default values of register formulates to $84 Mhz$.
        * Since PLL generated clock is used as system clock, the default internal clock is $84 MHz$.
        \item Timer 2 prescalar value\footnote{Refer to section \texttt{15.4.11} in RM0090 for TIMx\_PSC register mappings.} is 8399 hence the timer clock becomes `TIM2_CLK_f = CLK_INT / (PSC+1) = 84 M / (8399+1) = 10 kHz`
        * Event is generated after overflow of reaching the value of Auto-Reload Register\footnote{Refer to section \texttt{15.4.12} in RM0090 for TIMx\_ARR register mappings.}. Setting it's value to 10,000, the frequency of event becomes:
```     
        TIM2_EVT_f = TIM2_f / (ARR+1) = 10000 / (10000+1) = 1 Hz
```
* _Line 69_: Updates interrupt enable after every iteration of interrupt generation in timer DMA/interrupt update enable register\footnote{Refer to section \texttt{15.4.4} in RM0090 for TIMx\_DIER register mappings.} (TIMx\_DIER).
* _Line 72_: Enable timer 2 module in timer 2 control register 1\footnote{Refer to section \texttt{15.4.1} in RM0090 for TIMx\_CR1 register mappings.} (TIMx\_CR1);
* _Line 19_: Clear timer 2 update interrupt flag (UIF) that interrupt should not trigger twice, exibiting false positives. The UIF bit is cleared in timer status register\footnote{Refer to section \texttt{15.4.5} in RM0090 for TIMx\_SR register mappings.} (TIMx\_SR).

## Task E: Design Task - Digital Stopwatch
Design a stopwatch representing time passed on seven segment display. The clock should initiate as button is pressed and return to 0 after 59 seconds. As button is pressed next time, the watch should stop. Subsequent button press will represent minutes passed. Next button press will result in reset of stopwatch and the one after it initiates it again.
### State Diagram
<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab01-MicrocontrollerBasics/img/flow_4.png" alt="state" width="512"/>

### Use Cases
Some use cases are elaborated as follows:
* (button pressed) 0, 1, 2, 3, 4, (button pressed) 4, (button pressed) 0, (button pressed) 0, (button pressed) 0, 1, 2, 3 ...
* (button pressed) 0, 1, 2, 3, 4, 5, 6, 7, 8, ... 59, 0, 1, 2, 3, 4, 5, 6, 9,  ... 59, 0, 1, 2, 3, 4, (button pressed) 4, (button pressed) 2 (button pressed) 0, (button pressed) 1, 2, 3 ...
* ... (8 minutes passed) 0, 1, (button pressed) 1, (button pressed) 8 ...
* ... (87 minutes passed) 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, (button pressed) 16, (button pressed) 87, (button pressed) 0 ...
