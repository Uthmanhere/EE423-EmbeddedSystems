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
Follow the IAR Embedded Workbench tutorial guide with this lab and program _blink.c_ in microcontroller. Familiarize yourself with the debugging environment and try to understand the code as well.
### Code
```C
/* Blinking LED
 * ____________
 *     This program blinks on-board LEDs of STM32F429 Discovery Kit.
 *     The on-board LEDs are connected to pin 13 and pin 14 of
 *     GPIO port G
 */

/* Includes */
#include "stm32f4xx.h"

/* Function prototypes */
int main(void);
void delay(uint32_t);

int main(void)
{
        /* Enable clock for GPIOG */
        RCC->AHB1ENR |= 1 << 6;

        /* Set Pin 13 and Pin 14 in General Purpose Output Mode */
        GPIOG->MODER &= 0x00000000;
        GPIOG->MODER |= (0x01 << 2*14 | 0x01 << 2*13);
        
        /* Set Pin 13 and Pin 14 as pull-up pins */
        GPIOG->PUPDR &= 0x00000000;
        GPIOG->PUPDR |= (0x01 << 2*14 | 0x01 << 2*13);

        /* Set Pin 13 high */
        GPIOG->ODR |= (1 << 13);

        while(1)
        {
                delay(10000000);
                GPIOG->ODR ^= (1 << 13) | (1 << 14);  // Toggle LED
        }

}

void delay(uint32_t s)
{
        for(s; s>0; s--){
                asm(``NOP'');
        }
}
```
### Flowchart
A graphical representation of this code can be observed in figure.
![Image of Flowchart](https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_01/img/flow_0.png)
### Explanation
* _Line 18_: Enable clock for GPIO port G in AHB1 Enable Register\footnote{Refer to section \texttt{6.3.12} of RM0090 reference manual for RCC AHB1 peripheral clock enable (RCC\_AHB1ENR) register mappings.}. Clock is not provided to all peripherals by default for enabling energy efficiency. Hence clock needs to be enabled for each module in use.
* _Line 21 ,22_: Configure pins 13 \& 14 as general purpose output pins in GPIO Mode Enable Register.
* _Line 25, 26_: Set pins 13 \& 14 in Pull-up configuration through Push Pull Data Register. Other configurations can be Pull Down or Open Drain.
* _Line 29_: Set pin 13 high in Output Data Register.
* _Line 33_: Delay routine is defined in \texttt{line 39} which loops over the argument doing nothing ass \emph{NOP} assembly primitive.
* _Line 34_: Toggle LEDs using XOR operation.
