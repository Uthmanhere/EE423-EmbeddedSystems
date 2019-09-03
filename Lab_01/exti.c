/* Button Interrupting
 * ___________________
 *      This program toggles  on-board LED of STM32F429 Discovery Kit
 *      through external interrupt of on-board user button.
 */

/* includes */
#include "stm32f4xx.h"

/* function declarations */
void EXTI0_IRQHandler(void);
int main(void);

/* external interrupt handler */
void EXTI0_IRQHandler(void)
{
        // Check if the interrupt came from EXTI0
        if (EXTI->PR & (1 << 0)){
                GPIOG->ODR ^= (1 << 13);

                /* wait little bit */
                for(uint32_t j=0; j<1000000; j++);

                // Clear pending bit
                EXTI->PR = (1 << 0);
        }
}

/* main code */
int main(void)
{
        /* set up LED */
        RCC->AHB1ENR |= (1 << 6);
        GPIOG->MODER &= !(0x11 << 2*13);
        GPIOG->MODER |= 0x01 << 2*13;

        /* set up button */
        RCC->AHB1ENR |= (1 << 0);
        GPIOA->MODER &= 0xFFFFFFFC;
        GPIOA->MODER |= 0x00000000;

        // enable SYSCFG clock (APB2ENR: bit 14)
        RCC->APB2ENR |= (1 << 14);

        /* tie push button at PA0 to EXTI0 */
        SYSCFG->EXTICR[0] |= 0x00000000; // Write 0000 to map PA0 to EXTI0
        EXTI->RTSR |= 0x00001;   // Enable rising edge trigger on EXTI0
        EXTI->IMR |= 0x00001;    // Mask EXTI0

        // Set Priority for each interrupt request
        NVIC->IP[EXTI0_IRQn] = 0x1; // Priority level 1

        // enable EXT0 IRQ from NVIC
        NVIC->ISER[0] = 1 << EXTI0_IRQn;

        while(1);

}
