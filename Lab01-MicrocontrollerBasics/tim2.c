/* Button Interrupting
 * ___________________
 *      This program toggles  on-board LED of STM32F429 Discovery Kit
 *      through external interrupt of on-board user button.
 */

/* includes */
#include "stm32f4xx.h"

/* function declarations */
void TIM2_IRQHandler(void);
void EXTI0_IRQHandler(void);
int main(void);

/* timer interrupt handler */
void TIM2_IRQHandler(void)
{
        // Clear pending bit first
        TIM2->SR = (uint16_t)(~(1 << 0));

        GPIOG->ODR ^= (1 << 13);
}

/* external interrupt handler */
void EXTI0_IRQHandler(void)
{
        // Check if the interrupt came from EXTI0
        if (EXTI->PR & (1 << 0)){
                TIM2->CR1 ^= 1 << 0;

                /* wait little bit */
                for(uint32_t j=0; j<1000000; j++);

                // Clear pending bit
                EXTI->PR = (1 << 0);
        }
}

/* main code */
int main(void)
{

        // enable GPIOG clock
        RCC->AHB1ENR |= (1 << 6);
        GPIOG->MODER &= !(0x11 << 2*13);
        GPIOG->MODER |= 0x01 << 2*13;

        // enable SYSCFG clock
        RCC->APB2ENR |= (1 << 14);
        SYSCFG->EXTICR[0] |= 0x00000000; // Write 0000 to map PA0 to EXTI0
        EXTI->RTSR |= 0x00001;   // Enable rising edge trigger on EXTI0
        EXTI->IMR |= 0x00001;    // Mask EXTI0

        // Set Priority for each interrupt request
        NVIC->IP[EXTI0_IRQn] = 0x10; // Priority level 1

        // enable EXT0 IRQ from NVIC
        NVIC->ISER[0] = 1 << EXTI0_IRQn;        
        
        
        // enable TIM2 clock
        RCC->APB1ENR |= (1 << 0);

        // set prescalar and autoreload value
        TIM2->PSC = 8399;
        TIM2->ARR = 10000;

        // Update Interrupt Enable
        TIM2->DIER |= (1 << 0);

        // enable TIM2 IRQ from NVIC
        NVIC->ISER[0] |= 1 << TIM2_IRQn;

        // Enable Timer 2 module (CEN, bit0)
        TIM2->CR1 |= (1 << 0);
        
        while (1);

}
