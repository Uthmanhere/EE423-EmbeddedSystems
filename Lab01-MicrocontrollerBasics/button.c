/* Button Input
 * ____________
 *     This program toggles on-board LED of STM32F429 Discovery Kit.
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
        /* Enable clock for
        *       - GPIO Port G
        *       - GPIO Port A
        */
        RCC->AHB1ENR |= 1 << 6 | 1 << 0;

        /* Set Pin 13 and Pin 14 in General Purpose Output Mode */
        GPIOG->MODER &= 0x00000000;
        GPIOG->MODER |= (0x01 << 2*14 | 0x01 << 2*13);
        
        /* Set Pin 13 and Pin 14 as pull-up pins */
        GPIOG->PUPDR &= 0x00000000;
        GPIOG->PUPDR |= (0x01 << 2*14 | 0x01 << 2*13);
        
        /* Set Pin A in Input Mode */
        GPIOA->MODER &= 0xFFFFFFFC;
        GPIOA->MODER |= 0x00000000;
        
        /* Set Pin 13 and Pin 14 as neither pull-up, nor pull-down */
        GPIOA->PUPDR &= 0xFFFFFFFC;
        GPIOA->PUPDR |= 0x00000000;

        /* Set Pin 13 high */
        GPIOG->ODR |= (1 << 13);

        while(1)
        {
                delay(10000000);
                GPIOG->ODR ^= (1 << 13);        // Toggle LED
                if (GPIOA->IDR & 0x00000001)    // Toggle for Button Press
                        GPIOG->ODR ^= (1 << 14);
        }

}


void delay(uint32_t s)
{
        for(s; s>0; s--){
                asm("NOP");
            }
}
