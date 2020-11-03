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
                asm("NOP");
        }
}
