#include "stm32f4xx.h"
//#define __FPU_PRESENT  1
#include "arm_math.h"
// configuration routines 

void GPIO_config(void);// Function prototype for GPIO configuration
void DAC_config(void);// function prototype for DAC configuration
volatile float data_value = 0.0;// Value in volts
float freq = 5;
float volt_value;
uint8_t dac8bit;// The value to be loaded into 8 Bit register as using the 8 bit mode

int main()
{

  GPIO_config();
  DAC_config();

  //Enabling channel 1 
  DAC->CR |= 1UL;
  //Placing the value in data holding register using 8 bit rigth alligned mode 

  while(1)
  {
    for(uint32_t i=0;i < 400000; i++)
    {	
      data_value  = 1.5+(float)sin((float)2 * (float)3.14159265359 * (float)1 * (float)freq * ((float)i/1000 )); 
      dac8bit  =	(uint8_t)((data_value/3)*255);
      DAC->DHR8R1 = dac8bit;
    }
    //DAC->DHR8R1 = dac8bit;
    for (long j=0;j<4000*100;j++); // small delay

  }
}
void GPIO_config(void)
{
  // enable the clock of GPIO PORT A
  RCC->AHB1ENR |= (0x01 << 0);
  // set the pinA4 to analog mode 
  //GPIOA->MODER   &= ~((3ul << 2*pin)); pin corresponds to the pin number being used 
  GPIOA->MODER |= (0x03 << 2*4);  // choosing the analog mode by moving 11 to bit 8 and 9.

}


void DAC_config(void)
{
  // Enabling the DAC clock 
  RCC->APB1ENR |= (0x01 << 29);
  // Resetting all the other bits according to the reset values specified in the reference manual
  DAC->CR &= ~(0x3FFFUL << 16);
  // Channel 1 output buffer enable  
  DAC->CR &= ~(0x1FFFUL << 1);
}
