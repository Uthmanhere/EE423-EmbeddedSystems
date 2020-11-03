#include "stm32f4xx.h"

void GPIO_config(void);// Function prototype for GPIO configuration
void ADC_config(void);
uint8_t adcval = 1;
float volt_value;
int main()
{

GPIO_config();
ADC_config();

/*include next 2 lines for continous conversion*/
//ADC1->CR2 |= ADC_CR2_ADON;
//ADC1->CR2 |= (1 << 30);
while(1)
{
/*comment out next two lines for continous conversion mode*/
ADC1->CR2 |= ADC_CR2_ADON; // enabling the ADC
ADC1->CR2 |= (1 << 30); // starting the ADC 

	while((ADC1->SR & 0x2) != 0x2) // check until the EOC flag is not set
	{	}
	adcval = ADC1->DR;// Take the value from the ADC register to a variable
	
	volt_value = (((float)adcval)/255)*3;// conversion into voltage values.
	
	/*comment out next line for continous conversion mode*/
	ADC1->CR2 &= ~ADC_CR2_ADON; // Stopping in single conversion mode
	ADC1->SR = 0x00;// resetting all the flags 
 }
}

void GPIO_config(void)
{
	RCC->AHB1ENR |= (0x01 << 0);
	GPIOA->MODER &= ~((0x11 << 0)); // reseting the 0 of MODE Register 
	GPIOA->MODER |= (0x03 << 0);  // choosing the analog mode by moving 11 to bit 8 and 9.
  	GPIOA->PUPDR &= ~(0x00000003); //Configuring the mode
}


void ADC_config(void)
{
	RCC->APB2ENR = (0x01<<8);
	ADC1->CR1 &= 0x00000000;
	ADC1->CR1 |= ( 0x2 << 24 );// set resolutions to 8 bits + channel 0
	
	//ADC1->CR1 |= (0x10);
	ADC1->CR2 &= 0x00000000;
	ADC1->CR2 |= (0x1 << 10);// eoc flag
	/*include next line for continous conversion mode*/
//	ADC1->CR2 = (0x1 << 1);// continous conversion mode 
	ADC1->SMPR2 = 1;
	ADC1->SQR1 &= 0x00000000;
	ADC1->SQR1 |= (0x0 << 20); // 1 conversion 
	ADC1->SQR3 = 0x00;
}
