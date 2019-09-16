#include "stm32f4xx.h"

//Function prototypes
void GPIO_Config(void);
void DMA_Config(void);
void ADC_Config(void);
void startADC_DMA(uint32_t SrcAddr, uint32_t DstAddr, uint32_t length);

//Numeric variables
uint8_t adcVal[2];

int main(void)
{
	//GPIO Analog input Config
	GPIO_Config();
	//DMA Config
	DMA_Config();
	//ADC Config
	ADC_Config();
	
	//Start ADC DMA
	startADC_DMA((uint32_t)&ADC1->DR, (uint32_t)adcVal, 2);
	
	while(1)
	{
			
	}
}

void GPIO_Config(void)
{
	//Enable GPIOA clock
	RCC->AHB1ENR |= 0x01;
	
	//Set pins 0,1 as Analog
	GPIOA->MODER |= (0x03 <<0*2) | (0x03 <<1*2);
}

void DMA_Config(void)
{
	//DMA2 clock enable
	RCC->AHB1ENR |= (0x01 << 22); 
	//Disable DMA stream to start off with
	DMA2_Stream0->CR &= ~(0x01 << 0);
	while(((DMA2_Stream0->CR) & 0x1) == 0x1);  //Wait until DMA is disabled
	//Clear all the settings
	DMA2_Stream0->CR &= 0x00000000;
	
	//Apply DMA settings
	DMA2_Stream0->CR &= ~(0x07 << 25);  //Select Channel 0 of DMA
	DMA2_Stream0->CR &= ~(0x03 << 6);   //Direction: Peripheral to Memory
	DMA2_Stream0->CR &= ~(0x01 << 9);   //Disable Peripheral Increment
	DMA2_Stream0->CR |= (0x01 << 10);   //Enable Memory increment
	DMA2_Stream0->CR &= ~(0x03 << 11);  //Peripheral datasize = Byte
	DMA2_Stream0->CR &= ~(0x03 << 13);  //Memory datasize = Byte
	
	DMA2_Stream0->CR |= (0x01 << 8);    //Enable Circular mode
	DMA2_Stream0->CR &= ~(0x03 << 16);  //Priority Level = Low
	DMA2_Stream0->FCR &= ~(0x01 << 2);  //Disable FIFO - Direct Mode enabled
}

void ADC_Config(void)
{
	//Enable ADC1 clock
	RCC->APB2ENR |= (0x01 << 8);
	//ADC basic configuration
	ADC1->CR2 &= ~((0x01 << 30) | (0x01 << 0));  //Disable ADC to start off
	
	ADC1->CR1 &= ~(0x03 << 24); //Clear resolution field
	ADC1->CR1 |= (0x02 << 24);  //Set Resolution to 8-bits
	ADC1->CR1 |= (0x01 << 8);   //Enable Scan mode
	ADC1->CR1 |= (0x01 << 26);  //Enable OVR (Overrun) interrupt
	ADC1->CR1 |= (0x01 << 5);   //EOC Interrupt enable
	ADC1->CR2 &= ~(0x01 << 11); //Data Align to Right
	ADC1->CR2 |= (0x03 << 8);   //Enable DMA contiuous transfer
	ADC1->CR2 |= (0x01 << 1);   //Enable ADC continuous conversion
	ADC1->CR2 |= (0x01 << 10);  //EOC after single conversion
	ADC->CCR |= (0x03 << 16);   //Set clock prescalar to 8
	
	ADC1->SMPR2 |= (0x07 << 0); //Channel 0 Sampling time to 480 cycles
	ADC1->SMPR2 |= (0x07 << 3); //Channel 1 Sampling time to 480 cycles
	
	ADC1->SQR1 = (0x01 << 20); //Number of conversions = 2
	ADC1->SQR3 = 0x00;        
	ADC1->SQR3 |= (0x00 << 0);  //Channel 0 is 1st in conversion sequence
	ADC1->SQR3 |= (0x01 << 5);  //Channel 1 is 2nd in conversion sequence	
}

void startADC_DMA(uint32_t SrcAddr, uint32_t DstAddr, uint32_t length)
{
	ADC1->CR2 &= ~(0x01);  //Stop ADC1
	//Clear DBM bit to disable Double Buffer Mode
	DMA2_Stream0->CR &= ~(0x01 << 18);
	//Set DMA data length
	DMA2_Stream0->NDTR = length;
	//Set Source Address
	DMA2_Stream0->PAR = SrcAddr;
	//Set Destination Address
	DMA2_Stream0->M0AR = DstAddr;
	
	//Enable ADC DMA
	ADC1->CR2 |= (0x01 << 8);
	//DMA Stream Enable
	DMA2_Stream0->CR |= (0x01 << 0);

	//Clear ADC flags
	ADC1->SR = 0x00000000;

	ADC1->CR2 |= 0x01;  //Start ADC1
	ADC1->CR2 |= (0x01 << 30);
}
