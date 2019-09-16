#include "stm32f4xx.h"

//Functions prototypes
void GPIO_Config(void);
void DMA_Config(void);
void ADC_Config(void);
void startADC_DMA_DBM(uint32_t SrcAddr, uint32_t DstAddr1, uint32_t DstAddr2, uint32_t length);
void DMA2_Stream0_IRQHandler(void);

//Numeric variables
uint8_t adcVal1[5000];
uint8_t adcVal2[5000];
uint8_t boundval_buf1[10]; //array containing boundary values (first 5 and last 5) of buffer 1 (adcVal1)
uint8_t boundval_buf2[10]; //array containing boundary values (first 5 and last 5) of buffer 2 (adcVal2)

int main(void)
{
	//GPIO Analog input Config
	GPIO_Config();
	//DMA Config
	DMA_Config();
	//ADC Config
	ADC_Config();
	
	//Start ADC DMA in Double Buffer Mode
	startADC_DMA_DBM((uint32_t)&ADC1->DR, (uint32_t)adcVal1, (uint32_t)adcVal2, 5000);
	
	while(1)
	{
			
	}
}

void GPIO_Config(void)
{
	//Enable GPIOA clock
	RCC->AHB1ENR |= 0x01;
        //Enable GPIOG
        RCC->AHB1ENR |= (0x01 << 6);
	
	//Set pin 0 of Port A as Analog
	GPIOA->MODER |= (0x03 << 0*2);
	//Set pins 13 and 14 (LEDs) of Port G in general purpose output mode
	GPIOG->MODER |= (0x01 << 13*2) | (0x01 << 14*2);
	//Set pins 13 and 14 of Port G as pull-up pins                 
	GPIOG->PUPDR |= (0x01 << 13*2) | (0x01 << 14*2);
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
	
	NVIC_EnableIRQ(DMA2_Stream0_IRQn);
	NVIC_SetPriority(DMA2_Stream0_IRQn, 0);	
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
	
	ADC1->SQR1 = (0x00 << 20); //Number of conversions = 1
	ADC1->SQR3 = 0x00;        
	ADC1->SQR3 |= (0x00 << 0);  //Channel 0 is 1st in conversion sequence
}

void startADC_DMA_DBM(uint32_t SrcAddr, uint32_t DstAddr1, uint32_t DstAddr2, uint32_t length)
{
	ADC1->CR2 &= ~(0x01);  //Stop ADC1
	//Set DBM bit to enable Double Buffer Mode
	DMA2_Stream0->CR |= (0x01 << 18);
	//Set DMA data length
	DMA2_Stream0->NDTR = length;
	//Set Source Address
	DMA2_Stream0->PAR = SrcAddr;
	//Set Destination Address 1
	DMA2_Stream0->M0AR = DstAddr1;
        //Set Destination Address 2
	DMA2_Stream0->M1AR = DstAddr2;
	
	//Enable ADC DMA
	ADC1->CR2 |= (0x01 << 8);
	//Clear all interrupt flags for Stream 0
	DMA2->LIFCR |= 0x3D;
	//Enable Transfer complete Interrupt
	DMA2_Stream0->CR |= (0x01 << 4);
	//DMA Stream Enable
	DMA2_Stream0->CR |= (0x01 << 0);

	//Clear ADC flags
	ADC1->SR = 0x00000000;

	ADC1->CR2 |= 0x01;  //Start ADC1
	ADC1->CR2 |= (0x01 << 30);
}

//Interrupt Service Routine for DMA 2 Stream 0 Interrupts
void DMA2_Stream0_IRQHandler(void)
{        
        //Interrupt Handler for Transfer Complete Interrupt
	if (((DMA2->LISR) & (0x01 << 5)) == (0x01 << 5)) //Check if Transfer Complete Flag is set
        {
          //Check CT (Current Target) bit in CR register. If set, current target is Memory 1 (Dest 2). If clear, current target is Memory 0 (Dest 1)
          if (((DMA2_Stream0->CR) & (0x01 << 19)) == (0x01 << 19)) //True if DMA is serving Dest 2
          {
            GPIOG->ODR = (0x00 << 13) | (0x01 << 14); //Turn on LED 14 and turn off LED 13
            for(uint8_t i = 0; i < 5; i++)
            {
              boundval_buf1[i] = adcVal1[i];
            }
            for(uint8_t i = 5; i < 10; i++)
            {
              boundval_buf1[i] = adcVal1[4990+i];
            }
          }
          
          else if (((DMA2_Stream0->CR) & (0x01 << 19)) == (0x00 << 19)) //True if DMA is serving Dest 1
          {
            GPIOG->ODR = (0x01 << 13) | (0x00 << 14); //Turn on LED 13 and turn off LED 14
            for(uint8_t i = 0; i < 5; i++)
            {
              boundval_buf2[i] = adcVal2[i];
            }
            for(uint8_t i = 5; i < 10; i++)
            {
              boundval_buf2[i] = adcVal2[4990+i];
            }
          }
            
          //Clear transfer complete interrupt flag
          DMA2->LIFCR |= (0x01 << 5);
        }
}
