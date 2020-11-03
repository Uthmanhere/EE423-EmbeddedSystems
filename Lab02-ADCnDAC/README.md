# Introduction
## Analog-to-Digital Converter
This section provides a theoretical know-how regarding Analog to Digital of STM32F429xx devices. The 12-bit ADC is a successive approximation analog-to-digital converter. It has up to 19 multiplexed channels allowing it to measure signals from 16 external sources, two internal sources, and the VBAT channel. The A/D conversion of the channels can be performed in single, continuous, scan or discontinuous mode. The result of the ADC is stored into a left or right-aligned 16-bit data register. The analog watchdog feature allows the application to detect if the input voltage goes beyond the user-defined, higher or lower thresholds.

Please refer to [ADC Manual](#adc_man) for a developer's introduction to ADC Peripheral. [Task B](#adc_task) provides a trivial implementation to get you started for design [Task C](#sec_design).
## Digital-to-Analog Converter
This section provides a theoretical know-how regarding Digital to Analog Converter (DAC) of STM32F429xx devices. All the STM32F429xx devices contain two such DAC channels. Both of these channels are 12 bit buffered and convert a digital signal into an analog output with voltage signal ranging between 0 to 3 volts. Following are some of the features supported by these channels: 
* Left or right data alignment in 12-bit mode
* 8-bit or 10-bit monotonic output
* Triangular-wave generation
* Sine wave generation 
* Dual DAC channel independent or simultaneous conversions
* DMA capability for each channel
* External triggers for conversion
* Can be used in 8 bit mode or 12 bit mode.

Please refer to section \ref{sec:dac_man} for a developer's introduction to DAC Peripheral. Section \ref{sec:dac_task} provides a trivial implementation to get you started for design task in section \ref{sec:design}.

# ADC Programmer's Manual <a name="adc_man"></a>
## Board Pin-Out
In this lab we will be using channel0 of _ADC1_ through _PINA0_ of the board.
## Registers
Knowledge of the relevant registers is very important so as to configure the board properly. This section provides details about the registers to be used in the tasks to be performed in this lab.
### RCC Registers
The STM32 has a complex clock distribution network which ensures that only those peripherals that are actually needed are powered. This system is called Reset and Clock Control (RCC). RCC registers are used for clock control and resetting purposes.
#### RCC AHB1 peripheral clock register _RCC\_AHB1ENR_
This Register is used to control clock for various peripherals. Setting the LSB will enable the clock for GPIO port A.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/gpioa.PNG" width="800"/>

#### RCC APB2 peripheral clock enable register (_RCC\_APB2ENR_)
ADC1 is driven by the APB2 clock. Setting bit 8 of the register will enable the clock for ADC1 .

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/GPIOA_ADC.PNG" width="800"/>

### GPIO port mode register (GPIOA\_MODER)}
This register is used to specify the mode of the pin being used.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/port_mode.PNG" width="800"/>

Setting the bit 0 and 1 of the register will configure the PINA0 in analog mode. 
### ADC Registers
#### ADC Control Register 1 ADC\_CR1
This register is used for controlling various important features of the ADC. In this lab we will use bits 24 and 25 for setting the resolution of ADC. This register is also used to choose the channel of ADC.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/ADC_CR1.PNG" width="800"/>

#### ADC Control Register 2 ADC\_CR2
This register controls the main functionality of the ADC. Setting the LSB enables the ADC and setting bit 30 starts it. For continuous conversion mode we need to set bit 1 of the register.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/adc_cr2.PNG" width="800"/>

#### ADC Data Registers
The data received as the input is converted and stored in a 16 bit data register(ADCX\_DR). This register is written every time EOC(end of conversion) flag is set.
## Functional Description of ADC
### ADC Resolution
The reolution of the ADC can be configured into one of the following options:
* 6-bit.
* 8-bit.
* 10-bit.
* 12-bit.

### ADC Conversion
The voltage at the input of the ADC can be obtained using the following conversion formula:
```
     Voltage_ADC_in = ADCx_DR x 3 / 2^n
```
where n represents the resolution.
## ADC configuration
Following are the steps that should be followed while configuring a DAC channel:
* Enable the ADC Clock.
* Set the resolution and choose the channel.
* Enable the end of conversion flag.
* Choose the conversion mode
* During the single conversion mode, enable and start the ADC within the _while(1)_ loop and stop it after getting the value.
* If using the continuous conversion mode, enable and start the ADC once.

# DAC Programmer's Manual <a name="dac_man"></a>
## Board Pin-Out
The GPIO pins that are reserved for using DAC are PA4 or PA5. These pins need to be configured to 'analog' mode before using for DAC.
## Registers
Knowledge of the relevant registers is very important so as to configure the board properly. This section provides details about the registers to be used in the tasks to be performed in this lab.
### RCC Registers
The STM32 has a complex clock distribution network which ensures that only those peripherals that are actually needed are powered. This system is called Reset and Clock Control (RCC). RCC registers are used for clock control and resetting purposes.
#### RCC AHB1 peripheral clock register (RCC\_AHB1ENR)
This Register is used to control clock for various peripherals. Setting the LSB will enable the clock for GPIO port A.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/gpioa.PNG" width="800"/>

#### RCC APB1 peripheral clock enable register (RCC\_APB1ENR)
DAC is driven by the APB1 clock. Setting bit 29 of the register will enable the clock for DAC.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/DACEN.PNG" width="800"/>

### GPIO port mode register (GPIOA\_MODER)
This register is used to specify the mode of the pin being used.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/port_mode.PNG" width="800"/>

Setting the bit 8 and 9 of the register will configure the PINA4 in analog mode. 
### DAC Registers
#### DAC Control Register DAC\_CR
This register is used for controlling the functionality of DAC. It provides options to configure desired DAC channels with various features. 

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/daccr.PNG" width="800"/>

#### DAC Data Registers
The data is not written directly to the Data Output Register(DOR). The data is initially holded in a data holding register and is transferred to the DOR with next APB1 clock cycle. The choice of the register depends on the data format being used.
## Functional Description of DAC
### DAC Data Format
The data format depends on the selected configuration mode. In short for a single independent DAC channel following are three possibilities to write data in the output register:
* 8-bit right alignment.
* 12-bit left alignment.
* 12-bit right alignment.

### DAC Conversion
The data in the DAC output register cannot be written directly. Any data transfer to be performed through the DAC channel requires the loading of data in the DAC\_DHRx register first. Data stored in the DAC\_DHRx register are automatically transferred to the DAC\_DORx register after one clock cycle.

### DAC Output Voltage
Digital inputs are converted to output voltages on a linear conversion between 0 and 3 volts. The analog output voltages can be determined using the following equation:
```
      Voltage_DAC_out = Data x 3 / 2^n
```
where n corresponds to either 8 bit or 12 bit mode.

## Configuring DAC
This chapter provides details about configuration of DAC.
### Port and Pin Configuration
To use DAC we have to configure either of the two pins to analog mode before using it for conversion purposes. Following steps need to be followed in this regard:
* Enable the GPIO PORT (A) clock.
* Select the pin to be used and configure it to analog.
### DAC Configuration
Following are the steps that should be followed while configuring a DAC channel:
* Enable the DAC clock.
* Enable the output buffer for the desired channel.
* Reset the other bits.

# Lab Tasks
## Task A: Voltmeter using ADC <a name="adc_task"></a>
The code provided in [adc.c](./adc.c) is in accordance to the procedure explained in the section \ref{sec:adc_man}. Connect a potentiometer with the board using the power supply of the board itself. Read the output of potentionmeter from _PINA0_ using the [adc.c](./adc.c) code. View the value of variable _volt_value_ by right clicking on it and adding it to the live watch.
## Task B: Sine Wave Generation using DAC <a name="dac_task"></a>
The code provided [dac.c](./dac.c) is in accordance to the procedure explained in section \ref{sec:dac_man}. Copy this code in the IDE, build it and upload on the target device. Press Reset button once after uploading the code. Connect DAC pin to an oscilloscope and observe the results.
## Task C: Design Task <a name="sec_design"></a>
Learning from manuals and examples in [ADC Manual](#adc_man), [DAC Manual](#dac_man)\ref{sec:dac_man} and [Task A](#adc_task), [Task B](#dac_task); implement a firmware which reads certain amount of samples from ADC and stores it in an array/buffer. Perform __moving average filtering__ on the acquired samples and show the output through DAC. Refer to the flowchart below for a better understanding of task in hand.

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab_02/img/task_flow.png" width="800"/>
