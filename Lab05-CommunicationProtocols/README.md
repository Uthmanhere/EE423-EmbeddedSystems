# Introduction
## Programming Layers
Previously we have been programming microcontroller functionalities by accessing registers. We have observed that bits and bytes are need to be taken care of, in this approach and hence it is prone to errors. There exist other approaches to code microcontrollers which have pros and cons of their own.

Following is a list of layers through which microcontroller peripherals can be programmed:
    -  Register layer
    -  Peripheral driver library
    -  Harware abstraction layer (HAL)
Peripheral driver library is same as register layer except the fact that functions need to be called for achieving a particular functionality. Hardware abstraction layer on the other hand defines constructs for coders to just mention the goal and functionality is already implemented in the abstraction thereof.

Let us consider example of setting mode of a gpio pin.

### Register Layer
```C
GPIOA->MODER &= 11 << 2*3;
GPIOA->MODER |= 10 << 2*3;
```
### Peripheral Driver Library
```C
gpio_pin_config(GPIOA, GPIO_PIN_3, OUTPUT);
```

### Hardware Abstraction Layer
```C
DigitalOut indicator_led(PA_3);
```
### Mbed OS
Mbed OS is ARM's very own real-time operating system (RTOS). We will discuss core functionalities of RTOS (as processes, priorities, synchronization constructs) in later labs. One phenomenon RTOS comes along is Hardware Abstraction Layer.

Mbed OS implements RTOS and HAL functionality through Object-Oriented Principles in C++. Objects are instantiated and their functions are called to access peripherals. Hence Mbed OS code is clean and easy to learn.

It is a poor programming practice to use perpheral driver library or access registers when RTOS functionality has already been provided. Hence in this lab, you are introduced to HAL in perspective of Mbed OS.

### Development Environment
Mbed OS can be compiled in either of the two ways:
-  <a href="https://ide.mbed.com">Online Compiler</a>
-  Mbed Studio

## Inter-Integrated Circuit (I2C)
I$^2$C was introduced by Philips Semiconductor (now NXP Semiconductors) in 1982. For details of protocols refer to it's  <a href=https://www.nxp.com/docs/en/user-guide/UM10204.pdf>official manual</a>.


### Characteristics
The serial bus interface can be characterised by following:
-  Two-Wire: Only two bus lines are required
	-  Serial Data - SDA
	-  Serial Clock - SCL
-  Multiple Devices: Each device connected is software addressable through a unique address.
-  Communication: The bus operates in two communication modes.
	-  Master Transmit
	-  Master Receive
-  Capacity Limitation: The number of devices connected on the same bus is limited by bus capacitance only.
### Communication
#### Start Condition
A high-to-low switch in SDA line before that in SCL.
#### Address Frame
A 7 bit address unique address of the slave device.
#### Read/Write Bit
A single bit to specify either read or write operation on slave.
#### Data Frame
An 8 bit frame that containing data with MSB-first principle. 
#### Stop Condition
A low-to-high switch in SDA line before that in SCL.

## Serial Peripheral Interface (SPI)
SPI was introduced by Motorola in 1990's. Refer to [manual](https://opencores.org/usercontent/doc/1499360489g{manual}\footnote{https://opencores.org/usercontent/doc/1499360489) for complete details regarding the protocol.
### Characteristics
The SPI protocols is known for following characteristics:
-  Four Wires
    -  Master-Out-Slave-In - $MOSI$
    -  Master-In-Slave-Out - $MISO$
    -  Slave-Select - $\overline{SSg$
    -  Serial Clock - $SCK$
-  Full Dubplex communication.
-  Multiple slaves are supported.
### Communication
-  Master generates the clock first.
-  Slave-select pin is pulled down to reserve the bus.
-  Data frames are sent and received.
### Configurations
Following configurations need to be taken care of, when designing an SPI communication firmware:
-  Clock Speed
-  Clock Polarity
-  Clock Phase

# Lab Tasks
## Task A: SPI
STmicrocelectronics ARM Cortex M4F is slave while Arduino's ATmega328P has been used as master in these code snippets.
### Pinout
SPI Pin | STM32F429I Disc | Arduino UNO
------- | --------------- | -----------
MOSI | A7 | 11
MISO | A6 | 12
SCK | A5 | 13
SSEL | A4 | 10
### SPI Slave
#### Code Snippet
The code snippet [spiSlaveArm.cpp](./spiSlaveArm.cpp) for SPI slave is for STM32F429I firmware.
#### Explanation
An object of class SPISlave in instantiated with representative pins. Reply, read and receive functions of the object are being used for communication.
### SPI Master
The code snippet [spiMasterAvr.cpp](./spiMasterAvr.cpp) for SPI slave is for Arduino UNO firmware.

## Task B: I2C
STmicrocelectronics ARM Cortex M4F is master while Arduino's ATmega328P has been used as slave in these code snippets.
### Pinout
I2C Pin | STM32F429I Disc | Arduino UNO
------- | --------------- | -----------
SDA | B11 | A4
SCL | B10 | A5
### I^$C Master
The code snippet [i2cMasterArm.cpp](./i2cMasterArm.cpp) for SPI master is for STM32F429I Disc firmware.

#### Explanationg
An I2C (master) object is instantiated by specifying particular pins. Write function of the object is used to communicate with slave with specified address on the bus.
### I^2C Slave
The [i2cSlaveAvr.cpp](./i2cSlaveAvr.cpp) for SPI slave is for Arduino UNO firmware.

## Task C
Code application layer for communication with precoded Arduino UNO. The Arduino and STM32F429I Disc are communicating through SPI and I$^2$C protocols.
### Pinout
#### SPI
SPI Pin | STM32F429I Disc | Arduino UNO
------- | --------------- | -----------
MOSI | A7 | 11
MISO | A6 | 12
SCK | A5 | 13
SSEL | A4 | 10
#### I2C
I2C Pin | STM32F429I Disc | Arduino UNO
------- | --------------- | -----------
SDA | B11 | A4
SCL | B10 | A5
#### Seven Segment Display
BCD Pin | Arduino Uno
------- | -----------
A | 7
B | 6
C | 5
D | 4
### Communication Specifications
The precoded Arduino UNO shall be having following specifications you need to conform to:
-  SPI Master
    -  Constantly send Arduino UNO's state (bored, happy or sad.) Receive this response and print it on screen.
    -  Example
        -  Response: E5Happy
        -  Response type: E (emotion)
        -  Response lenght: 5
        -  Response emotion: Happy
        -  Print: ``Arduino is feeling Happy right now.''
-  I2C Slave
    -  Raises an interrupt to respond over internal configurations assigned through this bus.
    -  Configuration Strings
        -  CU: Count up (loop back to 0 after 9.)
        -  CD: Count Down (loop back to 9 after 0.)
        -  Rn: Set response emotion to number n where n ϵ {1,2,3,4}.

### Application Layer Specifications
The solution application should consist of three buttons (interrupt or polling), serial monitor and communications.
    -  Serial Monitor should print emotion response whenever received from Arduino through SPI.
    -  Button up should configure Arduino's seven segment to count one up.
    -  Button down should configure Arduino's seven segment to count one down.
    -  Button response should alter the response configuration eventually iterating over all emotions for next time it is pressed.
### Arduino Code
The firmware of arduino can be observed from the referred code snippet [avrApp.cpp](./avrApp.cpp).
### State Diagram

<img src="https://github.com/Uthmanhere/EE423-EmbeddedSystems/blob/master/Lab05-CommunicationProtocols/img/state.png" width="800">

