# Introduction
## Analog-to-Digital Converter
This section provides a theoretical know-how regarding Analog to Digital of STM32F429xx devices. The 12-bit ADC is a successive approximation analog-to-digital converter. It has up to 19 multiplexed channels allowing it to measure signals from 16 external sources, two internal sources, and the VBAT channel. The A/D conversion of the channels can be performed in single, continuous, scan or discontinuous mode. The result of the ADC is stored into a left or right-aligned 16-bit data register. The analog watchdog feature allows the application to detect if the input voltage goes beyond the user-defined, higher or lower thresholds.

Please refer to section \ref{sec:adc_man} for a developer's introduction to ADC Peripheral. Section \ref{sec:adc_task} provides a trivial implementation to get you started for design task in section \ref{sec:design}.
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
