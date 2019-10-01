#include<SPI.h>

void setup (void)
{
  Serial.begin(9600);
  
  SPI.begin();                            // Begins the SPI commnuication
  SPI.setClockDivider(SPI_CLOCK_DIV8);    // Sets clock for SPI communication at 8 (16/8=2Mhz)
  digitalWrite(SS, HIGH);                 // Setting SlaveSelect as HIGH
}

void loop(void)
{ 
  uint8_t Mastersend = 99;
  digitalWrite(SS, LOW);    // Starts communication with Slave connected to master
  SPI.transfer(Mastersend); // Send the mastersend value to slave
  digitalWrite(SS, HIGH);
  
  delay(1000);
}
