#include <Wire.h>
#include <SPI.h>                             //Library for SPI 

uint8_t resp_num = 0;
uint8_t count = 0;
char * req_array[] = {"happy", "sad", "bored", "shocked"};

void setup (void)
{
  pinMode(SS, OUTPUT);

  pinMode(7, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(4, OUTPUT);
  Serial.begin(9600);                   //Starts Serial Communication at Baud Rate 115200 
  
  SPI.begin();                            //Begins the SPI commnuication
  SPI.setClockDivider(SPI_CLOCK_DIV8);    //Sets clock for SPI communication at 8 (16/8=2Mhz)
  digitalWrite(SS, HIGH);                  // Setting SlaveSelect as HIGH (So master doesnt connnect with slave)
  
  Wire.begin(0x08);                // join i2c bus with address #8
  Wire.onReceive(receiveEvent); // register event
}

void loop(void)
{
  if (resp_num > 4) resp_num = 0;
  if (resp_num)
  {
    digitalWrite(SS, LOW);                  //Starts communication with Slave connected to master
    char * emotion = req_array[resp_num-1];
    uint8_t len = strlen(emotion);
    SPI.transfer('E');
    SPI.transfer(len);
    int em_count = 0;
    while(emotion[em_count])
    {
      SPI.transfer(emotion[em_count]);
      ++em_count;
    }
    digitalWrite(SS, HIGH);
  }
  delay(1000);
}

void display(uint8_t count)
{

  int n1, n2, n3, n4;
  if (count & 1) n1 = HIGH;
  else n1 = LOW;
  if (count & 2) n2 = HIGH;
  else n2 = LOW;
  if (count & 4) n3 = HIGH;
  else n3 = LOW;
  if (count & 8) n4 = HIGH;
  else n4 = LOW;

  Serial.print("n1 "); Serial.print(n1);
  Serial.print("n1op "); Serial.print(count & 1);
  
  digitalWrite(7,n1);
  digitalWrite(6,n2);
  digitalWrite(5,n3);
  digitalWrite(4,n4);

  Serial.println();
}

void receiveEvent(int num) {
  char req_type = Wire.read();
  uint8_t req_num = Wire.read();
  Serial.print(req_type);
  Serial.print(req_num);

  if (req_type == 'R')
  {
    resp_num = req_num;
  }
  else if (req_type == 'C')
  {
      switch (req_num)
      {
        case 'U': display(++count);
                  if (count == 9) count = 0;
                  break;
        case 'D': display(--count);
                  if (count == 0) count = 9;
                  break;
        default:
            digitalWrite(SS, LOW);
            SPI.transfer('E');
            SPI.transfer('R');
            SPI.transfer('R');
            digitalWrite(SS, HIGH);
            return;
      }
  }
  else
  {
    digitalWrite(SS, LOW);
    SPI.transfer('E');
    SPI.transfer('R');
    SPI.transfer('R');
    digitalWrite(SS, HIGH);
  }

  digitalWrite(SS, LOW);
  SPI.transfer('O');
  SPI.transfer('K');
  digitalWrite(SS, HIGH);
}
