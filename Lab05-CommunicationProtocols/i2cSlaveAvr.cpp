#include <Wire.h>

void setup() {
  Wire.begin(0x08);                // join i2c bus with address #8
  Wire.onReceive(receiveEvent); // register event
  Serial.begin(9600);           // start serial for output
}

void loop() {
  delay(100);
}

void receiveEvent() {
  while (Wire.available()) {
    int x = Wire.read();
    Serial.print(x);
  }
  Serial.println();
}
