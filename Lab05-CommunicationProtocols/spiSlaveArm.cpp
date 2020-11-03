#include "mbed.h"

SPISlave spi_slave(PA_7, PA_6, PA_5, PA_4);  // mosi, miso, sclk, ssel

int main() {    
   spi_slave.reply(0x00);  //first reply
   while(1) {
       if(spi_slave.receive()) {
           int v = spi_slave.read();  // Read byte from master
           printf("Value read is %d.\n\r", v);
       }
   }
}
