#include "mbed.h"

I2C i2c(PB_11, PB_10); 

const int addr8bit = 0x08 << 1;

int main() {
    char cmd[5] = {10, 20, 40, 70};
    while (1) {
        i2c.write(addr8bit, cmd, 4);
        wait(3);
    }
}
