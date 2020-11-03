#include "mbed.h"
#include "rtos.h"
 
DigitalOut boardled(LED1);
InterruptIn button(PA_0);
Thread thread_blink;

Semaphore sem(1);

bool semState = 0;
void btn_int() {
    // release the semaphore as
    // button is pressed.
    sem.release();
}
void blink() {
    while (1) {
    
        // wait for button to
        // release the semaphore
        sem.wait();
        
        
        // blink 8 times as the
        // semaphore is released
        for (int i=8; i>0; --i) {
            boardled = 1;
            Thread::wait(250);
            boardled = 0;
            Thread::wait(250);
        }
    }
}

 
int main() {

    // btn_int as rising edge interrupt.
    button.rise(&btn_int);
    
    // initialize thread.
    thread_blink.set_priority(osPriorityHigh);
    thread_blink.start(blink);
}
