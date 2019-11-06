#include "mbed.h"
#include "rtos.h"
 
DigitalOut boardled(LED1);
InterruptIn button(PA_0);
Thread thread_blink;

// memory pool as storage space for queue.
MemoryPool<int, 1> mpool;

// allocat memory to pool.
int * msgNum = mpool.alloc();

// intialize queue as int of size 1.
Queue<int, 1> queue;



bool semState = 0;
void btn_int() {

    *msgNum = (*msgNum + 1) % 8;
    
    // put the number in queue.
    queue.put(msgNum);
}
void blink() {
    
    int * blinkNum;
    while (1) {
    
        // receive queue event.
        osEvent evt = queue.get();
        
        // check if it is a message.
        if (evt.status == osEventMessage)
            blinkNum = (int *)evt.value.p; // receive value.
        
        // blink as many times as the value received.
        for (int i=(*blinkNum); i>0; --i) {
            boardled = 1;
            Thread::wait(250);
            boardled = 0;
            Thread::wait(250);
        }
    }
}
