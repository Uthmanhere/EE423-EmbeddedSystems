# Introduction
## RTOS
An operating system designed to use specifically in real-time systems. RTOS consists of following main components in terms of management and synchronization:
- Scheduler
- Communication Mechanisms
- Critical Region Mechanisms
- Memory Management
- Timing Services
- Peripheral Drivers
- Protocol Stack
- File System
- Device Management

## [Scheduling](https://www.freertos.org/implementation/a00005.html)
### Scheduler
The scheduler is the part of the kernel responsible for deciding which task should be executing at any particular time. The kernel can suspend and later resume a task many times during the task lifetime. 
### Scheduling Policy
The scheduling policy is the algorithm used by the scheduler to decide which task to execute at any point in time.
### Scheduler Types
- Run-to-Complete
- Time Slicing
- Round Robin
- Priority
    - Preemptive
    - Non-Prememptive
## Synchronization
There are several types of synchronization primitives in an RTOS to be used according the requirements. These primitives are named as following:

 - Mutex
 - Semaphore
 - Queue

Although concept is the same but these constructs vary in implementation for different real-time operating systems.
### Mutex
A program object that is created so that multiple program thread can take turns sharing the same resource
### Semaphore
Semaphore is simply a variable which is non-negative and shared between threads. This variable is used to solve the critical section problem and to achieve process synchronization in the multiprocessing environment

Refer to [Task C](./sem) for an implementation perspective.
### Queue
A queue is a FIFO (First In First Out) type buffer where data is written to the end (tail) of the queue and removed from the front (head) of the queue. It is also possible to write to the front of a queue. 

Refer to [Task D](#q) for an implementation perspective.
# Lab Tasks
## Task A: Processes
### Task
Import RTOS project from the [the link](https://os.mbed.com/users/mbed_official/code/rtos_basic/). Run the code in [proc.c](./proc.c) and understand initialization of processes in MBed OS.
## Task B: Basic Process Synchronization
Run the code in [syncProc.c](./syncProc.c) and observe the difference in output.
## Task C: Semaphores <a name="sem"></a>
Refer to cod in [sem.c](./sem.c).
### Explanation
Button interrupt releases the semaphore \texttt{blink} is waiting for, and allow it to blink a specified number of times. THe \texttt{blink} turns back to wait again.
## Task D: Queues <a name="q"></a>
Run the [queue.c](./queue.c) code to observe semaphores in implementation.
### Explanation
Button interrupts increments a count and send the value over queue. The _blink_ recieves value and blinks LED as many times as specified. Then _blink_ goes back to wait for next message from queue.

## Task E: Bounce Game
Instantiate the ball and background processes as threads while button interrupts as a rising edge interrupt. Run the code and try to play the game. Are you satisfied with the synchronization of graphics? Try to improve it.
### Code Snippets
You can find the code snippets in [rtos1.c](./rtos1.c).
#### Button Interrupt
```C
void btn_int()
{
    bounce_up = 1;
}
```
#### Ball
```C
void ball_thread()
{
    led1 = 1;
  
    // set font
    BSP_LCD_SetFont(&Font20);
    
    // set LCD background to white
    lcd.Clear(LCD_COLOR_WHITE);
    lcd.SetBackColor(LCD_COLOR_WHITE);
    
    // initial state of ball
    y_movement = 0;
    bool bounce_down = 0;
    
    while(1)
    {
        if (x_movement >= 220 && y_movement <= 70)  // collision!
        {
            // Prompte the end of the game.
            lcd.Clear(LCD_COLOR_WHITE);
            lcd.DisplayStringAt(0, LINE(1), (uint8_t *)"GAME OVER", CENTER_MODE);
            
            // reset positions.
            x_movement = 0;
            y_movement = 0;
            bounce_up = 0;
            bounce_down = 0;
            Thread::wait(5000);
        }
        else if (bounce_up)  // ball going up
        {
            lcd.SetTextColor(LCD_COLOR_RED);
            lcd.FillCircle(20+y_movement, 40, 20);
            y_movement += 3;
            if (y_movement >= 120)  // the highest ball can go.
            {
                bounce_up = 0;
                bounce_down = 1;
            }
            Thread::wait(250);
        }
        else if (bounce_down)  // ball going down
        {
            lcd.SetTextColor(LCD_COLOR_RED);
            lcd.FillCircle(20+y_movement, 40, 20);
            y_movement -= 5;
            if (y_movement <= 0)  // touching the ground.
                bounce_down = 0;
            Thread::wait(250);
        }
        else  // default position of the ball
        {
            lcd.Clear(LCD_COLOR_WHITE);
            lcd.SetBackColor(LCD_COLOR_WHITE);
            lcd.SetTextColor(LCD_COLOR_RED);
            lcd.FillCircle(20, 40, 20);      
            Thread::wait(250);
        }

    }
}
```
#### Background
```C
void bg_thread()
{
    // color and shape of obstacle.
    lcd.SetTextColor(LCD_COLOR_BLACK);
    lcd.FillTriangle(0, 75, 0, 250, 275, 300);
    
    // initial position
    x_movement = 0;
    
    while (1)
    {
        if (x_movement >= 250)  // the obstacle is through.
            x_movement = 0;
        
        // obstacle approaching.
        lcd.Clear(LCD_COLOR_WHITE);
        lcd.SetTextColor(LCD_COLOR_BLACK);
        lcd.FillTriangle(0,75,0,250-x_movement, 275-x_movement, 300-x_movement);
        x_movement += 5;
        Thread::wait(250);
    }
}
```
## Task F: Bounce Game
Improve the bounce game from last lab using synchronization primitives observed here such that graphics are more aligned this time. Following hints might help:

 - Use a separate thread to manage LCD resource.
 - Pass position arguments for ball and obstacle to thread responsible for LCD resource.
 - This thread-to-thread communication can be achieved through queues.
 - Use semaphores to avoid unaligned execution of threads.

The official documentation of [semaphors](https://os.mbed.com/docs/mbed-os/v5.14/apis/semaphore.html) and [queues](https://os.mbed.com/docs/mbed-os/v5.14/apis/queue.html) might help.

