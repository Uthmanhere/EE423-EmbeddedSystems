#include "mbed.h"
#include "rtos.h"
#include "LCD_DISCO_F429ZI.h"

LCD_DISCO_F429ZI lcd;
InterruptIn button(PA_0);
DigitalOut led1(LED1);

bool bounce_up;
int y_movement;
int x_movement;


void btn_int()
{
    bounce_up = 1;
}

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
