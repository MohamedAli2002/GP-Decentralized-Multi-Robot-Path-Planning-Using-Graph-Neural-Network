#ifndef _RBC_LED_
#define _RBC_LED_

#include<avr/io.h>
#include<avr/interrupt.h>
#include"avr_lib.h"
#include"Robocar_init.h"


enum{BLINK_ON,BLINK_OFF};

void Timer2_init(); 
//SIGNAL(SIG_OVERFLOW2);
ISR(TIMER2_OVF_vect);
  
#endif
