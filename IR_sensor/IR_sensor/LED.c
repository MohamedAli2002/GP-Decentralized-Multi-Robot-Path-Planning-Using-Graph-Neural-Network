/*****************************************************************************************************

	HBE-RoboCAR LED example
	
	Operation:		
		LED operates in compliance with a LED_status(MACRO Constant).
		(The motor does not move.)	
		
	Source explanation (LED.c) :
		1)	Includes AVR Register files and libraries and LED header files. 

		2)	Initializes HBE-RoboCAR each ports.(Robocar_init.c)
			Port B 2bit : Front LED
			Port B 3bit : Rear LED	

		3)	Sets the timer2 overflow for LED. (RBC_Led.c)

				TCCR2	=	0x05;		//	1024 prescaler 
				TCNT2	=	0xff - 80;	//	1/(7.3728MHz(main clock)/1024) * 80 = 0.011s (Timer2 Overflow Occurrence period)	
				TIMSK	|=	1 << TOIE2;	//	Set Timer2 Overflow Interrupt

		4)	LED_status is the variable to inform the operation of the motor.
			Timer2 every 0.5 seconds refers to LED and in compliance with the price LED controls. 

		ex)	When inputs LETF_U in LED_status, LED lights about U-turn of the left side directions.(Twinkles front, rear LED)
			When inputs FRONT in LED_status, LED is executed about the front.(Front LED twinkles only.)

	Actual training method :
		Changes LED_status with macro constants(Robocar_init.h) and practices.
		ex) LED_status=FRONT; 
	
	
********************************************************************************************************/
//	1)
#include<avr/io.h>
#include"avr_lib.h"
#include"Robocar_init.h"
#include"RBC_LED.h"


unsigned char LED_status;

int main(){
	
//	2)
	PORT_init();	
	
//	3)
	Timer2_init();	
	
//	4)
	LED_status=LEFT_U; 

	sei();			

	while(1);		

	return 0;
}
