#ifndef _ROBOCAR_INIT_
#define _ROBOCAR_INIT_

#include<avr/io.h>
#include"avr_lib.h"

//#define SENSITIVE (1.9)
#define SENSITIVE (3)

#define FRONT	0x06	
#define BACK	0x09	
#define LEFT	0x02	
#define RIGHT	0x04	
#define STOP    0x00	
#define LEFT_U	0x0A	
#define RIGHT_U	0x05	
#define B_RIGHT	0x08	
#define B_LEFT	0x01	


#define F_LED	0x04 
#define B_LED 0x08

#define F_LED_ON()	(PORTB |= F_LED)
#define F_LED_OFF()	(PORTB &= ~F_LED)

#define B_LED_ON()	(PORTB |= B_LED)
#define B_LED_OFF()	(PORTB &= ~B_LED)


#define BEEP_ON()	(PORTE |= 0x04)
#define BEEP_OFF()	(PORTE &= ~0x04)


#define CLK		0x02
#define Din		0x04
#define SYNC	0x01
#define DAC_PORT	PORTG
#define DAC_DDR		DDRG

void PORT_init();

void Convert_sDAC(u08 tmp); 

#endif
