#include"Robocar_init.h"

void PORT_init()
{
	
	DDRA |=	0x0F; 
    PORTA =0x00; 

	DDRB = 0xff;   
    PORTB =	0x01; 
	  

	DDRC = 0x00; 
	PORTC =0x00; 
	
	DDRE	|=	0x1C; 
	PORTE	|=	0x18;
		
	DDRD	|=	0xc0; 
	PORTD	|=	0xc0;

	DDRF	=	0x00;
	PORTF	=	0x00;	
	
	DDRG = 0xff;

}

void Convert_sDAC(u08 tmp){
	u16 data=0,mask;
	u08 i;

	mask=0x8000;
	
	data |= tmp<<4; 	//	xx mode	tmp xxxx
						//	  2bit+ 8bit	
						// mode(00) -> power down mode(normal mode) 
						
	
						// write sequence
	DAC_PORT	|=	SYNC;
	asm("nop");
	DAC_PORT	|=	(CLK);
	asm("nop");
	DAC_PORT	&=	~(CLK);
	asm("nop");
	DAC_PORT	&=	~SYNC;
	asm("nop");
	
	for(i=0;i<16;i++){	
		
		DAC_PORT	|=	(CLK);
		asm("nop");
	
		if(data & mask) 
			DAC_PORT	|=	Din;
		
		else			
			DAC_PORT	&=	~(Din);
				
		asm("nop");
		DAC_PORT	&=	~(CLK);
		asm("nop");
		data<<=1;	

	}//end for
	

}//end func

