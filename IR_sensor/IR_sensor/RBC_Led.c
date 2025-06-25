#include"RBC_Led.h"

unsigned char timer2Cnt=0;
unsigned char LED_value=0;
unsigned char LED_mode=0;
unsigned char Blink=0;

extern unsigned char LED_status;

/****************************************************************
	Timer2 Overflow Interrupt 
****************************************************************/
void Timer2_init(){

	TCCR2	=	0x05;		
	TCNT2	=	0xff - 80;	
	
	TIMSK	|=	1 << TOIE2;	
	TIFR	|=	1 << TOV2;	
}


/****************************************************************
	Timer2 Overflow ISR	
****************************************************************/
//SIGNAL(SIG_OVERFLOW2) 
ISR(TIMER2_OVF_vect)
{
  
	cli();
		 TCNT2	=	0xff - 80;		 
		 
		 timer2Cnt++;
			
		if(timer2Cnt==45){ //0.5s = 0.011s * 45
		  	timer2Cnt=0;
				
			switch(LED_status){ 
				
				case FRONT:
				case LEFT:
				case RIGHT:
					LED_value=0x04;
					Blink = BLINK_OFF;
				break;
			
				case BACK:
				case B_LEFT	:
				case B_RIGHT :
					LED_value=0x08;
					Blink = BLINK_ON;
					break;
			
				case LEFT_U:
				case RIGHT_U:
					LED_value=0x0c;
					Blink = BLINK_ON;
					break;
			
							
				default:
					PORTB &= ~0x0C;
					LED_value=0x00;
					Blink = BLINK_OFF;
				break;
			}//end switch

		
		if(LED_mode==0){
			if(Blink == BLINK_ON)
				PORTB&= ~0x0C;
			
				LED_mode=1;
		}//end if
		
		else{						
			PORTB |= LED_value;
			LED_mode=0;	
		}//end else
	
 	}//end if
		 	
	sei();

}
