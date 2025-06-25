/**************************************************************************************************
	HBE - RoboCAR infrared sensor example
	
	Operation:
		Infrared sensor value is Confirms from the Hyper terminal.

	
	Source explanation :
		1)	Stores ADC, the variable which value
		2)	Initializes UART communications with 115200 bps.
		3)	Initializes serial DAC output voltage with 0V. 			
		4)	Store Infrared sensor value to variable. 
		5)	Selects the value which is smallest the biggest value for ADC.
		6)	The calculated voltage is output serial DAC.
		7)	Outputs the infrared sensor value of port c with Hyper terminal.
			
************************************************************************************************/


#include<avr/io.h>
#include"avr_lib.h"
#include"Robocar_init.h"

unsigned char LED_status; 


int main(){
	
	u08	lineValue;
	
//	1)	
	u16 infr[4]={0,};
	
	PORT_init();
	
//	2)
	init_UART1(UART_115200);
	Printf_Attach(uart1_Str); 	

//	3)
	Convert_sDAC(0);
	Convert_sDAC(0);
	
//	4)
	infr[0]=ADC_Convert(4);		
	infr[1]=ADC_Convert(5);		
	infr[2]=ADC_Convert(6);		
	infr[3]=ADC_Convert(7);			
			
//	5)	
	u08 i,j;
	u16 ADC_max =infr[0];
	u16 ADC_min =infr[0];
	u16 V_ref = 0;
	
	for(i=1;i<4;i++){
		if(infr[i]>=ADC_max)
			ADC_max = infr[i];
		if(infr[i]<=ADC_min)
			ADC_min = infr[i];
	}
	

//	6)
	ADC_max = (ADC_max-ADC_min)*SENSITIVE;	

	V_ref = ADC_max*0.128;
	
	Convert_sDAC(V_ref);	
	Convert_sDAC(V_ref);	
	
	i = 0;

//	7)
	while(1){
		lineValue=PINC ;		
		
		Printf("\n\r %d) lineValue : ",i++);

		for(j=0;j<8;j++){
			if(lineValue & 0x80)
				putch_u1('1');
			else
				putch_u1('0');
			
			lineValue<<=1;
		}

		ms_delay(200);
	}



	while(1);


	return 0;
}

