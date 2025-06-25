#include"avr_lib.h"

void Printf_Attach(void (*userFunc)(u08 *data) ){
	print_Data = userFunc;

}
void us_delay(unsigned short time_us){
	
	register u08 i;

	for(i=0; i < time_us ; i++){		//4 
		asm volatile("PUSH R0");		//2
		asm volatile("POP R0");			//2	 = 8cycle = 1.08us
									
	}

}
void ms_delay(u16 time_ms){
	register u16 i;

	for(i=0; i < time_ms ; i++){	
		us_delay(250);	
		us_delay(250);	
		us_delay(250);	
		us_delay(250);	// 1us * 1000 = 1ms
		
	}
}

//spi

void AIC23B_spi_write(u08 cntl,u08 data){
	

	SPDR = cntl;
	while((SPSR & 0x80)==0x00);
	PORTB	&=	~(0x01);

	SPDR	=	data;
	while((SPSR & 0x80)==0x00);
	PORTB	|=0x01;

	ms_delay(10);

}

void spi_write(u08 data){

	SPDR	=	data;
	while((SPSR & 0x80)==0x00);

}

u08 cal_bit_place(u08 bit){ 
	char i=1;

	while(1){
		if(bit>0x01)
			bit=bit>>1;
		else if(bit==0x01)
			return i;
		else
			return 0;

			i++;
	}
}



void sprint_u16(u16 data, u08 * str){
			
	if(data>10000){
		str[0] = (data/10000)+'0';
		str[1] = ((data/1000)%10)+'0';
		str[2] = ((data/100)%10)+'0';
		str[3] = ((data/10) %10 )+'0';	
		str[4] = (data%10)+'0';	
		str[5] = '\0';	
	}
	else if(data>1000){
		str[0] = (data/1000)+'0';
		str[1] = ((data/100)%10)+'0';
		str[2] = ((data/10)%10)+'0';
		str[3] = (data%10)+'0';	
		str[4] = '\0';	
	}
	else if(data>100){
		str[0] = (data/100)+'0';
		str[1] = ((data/10)%10)+'0';		
		str[2] = (data%10)+'0';	
		str[3] = '\0';	
	}
	else if(data>10){
		str[0] = ((data/10))+'0';
		str[1] = (data%10)+'0';	
		str[2] = '\0';	
	}
	else if(data>0){
		str[0] = (data)+'0';	
		str[1] = '\0';	
	}
	else	
		str[0] = '\0';	
}

void SPI_Init(){
	DDRB  |= 0x07; // MISO=input, etc.=output
					// PB3(MISO), PB2(MOSI), PB1(SCK), PB0(/SS)    
	PORTB |= 0x01;  // CS=1, waiting for SPI start
	
	SPCR  = 0x50; // SPI mode 0, 4MHz
	SPSR  = 0x01; // SPI2X=1		
}

void init_UART1(u08 bps)  
{

     DDRD &= 0xFB;
	  DDRD |= 0x08;


     UCSR1A=0x00;
     UCSR1B=0x18;  
     UCSR1C=0x06;  

     UBRR1H=0x00;
     UBRR1L=bps;   

}
void init_UART0(u08 bps)   
{

    DDRE &= ~0x01;
	DDRE |= 0x02;  


     UCSR0A=0x00;
     UCSR0B=0x18;   
     UCSR0C=0x06;   

     UBRR0H=0x00;
     UBRR0L=bps;   

}

u08 getch_u1(){

	unsigned char data;

	while((UCSR1A & 0x80)==0);

	data=UDR1;

	UCSR1A	|=	0x80;

	return data;
}

void putch_u1(unsigned char data)
{
	while((UCSR1A & 0x20)==0x00);
	UDR1=data;
  
	UCSR1A |= 0x20;       
}

u08 getch_u0(){

	unsigned char data;

	while((UCSR0A & 0x80)==0);

	data=UDR0;

	UCSR0A	|=	0x80;

	return data;
}

void putch_u0(unsigned char data)
{
	while((UCSR0A & 0x20)==0x00);
	UDR0=data;
  
	UCSR0A |= 0x20;         // Å¬¸®¾î UDRE0
}

void uart1_Str(char *str){
	u08 i=0;
	while(str[i]!='\0')
		putch_u1(str[i++]);
}

void uart0_Str(char *str){
	u08 i=0;
	while(str[i]!='\0')
		putch_u0(str[i++]);
}

u16 square1(u08 x,u08 y){
	u08 i;
	u16 _ret=1;
	for(i=0;i<y;i++)
		_ret *= x;
	return _ret;
}


void Printf(unsigned char *fmt, ...){
	
		
	unsigned char *str,*ptr;
	unsigned char i;
	va_list arg;     	
	
	unsigned char ptrBuf[32];
	unsigned int data;
	unsigned char tmp;
	unsigned char base;
	float decimal;
	unsigned char dec_cnt=0;
	unsigned char buf[64]={0};
	

	str = buf;
			
	
	va_start ( arg, fmt );
		
	for(;*fmt;*fmt++){		//  *fmp != '\0' 
		if(*fmt != '%' ){	//
			*str++ = *fmt;
			continue;
		}
		fmt++;

		if(*fmt >= '0' && *fmt <= '9'){
			dec_cnt = *fmt -'0';
			fmt++;
		}
		else 
			dec_cnt=3;

		i=0;
			switch(*fmt){
				case 'c':					
					*str++ = va_arg(arg,char);
				continue;

				case 's':
					ptr = va_arg(arg,char*);	
					while((*str++ = *ptr++)!='\0');
					str--;
				continue;
			
				case 'd':					
					data = va_arg(arg,unsigned int);	
					base = 10;
				break;
				
				case 'b':					
					data = va_arg(arg,unsigned int);	
					base = 2;
				break;

				case 'x':					
					data = va_arg(arg,unsigned int);	
					base = 16;
				break;

				case 'f':
					decimal =  va_arg(arg,float);				
					
					if(decimal >= 1.0){ 
						
						ptr = &ptrBuf[31];
						data = 	decimal;	
				
						do{			
							tmp = data%10 + '0';
							*ptr-- = tmp;
							i++;		

						}while(data/=10);
	
					while(i--)
			 			*str ++ = *(++ptr);
						}//end 1.0
			
						else 
							*str++='0';					
						if(dec_cnt==0)	continue;				

						dec_cnt = (dec_cnt>3?3:dec_cnt);

						*str ++ = '.'; 
						
						data = 	decimal; 										
						data = (decimal*square1(10,dec_cnt)) - (data*square1(10,dec_cnt));	
					
					
						ptr = &ptrBuf[31];
									
						for(i=0;i<dec_cnt;i++){					
							tmp = (data)%10 + '0';
							data/=10;
							*ptr-- = tmp;								
						}

						for(i=dec_cnt;i>0;i--)
				 			*str ++ = *(++ptr);
					
					continue; //end 'f' 
								
					default:
					continue;
			}//END CASE
	


	
		ptr = &ptrBuf[31];		
		
		do{
			tmp = data%base;
	
			if(tmp >9)
				tmp += 'A'-10;
			else
				tmp += '0';

			*ptr-- = tmp;
			i++;
			
			
			
		}while(data/=base);
	
		while(i--)
			 *str ++ = *(++ptr);
		
	}//end for
	
	*str ='\0'; 
			
	print_Data(buf);
	
	va_end(arg);
}//end func


u16 ADC_Convert(u08 ch){ 
	
		u16 _ret=0;
		
		ADMUX=0xc0 | ch; //2.56v   
 		ADCSR=0xa7;     //128 
   		ADCSR |= 0x40; //ADC start conversion

		while( (ADCSR&0x10)==0x00);		
		ADCSR |= 0x10;
	
		_ret =	ADCL & 0xff;
		_ret |=	ADCH<<8 ; 	
		
		ADMUX=0x0;   
 		ADCSR=0x0;   
 		ms_delay(10);
 		
		return _ret;		
		
	
}


u08 Strcpy(void *dst,const void* src,u08 len){

	while(len--){
		*(char*)dst = *(char*)src;
		dst = (char*)dst+1;
		src = (char*)src+1;
	}


	return 0;
}


