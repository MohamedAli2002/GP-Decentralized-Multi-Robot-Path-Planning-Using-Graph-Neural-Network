#include <avr/io.h>
#include "avr_lib.h"
#include "Robocar_init.h"

#define BASE_SPEED 550
#define SEN_OF_IR 6

u08 lineValue, pin_binary;
u08 black = 0;
u08 white = 1;
u08 fb = 0;

u08 ir_line_detection(u08 pin_number) {
	lineValue = PINC;
	pin_binary = 0x01 << pin_number;
	return (lineValue & pin_binary) ? 1 : 0;
}

void setup_DAC(uint8_t code) {
	DDRG |= (1 << PG0) | (1 << PG1) | (1 << PG2);
	PORTG &= ~(1 << PG0);
	for (int8_t i = 7; i >= 0; i--) {
		PORTG = (PORTG & ~(1 << PG2)) | ((code & (1 << i)) ? (1 << PG2) : 0);
		PORTG |= (1 << PG1);
		PORTG &= ~(1 << PG1);
	}
	PORTG |= (1 << PG0);
}

void Timer1_init() {
	TCCR1A |= 0xA3;
	TCCR1B |= 0x02;
	TCCR1C = 0x00;
	TCNT1 = 0x0000;
	OCR1A = 0x0000;
	OCR1B = 0x0000;
}

void PWM_change(char OCR1x, unsigned int value) {
	switch (OCR1x) {
		case 'R': OCR1A = value; break;
		case 'L': OCR1B = value; break;
		case 'A': OCR1A = value; OCR1B = value; break;
		default: break;
	}
}

void Motor_mode(unsigned char mode) {
	PORTA &= 0xF0;
	PORTA |= mode;
}

void move_foreword() {
	fb = 0;
	PWM_change('A', 0x110);
	Motor_mode(FRONT);
}

void deviation_left() {
	fb = 0;
	PWM_change('R', 0x220);
	PWM_change('L', 0x00);
	Motor_mode(FRONT);
}

void deviation_right() {
	fb = 0;
	PWM_change('R', 0x00);
	PWM_change('L', 0x220);
	Motor_mode(FRONT);
}

void move_backword() {
	fb = 1;
	PWM_change('A', 0x150);
	Motor_mode(BACK);
	_delay_ms(2000);
}

void turn_left() {
	fb = 0;
	move_foreword();
	_delay_ms(6000);
	PWM_change('R', 0x220);
	PWM_change('L', 0x220);
	Motor_mode(LEFT_U);
	_delay_ms(8000);
}

void turn_right() {
	fb = 0;
	move_foreword();
	_delay_ms(6000);
	PWM_change('R', 0x220);
	PWM_change('L', 0x220);
	Motor_mode(RIGHT_U);
	_delay_ms(8000);
}

void stop_car() {
	fb = 0;
	Motor_mode(STOP);
}

unsigned char wait_command() {
	putch_u0('0');  // Request command
	return getch_u0();
}

int main() {
	DDRB |= (1 << PB0);
	PORTB |= (1 << PB0);

	setup_DAC(0x80);

	DDRC &= ~(1 << PC1); // Configure IR input

	PORT_init();
	Timer1_init();
	init_UART0(UART_115200);

	Convert_sDAC(0);
	Convert_sDAC(0);

	u16 infr[4] = {
		ADC_Convert(4),
		ADC_Convert(5),
		ADC_Convert(6),
		ADC_Convert(7)
	};

	u16 ADC_max = infr[0];
	u16 ADC_min = infr[0];
	for (u08 i = 1; i < 4; i++) {
		if (infr[i] > ADC_max) ADC_max = infr[i];
		if (infr[i] < ADC_min) ADC_min = infr[i];
	}

	u16 V_ref = (ADC_max - ADC_min) * SEN_OF_IR * 0.128;
	Convert_sDAC(V_ref);
	Convert_sDAC(V_ref);

	u08 test[8] = {0};
	while (1) {
		for (u08 i = 0; i < 8; i++) {
			test[i] = ir_line_detection(i);
		}

		if (test[1] == black && test[2] == black && test[3] == black &&
		test[4] == black && test[5] == black && test[6] == black) {

			stop_car();
			unsigned char command = wait_command();

			if (command == 'f') move_foreword();
			else if (command == 'l') turn_left();
			else if (command == 'r') turn_right();
			else if (command == 'b') move_backword();
			else if (command == 'i') stop_car();
		}

		if ((test[3] == black || test[4] == black) &&
		test[0] == white && test[1] == white &&
		test[2] == white && test[5] == white &&
		test[6] == white && test[7] == white) {

			if (fb == 1) move_backword();
			else move_foreword();
		}

		if ((test[0] == black || test[1] == black || test[2] == black) &&
		test[3] == white && test[4] == white &&
		test[5] == white && test[6] == white &&
		test[7] == white) {
			deviation_left();
		}

		if ((test[5] == black || test[6] == black || test[7] == black) &&
		test[0] == white && test[1] == white &&
		test[2] == white && test[3] == white &&
		test[4] == white) {
			deviation_right();
		}

		ms_delay(50);
	}

	return 0;
}