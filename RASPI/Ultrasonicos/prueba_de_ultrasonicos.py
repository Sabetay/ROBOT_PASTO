import RPi.GPIO as GPIO
import os
import time

GPIO.setmode(GPIO.BCM)

TRIG1 = 17
ECHO1 = 27
TRIG2 = 5
ECHO2 = 6

GPIO.setup([TRIG1, ECHO1, TRIG2, IN4, ECHO2], GPIO.IN)