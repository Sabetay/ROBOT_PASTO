import RPi.GPIO as GPIO
import time

# Configuración de pines (modo físico BOARD)
TRIG1 = 11
ECHO1 = 13
TRIG2 = 29
ECHO2 = 31

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

GPIO.setup(TRIG1, GPIO.OUT)
GPIO.setup(ECHO1, GPIO.IN)
GPIO.setup(TRIG2, GPIO.OUT)
GPIO.setup(ECHO2, GPIO.IN)

def medir_distancia(TRIG, ECHO, timeout=0.02):
    # Enviar pulso de 10us al TRIG
    GPIO.output(TRIG, False)
    time.sleep(0.00005)  # 50 µs
    GPIO.output(TRIG, True)
    time.sleep(0.00001)  # 10 µs
    GPIO.output(TRIG, False)

    # Esperar a que ECHO se ponga en HIGH (inicio pulso)
    start_time = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start - start_time > timeout:
            return None  # sin lectura

    # Esperar a que ECHO se ponga en LOW (fin pulso)
    start_time = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end - start_time > timeout:
            return None  # sin lectura

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2
    return round(distance, 2)

try:
    while True:
        dist1 = medir_distancia(TRIG1, ECHO1)
        dist2 = medir_distancia(TRIG2, ECHO2)

        if dist1 is not None:
            print(f"Distancia sensor 1: {dist1} cm")
        else:
            print("Sensor 1 fuera de rango")

        if dist2 is not None:
            print(f"Distancia sensor 2: {dist2} cm")
        else:
            print("Sensor 2 fuera de rango")

        print("--------------------------")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Medición detenida por el usuario")
    GPIO.cleanup()
