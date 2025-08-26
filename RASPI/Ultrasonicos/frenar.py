import RPi.GPIO as GPIO
import time

# Configuración de pines

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Pines Ultrasonicos (dos sensores adelante)
TRIG1 = 11
ECHO1 = 13
TRIG2 = 29
ECHO2 = 31

# Pines de control para el puente H (L298N)
IN1 = 16
IN2 = 18
IN3 = 38
IN4 = 40
ENA = 15  
ENB = 36

# Constantes
DISTANCIA_MINIMA = 75   # cm
VELOCIDAD_MAX = 100     # duty cycle %
INTERVALO = 0.1         # segundos entre mediciones


# Configuración de GPIO
 # Motores
for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
    GPIO.setup(pin, GPIO.OUT)

 # PWM en pines ENA y ENB
pwmA = GPIO.PWM(ENA, 1000)  # 1 kHz
pwmB = GPIO.PWM(ENB, 1000)

pwmA.start(0)
pwmB.start(0)

 # Sensores ultrasónicos
GPIO.setup(TRIG1, GPIO.OUT)
GPIO.setup(ECHO1, GPIO.IN)
GPIO.setup(TRIG2, GPIO.OUT)
GPIO.setup(ECHO2, GPIO.IN)


# Funciones
def medir_distancia(TRIG, ECHO, timeout=0.02):
    # Enviar pulso
    GPIO.output(TRIG, False)
    time.sleep(0.00005)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Esperar a HIGH
    start_time = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start - start_time > timeout:
            return None

    # Esperar a LOW
    start_time = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end - start_time > timeout:
            return None

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2
    return round(distance, 2)

def adelante(velocidad=VELOCIDAD_MAX):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(velocidad)
    pwmB.ChangeDutyCycle(velocidad)

def atras(velocidad=VELOCIDAD_MAX):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(velocidad)
    pwmB.ChangeDutyCycle(velocidad)

def frenar():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

# Programa principal

try:
    while True:
        d1 = medir_distancia(TRIG1, ECHO1)
        d2 = medir_distancia(TRIG2, ECHO2)

        # Promediar las lecturas válidas
        distancias = [d for d in [d1, d2] if d is not None]
        if distancias:
            distancia_media = sum(distancias) / len(distancias)
        else:
            distancia_media = None

        if distancia_media is not None:
            print(f"Distancia promedio frontal: {distancia_media:.2f} cm")
            if distancia_media <= DISTANCIA_MINIMA:
                print("Obstáculo detectado → FRENAR")
                frenar()
            else:
                print("Avanzando...")
                adelante()
        else:
            print("Sensores fuera de rango → FRENAR")
            frenar()

        print("--------------------------")
        time.sleep(INTERVALO)

except KeyboardInterrupt:
    print("Programa detenido por el usuario")

finally:
    frenar()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
