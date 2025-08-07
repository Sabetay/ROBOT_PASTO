import RPi.GPIO as GPIO
import time

# Modo de numeración de pines
GPIO.setmode(GPIO.BCM)

# Pines de control para el L298N
IN1 = 17
IN2 = 23
IN3 = 24
IN4 = 25
ENA = 5
ENB = 6

# Configurar pines como salida
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# PWM en pines ENA y ENB
pwmA = GPIO.PWM(ENA, 1000)  # 1 KHz
pwmB = GPIO.PWM(ENB, 1000)

# Iniciar PWM con duty cycle 0 (apagado)
pwmA.start(0)
pwmB.start(0)

def adelante(velocidad=100):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(velocidad)
    pwmB.ChangeDutyCycle(velocidad)

def atras(velocidad=100):
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

def limpiar():
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()

# Ejemplo de uso
try:
    print("Avanzando...")
    adelante(80)
    time.sleep(3)

    print("Deteniendo...")
    frenar()
    time.sleep(1)

    print("Atrás...")
    atras(80)
    time.sleep(3)

    print("Frenando...")
    frenar()

except KeyboardInterrupt:
    print("Interrumpido por el usuario")

finally:
    limpiar()
