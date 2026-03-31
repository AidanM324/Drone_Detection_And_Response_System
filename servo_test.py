from gpiozero import Servo
from time import sleep

pan = Servo(19)   # X pin
tilt = Servo(20)  # Y pin

while True:
    pan.value = 0
    tilt.value = 0
    sleep(2)

    pan.value = -1
    sleep(1)

    pan.value = 1
    sleep(1)