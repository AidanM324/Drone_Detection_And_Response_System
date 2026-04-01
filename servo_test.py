from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

factory = PiGPIOFactory()

pan = Servo(19, pin_factory=factory)
tilt = Servo(20, pin_factory=factory)

while True:
    print("Center")
    pan.value = 0
    tilt.value = 0
    sleep(2)

    print("Left")
    pan.value = -1
    sleep(1)

    print("Right")
    pan.value = 1
    sleep(1)