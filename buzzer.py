from gpiozero import TonalBuzzer
from gpiozero.tones import Tone
from time import sleep

buzzer = TonalBuzzer(18)

buzzer.play(Tone("A4"))  # play tone
sleep(1)
buzzer.stop()