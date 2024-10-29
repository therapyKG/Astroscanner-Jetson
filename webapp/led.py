import Jetson.GPIO as gpio

class LED:
    def __init__(self):
        #self.Relay_Ch1 = 26    #BCM
        #self.Relay_Ch2 = 20    #BCM
        self.Relay_Ch3 = 21     #BCM
        self.led_high = False
        gpio.setmode(gpio.BCM)
        gpio.setup(self.Relay_Ch3, gpio.OUT, initial = gpio.LOW)
        #gpio.output(self.Relay_Ch3, gpio.HIGH)

    def is_active(self):
        return self.led_high

    def toggle(self):
        if self.led_high:
            gpio.output(self.Relay_Ch3, gpio.LOW)
            self.led_high = False
            print("LED OFF!")
        else:
            gpio.output(self.Relay_Ch3, gpio.HIGH)
            self.led_high = True
            print("LED ON!")

    def cleanup(self):
        gpio.cleanup(self.Relay_Ch3)
        print("RELAY PIN CLEANED.")