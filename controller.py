import time

from inputs import get_gamepad
import math
import threading

DEADZONE = 0.2


def deadzone(val):
    if abs(val) < DEADZONE:
        return 0.0
    return val


class XboxController(object):
    """
    Tracks the values of an xbox controller using the inputs library
    Credit to @kevinhughes27 on GitHub
    https://github.com/kevinhughes27/TensorKart/blob/455d28083806814377b60567b559f97597ccd7e0/utils.py#L41
    """
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def read(self):  # return the buttons/triggers that you care about in this methode
        return [
            self.RightTrigger,
            self.RightBumper,
            self.LeftTrigger,
            self.LeftBumper,
            self.LeftJoystickX
        ]

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = deadzone(event.state / XboxController.MAX_JOY_VAL)  # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = deadzone(event.state / XboxController.MAX_JOY_VAL)  # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = deadzone(event.state / XboxController.MAX_JOY_VAL)  # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = deadzone(event.state / XboxController.MAX_JOY_VAL)  # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = deadzone(event.state / XboxController.MAX_TRIG_VAL)  # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = deadzone(event.state / XboxController.MAX_TRIG_VAL)  # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.Y = event.state  # previously switched with X
                elif event.code == 'BTN_WEST':
                    self.X = event.state  # previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state


if __name__ == '__main__':
    joy = XboxController()
    while True:
        print(joy.read())
        time.sleep(0.033)
