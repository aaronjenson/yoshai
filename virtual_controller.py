import os
import time

import uinput


class VirtualController:
    def __init__(self):
        # make sure uinput module is running
        os.system("modprobe uinput")
        self.device = uinput.Device([
            uinput.ABS_Y + (0, 255, 0, 0),
            uinput.ABS_X + (0, 255, 0, 0),
            uinput.ABS_RY + (0, 255, 0, 0),
            uinput.ABS_RX + (0, 255, 0, 0),
            uinput.ABS_Z + (0, 255, 0, 0),
            uinput.ABS_RZ + (0, 255, 0, 0),
            uinput.BTN_TL,
            uinput.BTN_TR,
            uinput.BTN_SOUTH,
            uinput.BTN_NORTH,
            uinput.BTN_WEST,
            uinput.BTN_EAST,
            uinput.BTN_THUMBL,
            uinput.BTN_THUMBR,
            uinput.BTN_SELECT,
            uinput.BTN_START,
            uinput.BTN_TRIGGER_HAPPY1,
            uinput.BTN_TRIGGER_HAPPY2,
            uinput.BTN_TRIGGER_HAPPY3,
            uinput.BTN_TRIGGER_HAPPY4
        ])

    def update(self, controls):
        rt = controls[0]
        rb = controls[1]
        lt = controls[2]
        lb = controls[3]
        x = controls[4]

        rt = map(rt, in_low=0)
        rb = discretize(rb)
        lt = map(lt, in_low=0)
        lb = discretize(lb)
        x = map(x)

        self.device.emit(uinput.ABS_RZ, rt)
        self.device.emit(uinput.BTN_TR, rb)
        self.device.emit(uinput.ABS_Z, lt)
        self.device.emit(uinput.BTN_TL, lb)
        self.device.emit(uinput.ABS_X, x)

    def press_a(self):
        self.device.emit(uinput.BTN_SOUTH, 1)
        time.sleep(0.25)
        self.device.emit(uinput.BTN_SOUTH, 0)

    def close(self):
        self.device.destroy()


def map(val, in_low=-1, in_high=1, out_low=0, out_high=255):
    """
    Maps given value linearly from one range to another range. Constrains value to out_range
    :param val: value to map
    :param in_low: low end of input range
    :param in_high: high end of input range
    :param out_low: lowest value to output
    :param out_high: highest value to output
    :return: value mapped to new range (as int)
    """
    val = max(val, in_low)
    val = min(val, in_high)

    val += in_low
    val /= (in_high - in_low)
    val *= (out_high - out_low)
    val -= out_low

    return int(val)


def discretize(val):
    '''
    Constrain val to either 0 or 1 (whichever is closest, 0.5 rounds up)
    :param val: continuous number
    :return: either 0 or 1
    '''
    if val < 0.5:
        return 0
    return 1


if __name__ == "__main__":
    c = VirtualController()
    print("controller initialized successfully")
    c.update([1, 0, 0, 1, 0.5])
    print("events sent successfully")
    c.close()
    print("controller closed, test successful")
