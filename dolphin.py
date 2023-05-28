import struct
import subprocess
import time

import Xlib.X
import torch
from Xlib.display import Display


DOLPHIN_COMMAND = "dolphin-emu"
GAME_FILE = "mario_kart.nkit.iso"
GAME_NAME = "Mario Kart Wii"


class Dolphin:
    def __init__(self, course):
        args = [DOLPHIN_COMMAND, '-e', GAME_FILE, '-s', f'saves/{course}.sav']
        self.process = subprocess.Popen(args)
        time.sleep(3)
        display = Display()
        root = display.screen().root
        self.window = find_window(root)
        if self.window is None:
            self.close()
            raise Exception(f"{GAME_NAME} window not found in X11 tree")

    def screenshot(self):
        geom = self.window.get_geometry()
        w, h = geom.width, geom.height
        raw = self.window.get_image(0, 0, w, h, Xlib.X.ZPixmap, 0xffffffff)
        d = struct.unpack('B' * len(raw.data), raw.data)
        t = torch.tensor(d)
        i = t.reshape((h, w, 4))
        i = i.movedim(2, 0)
        i = i.to(dtype=torch.uint8)[:3, :, :]
        i = i.flip(0)
        return i

    def close(self):
        self.process.terminate()


def find_window(window):
    name = window.get_wm_name()
    if name is not None and type(name) == str and GAME_NAME in name:
        return window
    children = window.query_tree().children
    for w in children:
        res = find_window(w)
        if res is not None:
            return res
    return None


if __name__ == '__main__':
    import torchvision

    dolphin = Dolphin('luigi-circuit')
    im = dolphin.screenshot()
    torchvision.io.write_png(im, 'sample_image.png')
    dolphin.close()
