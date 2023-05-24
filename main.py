import time
import subprocess

import Xlib.X
from Xlib.display import Display
from PIL import Image
import pickle

from controller import XboxController

DOLPHIN_COMMAND = "dolphin-emu"
GAME_FILE = "mario_kart.nkit.iso"
SAVE_FILE = "saves/moo-moo-meadows.sav"
GAME_NAME = "Mario Kart Wii"


def main():
    dolphin_args = [DOLPHIN_COMMAND, "-e", GAME_FILE, "-s", SAVE_FILE]
    dolphin = subprocess.Popen(dolphin_args)
    time.sleep(8)
    data = []
    try:
        display = Display()
        root = display.screen().root
        dolphin_window = find_window(root, dolphin_criteria)
        if dolphin_window is None:
            raise Exception("Mario Kart Wii window not found in X11 tree")

        contr = XboxController()
        while contr.A == 0:
            time.sleep(0.01)

        while True:
            im = screenshot(dolphin_window)
            controls = contr.read()
            controls.append(im)
            data.append(controls)
            print(controls)
            time.sleep(0.03)
    finally:
        dolphin.terminate()
        if len(data) > 0:
            file = open('bcdata/data.pkl', 'wb')
            pickle.dump(data, file)



def dolphin_criteria(window):
    name = window.get_wm_name()
    if name is not None and type(name) == str:
        return GAME_NAME in name
    return False


def find_window(window, condition):
    if condition(window):
        return window
    children = window.query_tree().children
    for w in children:
        res = find_window(w, condition)
        if res is not None:
            return res
    return None


def screenshot(window, show=False):
    t = time.time()
    geom = window.get_geometry()
    w, h = geom.width, geom.height
    raw = window.get_image(0, 0, w, h, Xlib.X.ZPixmap, 0xffffffff)
    im = Image.frombytes("RGB", (w, h), raw.data, "raw", "BGRX")
    print(f"screenshot took {time.time() - t}s")
    if show:
        im.show()
    return im


if __name__ == '__main__':
    main()
