import argparse
import math
import os
import time
import subprocess
import string
import secrets
import torch

import Xlib.X
from Xlib.display import Display
from PIL import Image
import pickle

from mario_dataset import MarioDataset, split_dataset
from mario_net import MarioNet
from controller import XboxController

DOLPHIN_COMMAND = "dolphin-emu"
GAME_FILE = "mario_kart.nkit.iso"
GAME_NAME = "Mario Kart Wii"


def parse_args():
    parser = argparse.ArgumentParser(prog='YoshAI')
    parser.add_argument('mode', choices=['collect', 'train-bc', 'run-bc', 'train-dagger', 'run-dagger', 'reformat'])
    parser.add_argument('-c', '--course', default='luigi-circuit',
                        help='which course to run (must be in saves/ dir, do not include .sav)')
    parser.add_argument('-d', '--data', help='directory (or file) of data (.pkl) to train with')
    parser.add_argument('-b', '--batch_size', default=64, help='batch size to use for training')
    parser.add_argument('-l', '--learning_rate', default=0.01, help='initial learning rate to use for training')
    parser.add_argument('--decay', default=0, help='weight decay to use for training')
    parser.add_argument('-e', '--epochs', default=10, help='number of epochs for training')
    parser.add_argument('-m', '--model_path', help='path to save or load model')
    parser.add_argument('-g', '--gamma', default=0.9, help='gamma for exponential learning rate decay')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'reformat':
        reformat(args)
        exit()

    if args.mode == 'train-bc':
        train_bc(args)
        exit()

    if args.mode == 'collect':
        collect(args)


def collect(args):
    dolphin = start_dolphin(args)
    data = []
    try:
        display = Display()
        root = display.screen().root
        dolphin_window = find_window(root, dolphin_criteria)
        if dolphin_window is None:
            raise Exception("Mario Kart Wii window not found in X11 tree")

        print("ready to start")
        contr = XboxController()
        while contr.A == 0:
            time.sleep(0.01)

        while True:
            im = screenshot(dolphin_window)
            controls = contr.read()
            name = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(32))
            name = f'{name}.png'
            im.save(f'bcdata/images/{name}')
            controls.append(name)
            data.append(controls)
            time.sleep(0.03)
    finally:
        dolphin.terminate()
        if len(data) > 0:
            file = f'bcdata/{args.course}_{len(os.listdir("bcdata"))}.pkl'
            with open(file, 'wb') as f:
                pickle.dump(data, f)


def reformat(args):
    files = []
    if os.path.isfile(args.data):
        files.append(args.data)
    elif os.path.isdir(args.data):
        for f in os.listdir(args.data):
            files.append(os.path.join(args.data, f))
    files = [f for f in files if os.path.isfile(f) and '.pkl' in f]
    for file in files:
        data = None
        with open(file, 'rb') as f:
            data = pickle.load(f)
            for d in data:
                name = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(32))
                name = f'{name}.png'
                d[5].save(f'bcdata/images/{name}')
                d[5] = name
        with open(file, 'wb') as f:
            pickle.dump(data, f)


def start_dolphin(args):
    dolphin_args = [DOLPHIN_COMMAND, "-e", GAME_FILE, "-s", f"saves/{args.course}.sav"]
    dolphin = subprocess.Popen(dolphin_args)
    time.sleep(3)
    return dolphin


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
    geom = window.get_geometry()
    w, h = geom.width, geom.height
    raw = window.get_image(0, 0, w, h, Xlib.X.ZPixmap, 0xffffffff)
    im = Image.frombytes("RGB", (w, h), raw.data, "raw", "BGRX")
    if show:
        im.show()
    return im


def train_bc(args):
    if args.data is None:
        print("No data provided for training")
        exit(1)
    files = []
    if os.path.isdir(args.data):
        for f in os.listdir(args.data):
            path = os.path.join(args.data, f)
            if os.path.isfile(path):
                files.append(path)
    elif os.path.isfile(args.data):
        files.append(args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training using device: {device}")

    model = MarioNet((832, 456), 4).to(device)

    train_ds, test_ds = split_dataset(MarioDataset(files, device=device))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

    best_loss = math.inf
    for e in range(args.epochs):
        total_loss = 0.
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0 and i > 0:
                print(f"epoch {e + 1}/{args.epochs}, batch {i} avg train loss: {total_loss / i}")

        test_loss = 0.
        for i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(
            f"epoch {e + 1}/{args.epochs} avg train loss: {total_loss / len(train_loader)}, avg test loss: {avg_test_loss}")
        scheduler.step()
        if args.model_path is not None and avg_test_loss < best_loss:
            print(f"saving model to {args.model_path}")
            best_loss = avg_test_loss
            model.save(args.model_path)


if __name__ == '__main__':
    main()
