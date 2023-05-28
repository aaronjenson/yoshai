import argparse
import math
import os
import pickle
import secrets
import string
import time

import torch
import torchvision

from controller import XboxController
from dolphin import Dolphin
from mario_dataset import MarioDataset, split_dataset
from mario_net import MarioNet
from virtual_controller import VirtualController


def parse_args():
    parser = argparse.ArgumentParser(prog='YoshAI')
    parser.add_argument('mode', choices=['collect', 'train', 'run', 'dagger', 'reformat'])
    parser.add_argument('-c', '--course', default='luigi-circuit',
                        help='which course to run (must be in saves/ dir, do not include .sav)')
    parser.add_argument('-d', '--data', help='directory (or file) of data (.pkl) to train with')
    parser.add_argument('-b', '--batch_size', default=64, help='batch size to use for training')
    parser.add_argument('-l', '--learning_rate', default=0.01, help='initial learning rate to use for training')
    parser.add_argument('--decay', default=0, help='weight decay to use for training')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='number of epochs for training')
    parser.add_argument('-m', '--model_path', help='path to save or load model')
    parser.add_argument('-g', '--gamma', default=0.9, help='gamma for exponential learning rate decay')
    parser.add_argument('-v', '--save_video', help='save a video of each captured frame while running')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'collect':
        collect(args)
    elif args.mode == 'train':
        train_bc(args)
    elif args.mode == 'run':
        run_model(args)
    elif args.mode == 'dagger':
        print("dagger mode not yet implemented")
    elif args.mode == 'reformat':
        reformat(args)


def collect(args):
    """
    Starts dolphin for data collection. Records game images and user inputs.
    :param args:
    :return:
    """
    dolphin = Dolphin(args.course)
    data = []
    try:

        print("ready to start")
        contr = XboxController()
        while contr.A == 0:
            time.sleep(0.01)

        while True:
            im = dolphin.screenshot()
            controls = contr.read()
            name = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(32))
            name = f'{name}.png'
            torchvision.io.write_png(im, f'bcdata/images/{name}')
            controls.append(name)
            data.append(controls)
            time.sleep(0.03)
    finally:
        dolphin.close()
        if len(data) > 0:
            file = f'bcdata/{args.course}_{len(os.listdir("bcdata"))}.pkl'
            with open(file, 'wb') as f:
                pickle.dump(data, f)


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

    model = MarioNet().to(device)

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


def run_model(args):
    dolphin = Dolphin(args.course)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MarioNet().to(device)
    model.load(args.model_path)

    vcont = VirtualController()
    video = torch.empty((0, 3, 456, 832))
    t = 0
    try:
        vcont.press_a()

        t = time.time()
        while True:
            im = dolphin.screenshot()
            im = torch.reshape(im, tuple([1]) + im.shape)
            if args.save_video is not None:
                video = torch.cat((video, im))
            im = im.to(dtype=torch.float32, device=device)
            model.eval()
            controls = model(im)
            controls = torch.flatten(controls)
            vcont.update(controls)
            print(controls)
    finally:
        t_diff = time.time() - t
        vcont.close()
        dolphin.close()
        if len(video) > 0:
            print(f"saving video to {args.save_video}")
            fps = len(video) // t_diff
            video = video.movedim(1, 3).to(dtype=torch.uint8)
            torchvision.io.write_video(args.save_video, video, fps)


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


if __name__ == '__main__':
    main()
