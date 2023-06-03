import argparse
import math
import os
import pickle
import secrets
import string
import threading
import time

import torch
import torchvision

from controller import XboxController
from data import expand_dir_tree, get_files
from dolphin import Dolphin
from mario_dataset import MarioDataset, split_dataset
from mario_net import MarioNet
from virtual_controller import VirtualController
from weighted_loss import WeightedLoss


def parse_args():
    parser = argparse.ArgumentParser(prog='YoshAI')
    subparsers = parser.add_subparsers(dest='mode', required=True, title='mode')

    c = subparsers.add_parser('collect', help="gather data for training")
    t = subparsers.add_parser('train', help="train model using data")
    r = subparsers.add_parser('run', help="run a trained model")
    d = subparsers.add_parser('dagger', help="gather data using a trained model")
    ref = subparsers.add_parser('reformat', help="for dev purposes only")

    for p in [c, t, r, d]:
        add_course_args(p)

    add_training_args(t)

    for p in [t, r, d]:
        add_pytorch_args(p)

    r.add_argument('--save_video', help='save a video of each captured frame while running')
    t.add_argument('--load_model', help='path to load model from')

    c.set_defaults(func=collect)
    t.set_defaults(func=train_bc)
    r.set_defaults(func=run_model)
    d.set_defaults(func=dagger)
    ref.set_defaults(func=reformat)

    return parser.parse_args()


def add_training_args(parser):
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size to use for training')
    parser.add_argument('-l', '--learning_rate', default=0.01, type=float,
                        help='initial learning rate to use for training')
    parser.add_argument('--decay', default=0, type=float, help='weight decay to use for training')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='number of epochs for training')
    parser.add_argument('-g', '--gamma', default=0.9, type=float, help='gamma for exponential learning rate decay')


def add_course_args(parser):
    parser.add_argument('-d', '--drift_mode', nargs='+', default=['automatic'])
    parser.add_argument('-c', '--course', nargs='+', default=['luigi-circuit'])
    parser.add_argument('-p', '--player', nargs='+', default=['mario'])
    parser.add_argument('-v', '--vehicle', nargs='+', default=['karts'])
    parser.add_argument('-k', '--kart', nargs='+', default=['classic'])


def add_pytorch_args(parser):
    parser.add_argument('--force_cpu', action='store_true', help='forces pytorch to use cpu instead of cuda')
    parser.add_argument('-m', '--model_path', help='path to save or load model')
    parser.add_argument('--steer_only', action='store_true', help='designates to only use the steer input')


def main():
    args = parse_args()
    args.func(args)


def collect(args):
    """
    Starts dolphin for data collection. Records game images and user inputs.
    :param args:
    :return:
    """
    print("starting data collection mode")

    course_opts = args_to_cours_opts(args)
    dirs = expand_dir_tree(course_opts)

    print(f"{len(dirs)} were selected for data collection")

    collect.course_done = False

    contr = XboxController()

    def input_thread():
        input("press enter when done")
        collect.course_done = True

    for config in dirs:
        course = course_opt_to_save_file(config)

        print(f"starting with save file: {course}")
        dolphin = Dolphin(course, virtual_control=False)
        data = []
        collect.course_done = False
        d = os.path.join(*config)
        os.makedirs(d, exist_ok=True)
        image_d = os.path.join(d, 'images')
        os.makedirs(image_d, exist_ok=True)
        print("when you're ready, press A to start the track")
        print("data will be collected and saved while you drive")
        print("when you're done, refocus on this terminal and press enter")
        try:
            while contr.A == 0:
                time.sleep(0.01)

            time.sleep(5)

            thread = threading.Thread(target=input_thread)
            thread.start()

            while not collect.course_done:
                im = dolphin.screenshot()
                controls = contr.read()
                name = random_name(32)
                file = os.path.join(image_d, f'{name}.png')
                while os.path.exists(file):
                    name = random_name(32)
                    file = os.path.join(image_d, f'{name}.png')
                torchvision.io.write_png(im, file)
                controls.append(name)
                data.append(controls)
                time.sleep(0.03)
        finally:
            dolphin.close()
            if len(data) > 0:
                name = random_name(8)
                file = os.path.join(d, f'{name}.pkl')
                while os.path.exists(file):
                    name = random_name(8)
                    file = os.path.join(d, f'{name}.pkl')
                with open(file, 'wb') as f:
                    pickle.dump(data, f)
                print(f'wrote {len(data)} datapoints to {file}')


def random_name(length):
    return ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(length))


def train_bc(args):

    files = get_files(args_to_cours_opts(args), ext='.pkl')
    if len(files) == 0:
        print('no data found')
        exit(1)

    device = torch.device("cuda" if not args.force_cpu and torch.cuda.is_available() else "cpu")
    print(f"training using device: {device}")

    model = MarioNet(outputs=(1 if args.steer_only else 5)).to(device)
    if args.load_model is not None:
        model.load(args.load_model)
    model.freeze()

    if len(files) > 4:
        train_ds = MarioDataset(files[:-1], device=device, steer_only=args.steer_only)
        test_ds = MarioDataset(files[-1:], device=device, steer_only=args.steer_only)
    else:
        train_ds, test_ds = split_dataset(MarioDataset(files, device=device, steer_only=args.steer_only))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # loss_fn = WeightedLoss(torch.nn.MSELoss(), torch.tensor([1, 1, 1, 1, 5], device=device))
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

            if i % 10 == 0 and i > 0:
                print(f"epoch {e + 1}/{args.epochs}, batch {i} avg train loss: {total_loss / i}")

        test_loss = 0.
        for i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)

        print(
            f"epoch {e + 1} avg train loss: {total_loss / len(train_loader)}, avg test loss: {avg_test_loss}")
        print(f"========== epoch {e + 1}/{args.epochs} done ==========")

        scheduler.step()

        if args.model_path is not None and avg_test_loss < best_loss:
            print(f"saving model to {args.model_path}")
            best_loss = avg_test_loss
            model.save(args.model_path)


def run_model(args):
    course_opts = args_to_cours_opts(args)
    dirs = expand_dir_tree(course_opts)

    if len(dirs) > 1:
        print("too many course options. must provide only one course configuration (for now)")
        exit(1)

    for config in dirs:
        f = course_opt_to_save_file(config)

        dolphin = Dolphin(f, virtual_control=True)
        device = torch.device("cuda" if not args.force_cpu and torch.cuda.is_available() else "cpu")

        model = MarioNet(outputs=(1 if args.steer_only else 5)).to(device)
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
                if args.steer_only:
                    controls = [1, 0, 0, 0, controls.item()]
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


def dagger(args):
    print("starting dagger data collection mode")
    course_opts = args_to_cours_opts(args)
    dirs = expand_dir_tree(course_opts)

    print(f"{len(dirs)} were selected for data collection")

    dagger.course_done = False

    contr = XboxController()

    def input_thread():
        input("press enter when done")
        dagger.course_done = True

    for config in dirs:
        course = course_opt_to_save_file(config)

        print(f"starting with save file: {course}")
        dolphin = Dolphin(course, virtual_control=True)
        device = torch.device("cuda" if not args.force_cpu and torch.cuda.is_available() else "cpu")

        model = MarioNet(outputs=(1 if args.steer_only else 5)).to(device)
        model.load(args.model_path)

        vcont = VirtualController()
        video = torch.empty((0, 3, 456, 832))
        t = 0

        data = []
        dagger.course_done = False
        d = os.path.join(*config)
        os.makedirs(d, exist_ok=True)
        image_d = os.path.join(d, 'images')
        os.makedirs(image_d, exist_ok=True)
        print("when you're ready, press A to start the track")
        print("the model will drive, and you'll give expert inputs on your controller")
        print("when you're done, refocus on this terminal and press enter")
        try:
            while contr.A == 0:
                time.sleep(0.01)
            vcont.press_a()

            time.sleep(5)

            thread = threading.Thread(target=input_thread)
            thread.start()

            while not dagger.course_done:
                im = dolphin.screenshot()
                controls = contr.read()
                name = random_name(32)
                file = os.path.join(image_d, f'{name}.png')
                while os.path.exists(file):
                    name = random_name(32)
                    file = os.path.join(image_d, f'{name}.png')
                torchvision.io.write_png(im, file)
                controls.append(name)
                data.append(controls)
                im = torch.reshape(im, tuple([1]) + im.shape)
                im = im.to(dtype=torch.float32, device=device)
                model.eval()
                controls = model(im)
                controls = torch.flatten(controls)
                if args.steer_only:
                    controls = [1, 0, 0, 0, controls.item()]
                vcont.update(controls)
        finally:
            vcont.close()
            dolphin.close()
            if len(data) > 0:
                name = random_name(8)
                file = os.path.join(d, f'{name}.pkl')
                while os.path.exists(file):
                    name = random_name(8)
                    file = os.path.join(d, f'{name}.pkl')
                with open(file, 'wb') as f:
                    pickle.dump(data, f)
                print(f'wrote {len(data)} datapoints to {file}')


def course_opt_to_save_file(config):
    course = os.path.join(*config)
    if not os.path.exists(course):
        print(f'{course} course config does not exist')
    files = os.listdir(course)
    f = None
    for file in files:
        if file.endswith('.sav'):
            f = file
            break
    if f is None:
        print(f'save file not found for course config {course}')
        exit()
    f = os.path.join(course, f)
    return f


def args_to_cours_opts(args):
    return [args.drift_mode, args.course, args.player, args.vehicle, args.kart]


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
