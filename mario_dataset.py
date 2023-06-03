import os

import torch
from torch.utils.data import Dataset
import pickle
import torchvision
import numpy as np


CONTROLS_LENGTH = 5


class MarioDataset(Dataset):
    def __init__(self, filenames, device='cpu', steer_only=False):
        self.device = device
        self.steer_only = steer_only
        data = []
        self.images = np.empty(0, dtype=str)
        self.controls = np.empty((0, 1 if steer_only else CONTROLS_LENGTH), dtype=float)

        if filenames is None:
            return
        for file in filenames:
            if ".pkl" in file:
                if not os.path.exists(file):
                    print(f'{file} does not exist')
                img_path = os.path.join(os.path.split(file)[0], 'images')
                f = open(file, 'rb')
                d = pickle.load(f)
                for x in d:
                    x[CONTROLS_LENGTH] = f'{os.path.join(img_path, x[CONTROLS_LENGTH])}.png'
                data.extend(d)
                f.close()

        for d in data:
            self.images = np.append(self.images, [d[CONTROLS_LENGTH]], axis=0)
            self.controls = np.append(self.controls, [d[CONTROLS_LENGTH - 1:CONTROLS_LENGTH] if steer_only else d[0:CONTROLS_LENGTH]], axis=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        im = torchvision.io.read_image(img_path).to(self.device, dtype=torch.float32)
        return im, torch.from_numpy(self.controls[idx]).to(self.device, dtype=torch.float32)


def split_dataset(data: MarioDataset, factor=0.2):
    """
    Splits a single dataset into a train and test dataset. Given dataset is modified and becomes the train dataset.
    :param data: dataset to split
    :param factor: percentage of elements to move into test dataset
    :return: train dataset, test dataset
    """
    train = data
    test = MarioDataset(None, train.device, steer_only=train.steer_only)
    test_size = int(len(train) * factor)
    test_indices = np.arange(len(train))
    test_indices = np.random.choice(test_indices, size=test_size, replace=False)

    test.images = np.array(train.images[test_indices])
    test.controls = np.array(train.controls[test_indices])

    train.images = np.delete(train.images, test_indices, axis=0)
    train.controls = np.delete(train.controls, test_indices, axis=0)

    return train, test


if __name__ == "__main__":
    dataset = MarioDataset(
        ["bcdata/luigi-circuit_1.pkl"], steer_only=True)
    train, test = split_dataset(dataset)
    print(f"train: {len(train)}, test: {len(test)}")
