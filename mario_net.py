import torch
from torch import nn
from mario_dataset import MarioDataset

class MarioNet(nn.Module):
    def __init__(self, input_shape: tuple[int, int], outputs: int):
        super().__init__()
        x_in, y_in = input_shape
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(11648, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, outputs)
        )

    def forward(self, x):
        return self.network(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    dataset = MarioDataset(['bcdata/luigi_circuit_0.pkl'])
    img, contr = dataset[0]
    net = MarioNet((832, 456), 4)
    out = net(img.reshape((1, 3, 832, 456)))
    print(f"predicted: {out}")
    print(f"actual:    {contr}")