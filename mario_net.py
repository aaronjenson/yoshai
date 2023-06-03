import collections
import time

import torch
import torchvision
from torch import nn
from mario_dataset import MarioDataset


class MarioNet(nn.Module):
    def __init__(self, outputs=5):
        super().__init__()
        resnet = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights='DEFAULT')
        layers = list(resnet.children())[:-1]
        self.flatten = nn.Flatten()
        self.end_layers = nn.Sequential(
            nn.Linear(237120, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, outputs)
        )
        self.network = nn.Sequential(*layers)

    def freeze(self, layers=0):
        """
        disables optimization in some or all layers from the pre-trained network
        freezes up to 'layers'. passing 0 freezes all layers, and -1 freezes all up to the last
        :param layers: number of layers to freeze (0 for all, negative numbers wrap to end)
        :return:
        """
        if layers <= 0:
            layers += len(self.network)
        for i, child in enumerate(self.network.children()):
            if i >= layers:
                break
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.network(x)
        if isinstance(x, collections.OrderedDict):
            # x = torch.cat([self.flatten(t) for t in x.values()], dim=1)
            x = self.flatten(x['low'])
        else:
            x = self.flatten(x)
        return self.end_layers(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    dataset = MarioDataset(['bcdata/luigi-circuit_1.pkl'])
    img, contr = dataset[0]
    net = MarioNet()
    t = time.time()
    out = net(img.reshape((1, 3, 832, 456)))
    dt = time.time() - t
    print(f"compute time: {dt}s")
    print(f"predicted: {out}")
    print(f"actual:    {contr}")
