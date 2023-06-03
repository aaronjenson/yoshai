import torch
from torch import nn


class WeightedLoss(nn.Module):
    def __init__(self, loss_fn, loss_weights: int | torch.Tensor = 1):
        """
        Weights the error by the loss_weights before calculating the loss
        using loss_fn. Should not be used with categorical loss functions.
        :param loss_fn: loss function to use after weighting error
        :param loss_weights: values to multiply outputs by. Can be scalar
                    or tensor with weights for each output.
        """
        super(WeightedLoss, self).__init__()
        self.loss = loss_fn
        self.weights = loss_weights

    def forward(self, output, target):
        # calc diff, scale by weights, add back to target, calc loss
        diff = output - target
        scaled = torch.mul(diff, self.weights)
        w_out = scaled + target
        loss = self.loss(w_out, target)
        return loss


if __name__ == '__main__':
    pred = torch.randn(3, 3)
    actual = torch.randn(3, 3)
    weights = torch.ones(3)
    weights[2] = 2
    loss_fn = nn.MSELoss()
    w_loss_fn = WeightedLoss(loss_fn, weights)
    loss = loss_fn(pred, actual)
    w_loss = w_loss_fn(pred, actual)

    print("predicted:")
    print(pred)
    print("actual:")
    print(actual)
    print("weights:")
    print(weights)
    print("unweighted loss:")
    print(loss)
    print("weighted loss:")
    print(w_loss)
