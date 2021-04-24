import torch.nn as nn
from torchsummary import summary

from config import N_CLASS, IMAGE_DIM, DROPOUT_PROB


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.n_class = N_CLASS
    self.layers = nn.Sequential(
        # LeNet-5 Layers
        nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=8, out_channels=64, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=16384, out_features=2048),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=2048, out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512, out_features=N_CLASS)
        )

  def forward(self, x):
    return self.layers(x)


def printModel():
    # net = Net().to(device)
    net = Net()
    # visualizing the model
    print('Your network:')
    summary(net, (3, IMAGE_DIM, IMAGE_DIM))

    return net
