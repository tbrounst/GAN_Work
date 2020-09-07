from IPython import display

import random
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from ganart.utils import Logger

device = torch.device('cpu')
model_to_use = 'C:\\Users\\Tom\\PycharmProjects\\pythonProject\\ganart\\models\\G_epoch_400'

def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n.to(device)

def filled(size, value):
    n = Variable(torch.randn(size, 100))
    n = n.new_full(n.size(),value)
    if torch.cuda.is_available(): return n.cuda()
    return n.to(device)


class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

        self.linear = torch.nn.Linear(100, 1024 * 7 * 10)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=(6, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=(8, 4),
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 7, 10)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)


generator = GeneratorNet()
generator.to(device)

G = GeneratorNet()
G.load_state_dict(torch.load(model_to_use, map_location=torch.device('cpu')))
G.to(device)
G.eval()

num_test_samples = 49
num_tests = 20
for i in range(num_tests):
    fake_data = G(noise(num_test_samples)).detach()
    logger = Logger(model_name='loaded-GANART', data_name='Lands')
    logger.log_images(fake_data, num_test_samples, 400, i, 0);
    logger.close()