#!/usr/bin/env python
# coding: utf-8

# # GANART: an AI that generates MtG Art

# In[2]:


from IPython import display

import copy
import random
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from ganart.utils import Logger

# ### DATA and MISC

# In[3]:


# device = torch.device('cuda:0')
device = torch.device('cpu')


# In[4]:


def lands():
    # data_path = 'C:\\Users\\Tom\\Desktop\\TestFolder'
    data_path = 'C:\\Users\\Tom\\Desktop\\TrainGAN\\'
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5), (.5))
         ])
    train_dataset = datasets.ImageFolder(root=data_path, transform=compose)
    # return train_dataset
    return datasets.MNIST(root='./dataset', train=True, transform=compose, download=True)


# Load data
data = lands()
batch_size = 3000
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
num_batches = len(data_loader)
print("num batches: " + str(num_batches))


# In[5]:


def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n


# In[6]:


def ones_target(size):
    data = Variable(torch.ones(size, 1))
    data = data.new_full(data.size(), random.uniform(0.9, 1))
    return data.to(device)


def zeros_target(size):
    data = Variable(torch.zeros(size, 1))
    return data.to(device)


# ### DISCRIMINATOR

# In[7]:


kernel_s = (4, 4)
stride_s= (2,2)


class DiscriminatorNet(torch.nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=kernel_s,
                stride=stride_s, padding=(1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=kernel_s,
                stride=stride_s, padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=kernel_s,
                stride=stride_s, padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=kernel_s,
                stride=stride_s, padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, 1),
            #nn.Linear(49152, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Convolutional layers
        # print("Dis: " + str(x.shape))
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024)
        #x = x.view(-1, 49152)
        x = self.out(x)
        return x


discriminator = DiscriminatorNet()
# discriminator.load_state_dict(torch.load('C:\\Users\\Tom\\PycharmProjects\\pythonProject\\data\\models\\DConv-GConv-GANART\\Lands\\D_epoch_300'))
discriminator.to(device)


# ### GENERATOR

# In[8]:


class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

        self.linear = torch.nn.Linear(100, 1024 * 1 * 1)

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
                in_channels=256, out_channels=128, kernel_size=8,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=1, kernel_size=8,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 1, 1)
        # Convolutional layers
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        # Apply Tanh
        return self.out(x)


generator = GeneratorNet()
# generator.load_state_dict(torch.load('C:\\Users\\Tom\\PycharmProjects\\pythonProject\\data\\models\\DConv-GConv-GANART\\Lands\\G_epoch_300'))
generator.to(device)


# In[9]:


d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss = nn.BCELoss()


# In[10]:


def train_discriminator(optimizer, discriminator2, real_data, fake_data):
    N = real_data.size(0)
    real_data.to(device)
    fake_data.to(device)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    # prediction_real = discriminator(real_data.cuda())
    prediction_real = discriminator2(real_data)
    # print("Generated real prediction")
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()
    # print("Trained on real")

    # 1.2 Train on Fake Data
    prediction_fake = discriminator2(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    # print("Trained on fake")

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


# In[11]:


def train_generator(optimizer, discriminator, fake_data):
    N = fake_data.size(0)
    fake_data.to(device)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


# In[12]:


num_test_samples = 4
test_noise = noise(num_test_samples)
num_epochs = 400
num_rolls = 5

# In[13]:


logger = Logger(model_name='DConv-GConv-GANART', data_name='Lands')

for epoch in range(num_epochs):
    print(epoch)
    for n_batch, (real_batch, _) in enumerate(data_loader):
        print("Batch: " + str(n_batch))

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        fake_data = generator(noise(real_data.size(0))).detach()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, discriminator, real_data, fake_data)
        print("Finished training discriminator")

        # 2. Train Generator
        # Unroll D
        discriminator_copy = copy.deepcopy(discriminator)
        data_loader_copy = copy.deepcopy(data_loader)
        iterator = iter(data_loader_copy)
        for k in range(num_rolls - 1):
            print("Unroll thing: " + str(k))
            if (n_batch + k < num_batches):
                real_data = Variable(next(iterator)[0])
                fake_data = generator(noise(real_data.size(0))).detach()
                train_discriminator(d_optimizer, discriminator_copy, real_data, fake_data)
        # Train G
        fake_data = generator(noise(real_batch.size(0)))
        g_error = train_generator(g_optimizer, discriminator_copy, fake_data)
        print("Finished generator")

        # 3. Log
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display Progress
        if (n_batch) % 50 == 0:
            display.clear_output(True)
            # Display Images
            test_images = generator(test_noise).data.cpu()
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
        if (epoch) % 50 == 0:
            # Model Checkpoints
            logger.save_models(generator, discriminator, epoch)
logger.save_models(generator, discriminator, epoch)
logger.close()