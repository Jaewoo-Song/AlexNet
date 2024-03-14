import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, sampler
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from VGG import VGG16

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

batch_size = 64
learning_rate = 0.0002
num_epoch = 10
global train_loader
global val_loader
global test_loader

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def prepare_dataset():
    global train_loader
    global val_loader
    global test_loader

    # transform pipeline is applied by dataloader
    # and dataloader normalize data by default [0, 1]
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar10_train = datasets.CIFAR10(
        root="cifar10",
        train=True,
        transform=transform,
        download=True
    )
    cifar10_test = datasets.CIFAR10(
        root="cifar10",
        train=False,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(cifar10_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(
        range(int(len(cifar10_train) * 0.98))))
    val_loader = DataLoader(cifar10_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(
        range(int(len(cifar10_train) * 0.98)), len(cifar10_train)))
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))


def train(model):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_arr = []
    for epoch in range(num_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 30 == 0:
                print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        loss_arr.append(loss.cpu().detach().numpy())

    plt.plot(loss_arr)
    plt.show()


def run():
    prepare_dataset()
    model = VGG16().to(device)
    train(model)


if __name__ == '__main__':
    run()
