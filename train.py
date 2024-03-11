import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn import functional


def train(model, device, train_loader, criterion, optimizer: Optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # set gradient zero before backpropagation, all params are registered when this optimizer is instantiated.
        optimizer.zero_grad()

        output = model(data)  # 64x10
        loss: Tensor = criterion(output, target)
        loss.backward()  # calculate gradient

        optimizer.step()  # update weight based on the calculated gradient

        if (batch_idx + 1) % 30 == 0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
