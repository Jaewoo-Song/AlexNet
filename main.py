import time
import torch
from torch import optim, Tensor
from torch.cuda.nccl import reduce
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchsummary import summary

from AlexNet import FashionMnistAlexNet
from train import train
from test import test

device = "cuda:1" if torch.cuda.is_available() else "else"
print(f"torch: {torch.__version__}")
print(f"device: {device}")

# Dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
train_dataset: Dataset = datasets.FashionMNIST(root="fashionMNIST", train=True, download=True, transform=transform)
test_dataset: Dataset = datasets.FashionMNIST(root="fashionMNIST", train=False, download=True, transform=transform)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# show samples
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# dataloader
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# model
model = FashionMnistAlexNet().to(device)
# summary(model, (1, 224, 224), batch_size)


# optimizer
optimizer = optim.Adam(model.parameters())
criterion = F.nll_loss

epochs = 10
for epoch in range(epochs):
    start = time.time()
    train(model, device, train_dataloader, criterion, optimizer, epoch)
    test(model, device, test_dataloader, criterion)
    elapsed = time.time() - start

    print("Elapsed Time: {:.2f} s".format(elapsed))
    print('=' * 50)

