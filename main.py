import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

# for model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.cifar import CIFAR10


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    # parser.add_argument("--gpu_ids", nargs="+", default=["0", "1", "2", "3"])
    # parser.add_argument("--world_size", type=int, default=0)
    parser.add_argument(
        "--train_path", type=str, default="~/datasets/imagenet/imagenet100/train"
    )
    parser.add_argument(
        "--val_path", type=str, default="~/datasets/imagenet/imagenet100/val"
    )
    parser.add_argument("--num_classes", type=int, default=100)

    return parser


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_1x1,
        num_3x3_red,
        num_3x3,
        num_5x5_red,
        num_5x5,
        num_pool_proj,
    ):
        super(InceptionBlock, self).__init__()

        self.b_1x1 = ConvBlock(in_channels, num_1x1, kernel_size=1)

        self.b_3x3_red = ConvBlock(in_channels, num_3x3_red, kernel_size=1)
        self.b_3x3 = ConvBlock(num_3x3_red, num_3x3, kernel_size=3, padding=1)

        self.b_5x5_red = ConvBlock(in_channels, num_5x5_red, kernel_size=1)
        self.b_5x5 = ConvBlock(num_5x5_red, num_5x5, kernel_size=5, padding=2)

        self.b_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b_pool_proj = ConvBlock(in_channels, num_pool_proj, kernel_size=1)

    def forward(self, x):
        x1 = self.b_1x1(x)
        x2 = self.b_3x3(self.b_3x3_red(x))
        x3 = self.b_5x5(self.b_5x5_red(x))
        x4 = self.b_pool_proj(self.b_maxpool(x))

        return torch.cat([x1, x2, x3, x4], 1)


class Auxiliary(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Auxiliary, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Inception(nn.Module):
    def __init__(self, in_channels=3, use_auxiliary=True, num_classes=10):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)

        self.use_auxiliary = use_auxiliary
        if use_auxiliary:
            self.auxiliary4a = Auxiliary(512, num_classes)
            self.auxiliary4d = Auxiliary(528, num_classes)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

    def forward(self, x):
        y = None
        z = None

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)

        x = self.inception4a(x)
        if self.training and self.use_auxiliary:
            y = self.auxiliary4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.use_auxiliary:
            z = self.auxiliary4d(x)

        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)

        x = self.linear(x)

        return x, y, z


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    lr_scheduler,
    num_epochs,
    use_auxiliary=True,
):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:  # Each epoch has a training and validation phase
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(
                dataloaders[phase]
            ):  # Iterate over data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # Zero the parameter gradients

                with torch.set_grad_enabled(
                    phase == "train"
                ):  # Forward. Track history if only in train

                    if (
                        phase == "train"
                    ):  # Backward + optimize only if in training phase
                        if use_auxiliary:
                            outputs, aux1, aux2 = model(inputs)
                            loss = (
                                criterion(outputs, labels)
                                + 0.3 * criterion(aux1, labels)
                                + 0.3 * criterion(aux2, labels)
                            )
                        else:
                            outputs, _, _ = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        loss.backward()
                        optimizer.step()

                    if phase == "val":
                        outputs, _, _ = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == "train" and i % 100 == 0:
                    print(
                        "Epoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}".format(
                            epoch,
                            args.epoch,
                            i,
                            len(dataloaders[phase]),
                            loss.item(),
                        )
                    )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == "val":  # Adjust learning rate based on val loss
                lr_scheduler.step(epoch_loss)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def main_worker(args):
    # transform_train = transforms.Compose(
    #     [
    #         transforms.Resize(256),
    #         transforms.RandomCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
    #         ),
    #     ]
    # )

    # transform_test = transforms.Compose(
    #     [
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
    #         ),
    #     ]
    # )

    # train_set = CIFAR10(
    #     root="cifar10", train=True, transform=transform_train, download=True
    # )

    # test_set = CIFAR10(
    #     root="cifar10", train=False, transform=transform_test, download=True
    # )

    # train_loader = DataLoader(
    #     dataset=train_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )

    # val_loader = DataLoader(
    #     dataset=test_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )

    # dataset
    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(args.train_path, transform=transform_train)
    val_dataset = datasets.ImageFolder(args.val_path, transform=transform_val)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Inception(num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, verbose=True
    )

    model, val_acc_history = train_model(
        model,
        {"train": train_loader, "val": val_loader},
        criterion,
        optimizer,
        device,
        lr_scheduler,
        args.epoch,
    )

    print(val_acc_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "GoogLeNet ImageNet training", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    main_worker(args)
