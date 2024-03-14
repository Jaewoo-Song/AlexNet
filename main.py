import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as tfs

# for model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# for dataset
from torchvision.datasets.cifar import CIFAR10

from VGG import VGG16


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--vis_step", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2'])
    parser.add_argument("--gpu_ids", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--world_size", type=int, default=0)
    parser.add_argument("--port", type=int, default=2022)
    parser.add_argument("--root", type=str, default="./cifar10")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="./save")
    parser.add_argument("--save_file_name", type=str, default="vgg_cifar")

    return parser


def main_worker(rank, args):

    # 1. argparse (main)
    # 2. init dist
    local_gpu_id = init_for_distributed(rank, args)

    # 4. data set
    transform_train = tfs.Compose(
        [
            tfs.Resize(256),
            tfs.RandomCrop(224),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = tfs.Compose(
        [
            tfs.Resize(256),
            tfs.CenterCrop(224),
            tfs.ToTensor(),
            tfs.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = CIFAR10(
        root=args.root, train=True, transform=transform_train, download=True
    )

    test_set = CIFAR10(
        root=args.root, train=False, transform=transform_test, download=True
    )

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=int(args.batch_size / args.world_size),
        shuffle=False,
        num_workers=int(args.num_workers / args.world_size),
        sampler=train_sampler,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=int(args.batch_size / args.world_size),
        shuffle=False,
        num_workers=int(args.num_workers / args.world_size),
        sampler=test_sampler,
        pin_memory=True,
    )

    # 5. model
    model = VGG16()
    model = model.cuda(local_gpu_id)

    model = DDP(module=model, device_ids=[local_gpu_id])

    # 6. criterion
    criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(local_gpu_id)

    # 7. optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.start_epoch != 0:
        checkpoint = torch.load(
            os.path.join(args.save_path, args.save_file_name)
            + ".{}.pth.tar".format(args.start_epoch - 1),
            map_location=torch.device("cuda:{}".format(local_gpu_id)),
        )
        # load model state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        # load optim state dict
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.rank == 0:
            print("\nLoaded checkpoint from epoch %d.\n" % (int(args.start_epoch) - 1))

    for epoch in range(args.start_epoch, args.epoch):

        # 9. train
        tic = time.time()
        model.train()
        train_sampler.set_epoch(epoch)

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(local_gpu_id)
            labels = labels.to(local_gpu_id)
            outputs = model(images)

            # ----------- update -----------
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # time
            toc = time.time()

            if (
                i % args.vis_step == 0 or i == len(train_loader) - 1
            ) and args.rank == 0:
                print(
                    "Epoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}, Time: {5:.2f}".format(
                        epoch, args.epoch, i, len(train_loader), loss.item(), toc - tic
                    )
                )

        # save pth file
        if args.rank == 0:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(
                checkpoint,
                os.path.join(
                    args.save_path, args.save_file_name + ".{}.pth.tar".format(epoch)
                ),
            )
            print("save pth.tar {} epoch!".format(epoch))

        # 10. test
        if args.rank == 0:
            model.eval()

            val_avg_loss = 0
            correct_top1 = 0
            correct_top5 = 0
            total = 0

            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    images = images.to(args.rank)
                    labels = labels.to(args.rank)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_avg_loss += loss.item()
                    # ------------------------------------------------------------------------------
                    # rank 1
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct_top1 += (pred == labels).sum().item()

                    # ------------------------------------------------------------------------------
                    # rank 5
                    _, rank5 = outputs.topk(5, 1, True, True)
                    rank5 = rank5.t()
                    correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

                    # ------------------------------------------------------------------------------
                    for k in range(5):  # 0, 1, 2, 3, 4, 5
                        correct_k = (
                            correct5[: k + 1].reshape(-1).float().sum(0, keepdim=True)
                        )
                    correct_top5 += correct_k.item()

            accuracy_top1 = correct_top1 / total
            accuracy_top5 = correct_top5 / total

            val_avg_loss = val_avg_loss / len(test_loader)  # make mean loss

            print("top-1 percentage :  {0:0.3f}%".format(correct_top1 / total * 100))
            print("top-5 percentage :  {0:0.3f}%".format(correct_top5 / total * 100))

    return 0


def init_for_distributed(rank, args):

    # 1. setting for distributed training
    args.rank = rank
    local_gpu_id = int(args.gpu_ids[args.rank])
    torch.cuda.set_device(local_gpu_id)
    if args.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23456",
        world_size=args.world_size,
        rank=args.rank,
    )

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(args.rank == 0)

    print(args)
    return local_gpu_id


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "vgg16 cifar10 training", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    args.world_size = len(args.gpu_ids)
    args.num_workers = len(args.gpu_ids) * args.num_workers

    mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)
