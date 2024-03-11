import torch


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # 64x10
            test_loss += criterion(output, target, reduction="sum").item()  # batch_size 만큼 더해서 loss에 합치기
            pred_value, pred_label = output.max(dim=1, keepdim=False)
            correct += pred_label.eq(target).sum().item()

        test_loss /= len(test_loader.dataset)
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

