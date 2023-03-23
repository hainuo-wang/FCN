import sys
import time

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model.GhostNet_FCN_model import ghostnet

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    # transforms.Normalize(0.5, 0.5)
])

train_dataset = datasets.CIFAR10(root='CIFAR10', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.CIFAR10(root='CIFAR10', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
model = ghostnet(input_channel=3, num_classes=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ExponentialLR(optimizer, gamma=0.9)


def train(epoch, train_loader):
    running_loss = 0.0
    correct = 0
    total = 0
    times = 0
    train_loader = tqdm(train_loader, desc="train", file=sys.stdout, colour="Green")
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        times += 1
    print('epoch:%2d  loss:%.3f  train_acc:%.2f' % (epoch + 1, running_loss / times, 100 * correct / total))
    return running_loss / times, 100 * correct / total


def test(test_loader):
    correct = 0
    total = 0
    test_loader = tqdm(test_loader, desc="test ", file=sys.stdout, colour="red")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))
    return 100 * correct / total


def record(filename, epoch, train_accuracy, val_accuracy, loss, lr):
    filename = filename
    data = str(epoch) + '  ' + str(train_accuracy) + '  ' + str(val_accuracy) + '  ' + str(loss) + '  ' + str(lr)
    with open(filename, 'a') as f:
        f.write(data)
        f.write('\n')


def record_time(filename, runningtime):
    with open(filename, 'a') as f:
        f.write(runningtime)
        f.write('\n')


if __name__ == '__main__':
    start = time.perf_counter()
    filename = 'GhostNet_FCN.txt'
    title = 'epoch' + '  ' + 'accuracy_train' + '  ' + 'accuracy_val' + '  ' + 'loss' + '  ' + 'learnning_rate'
    with open(filename, 'a') as f:
        f.write(title)
        f.write('\n')
    total_accuracy = []
    for epoch in range(100):
        loss, train_accuracy = train(epoch, train_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        val_accuracy = test(test_loader)
        total_accuracy.append(val_accuracy)
        record(filename, epoch, train_accuracy, val_accuracy, loss, lr)
    end = time.perf_counter()
    running_time = 'runningtime:' + '  ' + str((end - start) // 60) + 'min' + '  ' + str((end - start) % 60) + 's'
    record_time(filename, running_time)

