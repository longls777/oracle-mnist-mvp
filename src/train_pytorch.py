import argparse
import torch, os, gzip
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import mnist_reader
from model import VisionTransformer
from modelv2 import BilinearModel

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2),
                                         torch.nn.ReLU())
        self.dense = torch.nn.Sequential(torch.nn.Linear(7 * 7 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.2),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return F.log_softmax(x, dim=1)


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.fc1 = nn.Linear(12 * 12 * 20, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 12 * 12 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ImageList(Dataset):

    def __init__(self, path, kind, channels, transform=None):
        (train_set, train_labels) = mnist_reader.load_data(path, kind)
        self.train_set = train_set
        self.train_labels = train_labels
        self.channels = channels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.channels == 3:
            img = img.repeat_interleave(3, dim=0)
        return img, target

    def __len__(self):
        return len(self.train_set)


def train(args, model, device, train_loader, optimizer, epoch, lossF):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossF(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, lossF):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += lossF(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=300, metavar='N', help='input batch size for testing (default: 300)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 15)')
    parser.add_argument('--net', type=str, default='Net1', choices=["Net1", "Net2", "Net3", "Net4", "Net5"], help='type of network')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--data-dir', type=str, default='../data/oracle/', help='data path')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument("--n_channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--embed_dim", type=int, default=64, help="dimensionality of the latent space")
    parser.add_argument("--n_attention_heads", type=int, default=4, help="number of heads to be used")
    parser.add_argument("--forward_mul", type=int, default=2, help="forward multiplier")
    parser.add_argument("--n_layers", type=int, default=6, help="number of encoder layers")
    parser.add_argument("--load_model", type=bool, default=False, help="Load saved model")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch Size")
    parser.add_argument("--img_size", type=int, default=28, help="Img size")
    parser.add_argument('--n_classes', type=int, default=10)

    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    if args.net == 'Net1':
        model = Net1().to(device)
        channel = 1
        loss = F.nll_loss
    elif args.net == 'Net2':
        model = Net2().to(device)
        channel = 1
        loss = F.nll_loss
    elif args.net == 'Net3':
        model = Net3().to(device)
        channel = 1
        loss = F.nll_loss
    elif args.net == 'Net4':
        model = VisionTransformer(args).to(device)
        channel = 1
        lossF = nn.CrossEntropyLoss()
    elif args.net == 'Net5':
        model = BilinearModel().to(device)
        channel = 3
        lossF = nn.CrossEntropyLoss()
    else:
        print("unknown net")

    if args.net == 'Net4':
        optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-3)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    tr_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomCrop(args.img_size, padding=2),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
    te_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize([args.img_size, args.img_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
    
    train_data = ImageList(path=args.data_dir, kind='train', channels = channel,
                           transform=tr_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    test_data = ImageList(path=args.data_dir, kind='t10k', channels = channel,
                          transform=te_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)


    cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, lossF)
        test(args, model, device, test_loader, lossF)
        cos_decay.step()

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_{}_{}.pt".format(args.net, test(args, model, device, test_loader, lossF)))

main()