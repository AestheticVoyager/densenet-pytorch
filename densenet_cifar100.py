import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
import argparse
import time
import os
import sys

# DenseNet Architecture Components
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=0.2):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout_rate)
    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, dropout_rate))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        x = self.conv(torch.relu(self.bn(x)))
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5, 
                 num_classes=100, dropout_rate=0.2, init_channels=64):
        super(DenseNet, self).__init__()
        # Initial convolution
        in_channels = init_channels
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, padding=1, bias=False)
        # Dense blocks
        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = DenseBlock(in_channels, num_layers, growth_rate, dropout_rate)
            self.dense_blocks.append(block)
            in_channels += num_layers * growth_rate
            # Transition layer (except after the last block)
            if i != len(block_config) - 1:
                out_channels = int(in_channels * compression)
                trans = TransitionLayer(in_channels, out_channels)
                self.trans_layers.append(trans)
                in_channels = out_channels
        # Final batch norm
        self.bn = nn.BatchNorm2d(in_channels)
        # Linear layer
        self.fc = nn.Linear(in_channels, num_classes)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        for i in range(len(self.dense_blocks)):
            x = self.dense_blocks[i](x)
            if i < len(self.trans_layers):
                x = self.trans_layers[i](x)
        x = torch.relu(self.bn(x))
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def densenet121(growth_rate=32, num_classes=100, dropout_rate=0.2):
    return DenseNet(growth_rate=growth_rate, block_config=(6, 12, 24, 16), 
                   num_classes=num_classes, dropout_rate=dropout_rate, init_channels=64)


def densenet169(growth_rate=32, num_classes=100, dropout_rate=0.2):
    return DenseNet(growth_rate=growth_rate, block_config=(6, 12, 32, 32), 
                   num_classes=num_classes, dropout_rate=dropout_rate, init_channels=64)


def densenet201(growth_rate=32, num_classes=100, dropout_rate=0.2):
    return DenseNet(growth_rate=growth_rate, block_config=(6, 12, 48, 32), 
                   num_classes=num_classes, dropout_rate=dropout_rate, init_channels=64)


def densenet264(growth_rate=32, num_classes=100, dropout_rate=0.2):
    return DenseNet(growth_rate=growth_rate, block_config=(6, 12, 64, 48), 
                   num_classes=num_classes, dropout_rate=dropout_rate, init_channels=64)


def densenet_cifar(growth_rate=12, num_classes=100, dropout_rate=0.2):
    return DenseNet(growth_rate=growth_rate, block_config=(16, 16, 16), 
                   num_classes=num_classes, dropout_rate=dropout_rate, init_channels=32)


def train(model, trainloader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(trainloader)
    return avg_loss, accuracy


def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='DenseNet CIFAR-100 Implementation')
    parser.add_argument('--model', type=str, default='cifar', choices=['cifar', '121', '169', '201', '264'],
                        help='DenseNet variant (default: cifar)')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Growth rate for DenseNet (default: 12)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1)')
    parser.add_argument('--scheduler', type=str, default='multistep', 
                        choices=['multistep', 'cosine', 'plateau'],
                        help='Learning rate scheduler (default: multistep)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer (default: sgd)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD (default: 0.9)')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("GPU is NOT available.")
        sys.exit(1)
    print(f"Using device: {device}")
    # Data preprocessing and augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    num_classes = 100
    
    model_dict = {
        'cifar': densenet_cifar,
        '121': densenet121,
        '169': densenet169,
        '201': densenet201,
        '264': densenet264
    }
    
    model = model_dict[args.model](
        growth_rate=args.growth_rate, 
        num_classes=num_classes, 
        dropout_rate=args.dropout
    )
    model = model.to(device)
    print(f"Using DenseNet-{args.model} with growth rate {args.growth_rate} and dropout {args.dropout}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    else:  # adam
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    # Learning rate scheduler
    if args.scheduler == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:  # plateau
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, threshold=0.002, verbose=True)
    # Training loop
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test(model, testloader, criterion, device)
        # Update learning rate based on scheduler
        if args.scheduler == 'plateau':
            scheduler.step(test_acc)
        else:
            scheduler.step()
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'densenet{args.model}_cifar100_best.pth')
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch: {epoch:3d} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}% | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    training_time = time.time() - start_time
    print(f'Training completed in {training_time/60:.2f} minutes')
    print(f'Best test accuracy: {best_acc:.2f}%')
    # Load best model and evaluate
    model.load_state_dict(torch.load(f'densenet{args.model}_cifar100_best.pth'))
    final_test_loss, final_test_acc = test(model, testloader, criterion, device)
    print(f'Final test accuracy: {final_test_acc:.2f}%')
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'training_curves_densenet{args.model}.png')
    plt.show()
    # Save results to file
    with open(f'results_densenet{args.model}.txt', 'w') as f:
        f.write(f'Model: DenseNet-{args.model}\n')
        f.write(f'Growth rate: {args.growth_rate}\n')
        f.write(f'Dropout rate: {args.dropout}\n')
        f.write(f'Optimizer: {args.optimizer}\n')
        f.write(f'Learning rate scheduler: {args.scheduler}\n')
        f.write(f'Training time: {training_time/60:.2f} minutes\n')
        f.write(f'Best test accuracy: {best_acc:.2f}%\n')
        f.write(f'Final test accuracy: {final_test_acc:.2f}%\n')
        f.write(f'Number of parameters: {sum(p.numel() for p in model.parameters()):,}\n')


if __name__ == "__main__":
    main()
