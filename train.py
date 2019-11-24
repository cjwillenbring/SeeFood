import sys
from datasets import loaders
from model import load_model
import torch
from torch import nn, optim

"""
All of the models from torchvision expect the input to be 224 x 224 and mean normalized.
"""

# this normalizes the inputs based on the stddev and mean for ImageNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ", device)


def train(network, c, op, loader, dataset):
    network.train()
    running_corrects = 0
    running_loss = 0
    for inputs, labels in loader:
        op.zero_grad()
        inputs = inputs.to(device=device, dtype=torch.half)
        labels = labels.to(device=device, dtype=torch.half)
        outputs = network(inputs)
        loss = c(outputs, labels)
        loss.backward()
        op.step()
        _, preds = torch.max(outputs, 1)
        running_loss += float(loss) * float(inputs.size(0))
        # this totally may have been it. If it were batch size it would have failed at any one point in time.
        running_corrects += torch.sum(preds == labels.data).item()
    return running_corrects / len(dataset), running_loss / len(dataset)


def evaluate(network, c, loader, dataset):
    network.eval()
    running_corrects = 0
    running_loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device=device, dtype=torch.half)
            labels = labels.to(device=device, dtype=torch.half)
            outputs = network(inputs)
            loss = c(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
    return running_corrects / len(dataset), running_loss / len(dataset)


def train_model(network, c, op, num_epochs=3):
    train_loader, val_loader, train_set, val_set = loaders(sys.argv[1])
    best_acc = 0.0
    train_accs = []
    val_accs = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        train_acc, train_loss = train(network, c, op, train_loader, train_set)
        train_accs += train_acc
        val_acc, val_loss = evaluate(network, c, train_loader, train_set)
        val_accs += val_acc
        if val_acc > best_acc:
            torch.save(model.state_dict(), 'model.pt')
            best_acc = val_acc
        print('Train: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
        print('Val: {:.4f}, acc: {:.4f}'.format(val_loss, val_acc))
    return network


if __name__ == '__main__':
    model = load_model().half()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.045)
    model = train_model(model, criterion, optimizer, 1000)
