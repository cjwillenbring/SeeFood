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


def train_model(network, c, op, num_epochs=3):
    train_loader, val_loader, train_set, val_set = loaders(sys.argv[1])
    image_datasets = {'val': val_set, 'train': train_set}
    dataloaders = {'val': val_loader, 'train': train_loader}
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                network.train()
            else:
                network.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                op.zero_grad()

                outputs = network(inputs)
                loss = c(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    op.step()

                _, preds = torch.max(outputs, 1)
                running_loss += float(loss) * float(inputs.size(0))
                # this totally may have been it. If it were batch size it would have failed at any one point in time.
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects / len(image_datasets[phase])
            if epoch_acc > best_acc and phase == 'val':
                torch.save(model.state_dict(), 'model.pt')

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return network


if __name__ == '__main__':
    model = load_model().half()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.045)
    model = train_model(model, criterion, optimizer, 1000)
