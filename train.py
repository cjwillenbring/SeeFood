from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import time
import sys
import os

NUM_CLASSES = 101

DROPOUT = [0.5, 0.5]

LINEAR_SIZE = 256

torch.random.seed = 1234

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_datasets():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    }

    data_dir = sys.argv[1]
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=128,
                                                  shuffle=True,
                                                  num_workers=16,
                                                  pin_memory=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names


def num_correct(predictions, labels):
    return (predictions == labels.data).sum().float()


def calculate_loss(loss, m):
    return loss.item() * m


def convert_device(batch):
    inputs, labels = batch
    return inputs.to(device), labels.to(device)


def train(model, criterion, optimizer, scheduler, loader, size):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0
    print('train size:', size)
    for inputs, labels in map(convert_device, loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        _, preds = torch.max(outputs, 1)
        # number of examples in batch, but why>
        epoch_loss += loss.item() * size
        epoch_accuracy += num_correct(preds, labels)
        optimizer.step()
    scheduler.step()
    return epoch_loss / size, epoch_accuracy / size


def evaluate(model, criterion, loader, size):
    model.eval()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    print('eval size: ', size)
    with torch.no_grad():
        for inputs, labels in map(convert_device, loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            epoch_loss += loss.item() * size
            epoch_accuracy += num_correct(preds, labels)
    return epoch_loss / size, epoch_accuracy / size


def run(model, criterion, optimizer, scheduler, loaders, sizes, n_epochs=50):
    best_accuracy = 0.0
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss, train_accuracy = train(model, criterion, optimizer, scheduler, loaders['train'], sizes['train'])
        valid_loss, valid_accuracy = evaluate(model, criterion, loaders['val'], sizes['val'])

        if valid_accuracy < best_accuracy:
            torch.save(model.load_state_dict(), 'model.pt')

        epoch_time = time.time() - start_time
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_time}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_accuracy * 100:.2f}%')

    # for every epoch we want to train the model and evaluate it
    # manual error analysis


def load_model():
    model_ft = torchvision.models.resnet50(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    num_features = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

    model_ft.fc = torch.nn.Sequential(
        torch.nn.Dropout(DROPOUT[0]),
        torch.nn.Linear(num_features, NUM_CLASSES)
    )
    model_ft = model_ft.to(device)
    print('DEVICE: ', device)
    return model_ft


def main():
    model = load_model()
    loaders, sizes, classes = load_datasets()
    print(classes)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=0.008)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    run(model, criterion, optimizer_ft, exp_lr_scheduler, loaders, sizes, 25)


if __name__ == '__main__':
    main()
