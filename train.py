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
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
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


# what the fuck is going on with the val set!!!!!!!!!!!!!


def train(model, criterion, optimizer, loader, size):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0
    print('train size:', size)
    for inputs, labels in map(convert_device, loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        l2, cross_entropy = criterion
        loss = cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        # number of examples in batch, but why>
        epoch_loss += loss.item() * size
        epoch_accuracy += num_correct(preds, labels)
    return epoch_loss / size, epoch_accuracy / size


def evaluate(model, criterion, loader, size):
    model.eval()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    print('eval size: ', size)
    with torch.no_grad():
        for inputs, labels in map(convert_device, loader):
            outputs = model(inputs)
            l2, cross_entropy = criterion
            loss = cross_entropy(outputs, labels)
            _, preds = torch.max(outputs, 1)
            epoch_loss += loss.item() * size
            epoch_accuracy += num_correct(preds, labels)
    return epoch_loss / size, epoch_accuracy / size


def run(model, criterion, optimizer, scheduler, loaders, sizes, n_epochs=50):
    best_accuracy = 0.0
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss, train_accuracy = train(model, criterion, optimizer, loaders['train'], sizes['train'])
        valid_loss, valid_accuracy = evaluate(model, criterion, loaders['val'], sizes['val'])
        scheduler.step()

        if valid_accuracy < best_accuracy:
            torch.save(model.load_state_dict(), 'model.pt')

        epoch_time = time.time() - start_time
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_time}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_accuracy * 100:.2f}%')

    # for every epoch we want to train the model and evaluate it
    # manual error analysis


def load_model():
    model_ft = torchvision.models.vgg16(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    num_features = model_ft.classifier[6].in_features
    # this kind of suggests that the models tried so far do not make good fixed feature extractors
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

    model_ft.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES),
        nn.LogSoftmax(dim=1))
    model_ft = model_ft.to(device)
    print('DEVICE: ', device)
    return model_ft


def main():
    model = load_model()
    loaders, sizes, classes = load_datasets()
    l2 = torch.nn.MSELoss()
    cross_entropy = nn.NLLLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.classifier.parameters(), lr=0.01, amsgrad=True)
    criterion = (l2, cross_entropy)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    run(model, criterion, optimizer_ft, exp_lr_scheduler, loaders, sizes, 25)


if __name__ == '__main__':
    main()
