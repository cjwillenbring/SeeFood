from torchvision import transforms, datasets, models
import sys
import os
import torch
from torch import nn, optim


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


eval_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])


data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'val': eval_transform
}

image_datasets = {
    'train':
    datasets.ImageFolder(os.path.join(sys.argv[1], 'train'), data_transforms['train']),
    'val':
    datasets.ImageFolder(os.path.join(sys.argv[1], 'val'), data_transforms['val'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=512,
                                shuffle=True,
                                num_workers=8, pin_memory=True),  # for Kaggle
    'val': torch.utils.data.DataLoader(image_datasets['val'],
                                batch_size=512,
                                shuffle=False,
                                num_workers=8, pin_memory=True)  # for Kaggle
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ", device)

model = models.resnet18(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

# train on more data. out. regularization? sure.

model.fc = nn.Sequential(
    nn.Dropout(0.8),
    nn.Linear(512, 512),
    nn.BatchNorm1d(512),
    nn.LeakyReLU(inplace=True),
    nn.Dropout(0.75),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(inplace=True),
    nn.Dropout(0.7),
    nn.Linear(256, 128),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(inplace=True),
    nn.Linear(128, 101)
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), weight_decay=1e-3, lr=0.001)


def train_model(network, c, op, num_epochs=3):
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

                outputs = network(inputs)
                loss = c(outputs, labels)

                if phase == 'train':
                    op.zero_grad()
                    loss.backward()
                    op.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            if epoch_acc > best_acc and phase == 'val':
                torch.save(model.state_dict(), 'model.pt')

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return network


model = train_model(model, criterion, optimizer, 500)

torch.save(model.state_dict(), 'model.pt')
