from torchvision import transforms, datasets, models
import sys
import os
import torch
from torch import nn, optim


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'val':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'train':
    datasets.ImageFolder(os.path.join(sys.argv[1], 'train'), data_transforms['train']),
    'val':
    datasets.ImageFolder(os.path.join(sys.argv[1], 'val'), data_transforms['val'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=8),  # for Kaggle
    'val':
    torch.utils.data.DataLoader(image_datasets['val'],
                                batch_size=32,
                                shuffle=False,
                                num_workers=8)  # for Kaggle
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ", device)

model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(128, 101)
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), weight_decay=1e-5)


def train_model(network, c, op, num_epochs=3):
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

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return network


model = train_model(model, criterion, optimizer, 50)

torch.save(model.load_state_dict(), 'model.pt')
