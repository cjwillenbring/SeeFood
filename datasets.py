from hyperparameters import BATCH_SIZE
from torch.utils import data
from torchvision import transforms, datasets
import os

mean_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# need to resize the image to get the right dims, then transform to tensor and normalize
evaluation_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    mean_normalize
])

train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    mean_normalize
])


def load_data_sets(root):
    """
    Load the train, dev and test data sets
    :param root: the root directory of the data sets.
    We expect data sets to be separated into distinct folders and each class's examples to be in distinct folders.
    :return: the dataset objects for the train, dev and test sets
    """
    train_path = os.path.join(root, 'train')
    train_set = datasets.ImageFolder(train_path, train_transform)
    val_path = os.path.join(root, 'val')
    val_set = datasets.ImageFolder(val_path, evaluation_transform)
    # test_path = os.path.join(root, 'test')
    # test_set = datasets.ImageFolder(test_path, evaluation_transform)
    return train_set, val_set


def loaders(root):
    """
    Build the data loaders for the train, dev and test sets
    :param root: the directory containing the data sets
    :return: the loaders for each data set
    """
    train_set, val_set = load_data_sets(root)
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    return train_loader, val_loader, train_set, val_set
