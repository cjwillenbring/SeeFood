import torch
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained():
    model = load_model()


def load_model():
    """
    Loads the model to be used for training / inference
    :return: a torch.nn.Module
    """
    model = resnet18(num_classes=101)
    if torch.cuda.is_available():
        return model.cuda()
    return model
