import torch
from torchvision.models import resnet101

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained():
    model = load_model()


def unfreeze_last_n(n):
    def unfreeze(model):
        for i, param in enumerate(reversed(list(model.parameters()))):
            if i < n:
                param.requires_grad = True
    return unfreeze


def load_model():
    """
    Loads the model to be used for training / inference
    :return: a torch.nn.Module
    """
    model = resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    unfreeze = unfreeze_last_n(10)
    unfreeze(model)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2048, 101)
    )
    if torch.cuda.is_available():
        return model.cuda()
    return model
