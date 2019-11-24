import torch
from torch_inception_resnet_v2 import InceptionResNetV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained():
    model = load_model()


def load_model():
    """
    Loads the model to be used for training / inference
    :return: a torch.nn.Module
    """
    model = InceptionResNetV2(101)
    if torch.cuda.is_available():
        return model
    return model
