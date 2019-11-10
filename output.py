import torch


def build_predictions(model_output):
    return torch.max(model_output)


def predict_one(model_output):
    values, indices = build_predictions(model_output)
    return values.item(), indices.item()