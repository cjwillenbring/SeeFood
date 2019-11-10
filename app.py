from train import eval_transform
import os
from torch import nn
import torch
import random
from torchvision import transforms, models
from flask import Flask, request
from PIL import Image
from io import BytesIO
import base64
from output import predict_one
from config import get_class_name, random_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(128, 101)
).to(device)

model.load_state_dict('model.pt')

app = Flask(__name__)


def image_from_base64(b64):
    data = {'img': b64}
    return Image.open(BytesIO(base64.b64decode(data)))


def mock_response():
    return {'food': random_class(), 'confidence': random.random()}


def predict_base64(b64):
    img = image_from_base64(b64)
    x = eval_transform(img)
    output = model(x)
    score, k = predict_one(output)
    return {'confidence': score, 'food': get_class_name(k)}


@app.route('/predictions', methods=['POST'])
def hello_world():
    if type(request.get_json()['base64']) != str:
        return {'code': 403, 'message': 'Invalid request body. Request body must be a base64 encoded image.'}
    return predict_base64(request.get_json(['base64']))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
