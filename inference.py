from train import load_model
import os
from torchvision import transforms
from flask import Flask, request
from PIL import Image
from io import BytesIO
import base64

model = load_model()

model.load_state_dict()

app = Flask(__name__)


inference_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def image_from_base64(b64):
    data = {'img': b64}
    return Image.open(BytesIO(base64.b64decode(data)))


@app.route('/', methods=['POST'])
def hello_world():
    img = image_from_base64(request.get_json())
    tensor = inference_transforms(img)
    return 'Hello {}!\n'.format(target)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
