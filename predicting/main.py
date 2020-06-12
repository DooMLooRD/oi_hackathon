from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torchvision.transforms as transforms
from pathlib import Path
import argparse

from detectionConvNetwork import DetectionConvNetwork


def loadModels():
    model = DetectionConvNetwork()
    device = torch.device('cpu')
    model.load_state_dict(torch.load(
        'model\\best_model_157_normalization.pt', map_location=device))
    model.eval()

    return model


def initDataLoader():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    return train_transform


def logpoints(keypoints):
    keypoints = keypoints.data.numpy()
    x = int(keypoints[0][0].item() * 640)
    y = int(keypoints[0][1].item() * 360)
    print(f'{x};{y}')


def showpoints(frame, keypoints, index, out):
    keypoints = keypoints.data.numpy()
    x = int(keypoints[0][0].item() * 640)
    y = int(keypoints[0][1].item() * 360)

    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    out.write(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image.save(f'results\\frames\\{index}.png')


def initLogging(vid):
    Path('results\\video').mkdir(parents=True, exist_ok=True)
    Path('results\\frames').mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    return cv2.VideoWriter('results\\video\\output.mp4', fourcc, 24.0,
                           (frame_width, frame_height))


def main(args):
    vid = cv2.VideoCapture(args['filename'])
    if args['verbose']:
        out = initLogging(vid)

    model = loadModels()
    transform = initDataLoader()
    counter = 1

    while(True):
        ret, frame = vid.read()
        if not ret:
            break
        converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(converted_frame)
        keypoints = model(transform(
            image).unsqueeze(1).permute(1, 0, 2, 3))
        # logpoints(keypoints)
        if args['verbose']:
            showpoints(frame, keypoints, counter, out)
        counter += 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", type=str, required=True,
                    help="filename")
    ap.add_argument("-v", "--verbose", dest='verbose',
                    action='store_true', default=False)
    args = vars(ap.parse_args())
    main(args)
