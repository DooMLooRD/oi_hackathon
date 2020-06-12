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


def load_models():
    device = torch.device('cpu')

    bot_detection_model = DetectionConvNetwork()
    top_detection_model = DetectionConvNetwork()

    bot_detection_model.load_state_dict(torch.load(
        'model\\bot_detection_model.pt', map_location=device))
    top_detection_model.load_state_dict(torch.load(
        'model\\top_detection_model.pt', map_location=device))

    bot_detection_model.eval()
    top_detection_model.eval()

    return bot_detection_model, top_detection_model


def init_data_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    return transform


def log_points(bot_keypoints, top_keypoints):
    x_bot = '#'
    y_bot = '#'
    x_top = '#'
    y_top = '#'

    if bot_keypoints != None:
        bot_keypoints = bot_keypoints.data.numpy()
        x_bot = int(bot_keypoints[0][0].item() * 640)
        y_bot = int(bot_keypoints[0][1].item() * 360)

    if top_keypoints != None:
        top_keypoints = top_keypoints.data.numpy()
        x_top = int(top_keypoints[0][0].item() * 640)
        y_top = int(top_keypoints[0][1].item() * 360)

    print(f'{x_bot};{y_bot};{x_top};{y_top}')


def show_points(frame, bot_keypoints, top_keypoints, index, out):
    if bot_keypoints != None:
        bot_keypoints = bot_keypoints.data.numpy()
        x_bot = int(bot_keypoints[0][0].item() * 640)
        y_bot = int(bot_keypoints[0][1].item() * 360)
        cv2.circle(frame, (x_bot, y_bot), 5, (0, 0, 255), -1)

    if top_keypoints != None:
        top_keypoints = top_keypoints.data.numpy()
        x_top = int(top_keypoints[0][0].item() * 640)
        y_top = int(top_keypoints[0][1].item() * 360)
        cv2.circle(frame, (x_top, y_top), 5, (255, 0, 0), -1)

    out.write(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image.save(f'results\\frames\\{index}.png')


def init_logging(vid):
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
        out = init_logging(vid)

    bot_detection_model, top_detection_model = load_models()
    transform = init_data_loader()
    counter = 1

    while(True):
        ret, frame = vid.read()
        if not ret:
            break
        converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(converted_frame)

        bot_keypoints = None
        top_keypoints = None

        if True:
            bot_keypoints = bot_detection_model(transform(
                image).unsqueeze(1).permute(1, 0, 2, 3))

        if True:
            top_keypoints = top_detection_model(transform(
                image).unsqueeze(1).permute(1, 0, 2, 3))

        # log_points(bot_keypoints, top_keypoints)

        if args['verbose']:
            show_points(frame, bot_keypoints, top_keypoints, counter, out)
        counter += 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", type=str, required=True,
                    help="filename")
    ap.add_argument("-v", "--verbose", dest='verbose',
                    action='store_true', default=False)
    args = vars(ap.parse_args())
    main(args)
