from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2
import torchvision.transforms as transforms

from detectionConvNetwork import DetectionConvNetwork


def showpoints(image, keypoints):

    plt.figure()
    keypoints = keypoints.data.numpy()
    plt.imshow(image)
    plt.scatter(keypoints[0][0].item(), keypoints[0]
                [1].item(), s=50, marker='.', c='r')
    plt.show()


def main():
    model = DetectionConvNetwork()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()
    train_transform = transforms.Compose([
        # transforms.Resize((640, 360)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])
    vid_path = 'video.mp4'
    vid = cv2.VideoCapture(vid_path)
    while(True):
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        keypoints = model(train_transform(
            image).unsqueeze(1).permute(1, 0, 2, 3))
        print(keypoints)
        showpoints(image, keypoints)


if __name__ == '__main__':
    main()
