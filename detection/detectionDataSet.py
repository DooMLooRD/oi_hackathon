import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DetectionDataSet(Dataset):
    def __init__(self, root, variant, transforms):
        self.root = root
        self.transforms = transforms
        self.variant = variant
        self.frames = list(
            sorted(os.listdir(os.path.join(root, f'detection_frames_{variant}'))))
        self.targets = list(
            sorted(os.listdir(os.path.join(root, f'detection_targets_{variant}'))))

    def __getitem__(self, idx):
        frame_path = os.path.join(
            self.root, f'detection_frames_{self.variant}', self.frames[idx])
        target_path = os.path.join(
            self.root, f'detection_targets_{self.variant}', self.targets[idx])
        frame = Image.open(frame_path).convert("RGB")
        with open(target_path, 'r') as file:
            target = file.read().replace('\n', '').split(';')
        points = []
        width, height = frame.size
        for i in range(len(target)):
            target_val = target[i]
            # if i % 2 == 0:
            #     target_val = target_val/width
            # else:
            #     target_val = target_val/height
            points.append(int(target_val))

        position = torch.as_tensor(points, dtype=torch.float32)
        image_id = torch.tensor([idx])

        target = position

        if self.transforms is not None:
            frame = self.transforms(frame)

        return frame, target

    def __len__(self):
        return len(self.frames)
