from pathlib import Path
import shutil
import os
from PIL import Image, ImageEnhance
import cv2
import random


class DataService:
    def __init__(self, dir_path):
        self.dir_path = dir_path

        self.classification_path_1 = self.dir_path + '\\classification_1'
        self.classification_path_2 = self.dir_path + '\\classification_2'

        Path(f'{self.classification_path_1}\\yes').mkdir(
            parents=True, exist_ok=True)
        Path(f'{self.classification_path_1}\\no').mkdir(
            parents=True, exist_ok=True)
        Path(f'{self.classification_path_2}\\yes').mkdir(
            parents=True, exist_ok=True)
        Path(f'{self.classification_path_2}\\no').mkdir(
            parents=True, exist_ok=True)

    def getData(self):
        vid_dir = self.dir_path+'\\video'
        box_dir = self.dir_path+'\\bboxes'
        vid_files = os.listdir(vid_dir)
        target_files = os.listdir(box_dir)
        counter = 0
        for i in range(len(vid_files)):
            vid = cv2.VideoCapture(f'{vid_dir}\\{vid_files[i]}')
            with open(f'{box_dir}\\{target_files[i]}', 'r+') as read_file:
                while(True):
                    ret, frame = vid.read()
                    if not ret:
                        break
                    counter += 1
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pilimg = Image.fromarray(frame)
                    target = read_file.readline().replace('\n', '').split(';')

                    if target[0] == '#':
                        pilimg.save(
                            f'{self.classification_path_1}\\no\\{counter}.png')
                    else:
                        pilimg.save(
                            f'{self.classification_path_1}\\yes\\{counter}.png')

                    if target[2] == '#':
                        pilimg.save(
                            f'{self.classification_path_2}\\no\\{counter}.png')
                    else:
                        pilimg.save(
                            f'{self.classification_path_2}\\yes\\{counter}.png')

    def augmentData(self):
        for i in range(2):
            if i == 0:
                imagePath = self.classification_path_1
            if i == 1:
                imagePath = self.classification_path_2

            self.augment(imagePath + '\\yes\\', imagePath + '\\no\\')
            print('ALL DONE. GLHF my friend')

    def augment(self, pathToMatch, pathToAugment):
        print(f'Augumenting data for path: {pathToAugment}')
        filesInAugment = os.listdir(pathToAugment)
        numOfAugment = len(filesInAugment)

        numOfMatch = len(os.listdir(pathToMatch))

        print(f'Files in folder to augment: {numOfAugment}')
        print(f'Files in folder to match: {numOfMatch}')

        fileNumber = 1
        fileIndex = 0

        while numOfAugment < numOfMatch:
            im = Image.open(pathToAugment + filesInAugment[fileIndex])
            enhancer = ImageEnhance.Contrast(im)

            ratio = random.uniform(0.5, 1.5)
            rotateAngle = random.uniform(-15.0, 15.0)
            im_out = enhancer.enhance(ratio)

            im_out = im_out.rotate(rotateAngle)

            im_out.save(
                pathToAugment + f'aug_{fileNumber}{os.path.splitext(pathToAugment + filesInAugment[fileIndex])[1]}')

            fileNumber += 1
            fileIndex += 1
            numOfAugment += 1

            if fileIndex >= len(filesInAugment):
                fileIndex = 0
