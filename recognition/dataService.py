from pathlib import Path
import shutil
import os
from PIL import Image, ImageEnhance
import cv2


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
