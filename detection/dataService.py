from pathlib import Path
import shutil
import os
from PIL import Image, ImageEnhance
import cv2


class DataService:
    def __init__(self, dir_path):
        self.dir_path = dir_path

        self.detection_targets_path_1 = self.dir_path + '\\detection_targets_1'
        self.detection_targets_path_2 = self.dir_path + '\\detection_targets_2'

        self.detection_frames_path_1 = self.dir_path + '\\detection_frames_1'
        self.detection_frames_path_2 = self.dir_path + '\\detection_frames_2'

        Path(self.detection_targets_path_1).mkdir(parents=True, exist_ok=True)
        Path(self.detection_targets_path_2).mkdir(parents=True, exist_ok=True)

        Path(self.detection_frames_path_1).mkdir(parents=True, exist_ok=True)
        Path(self.detection_frames_path_2).mkdir(parents=True, exist_ok=True)

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

                    if target[0] != '#':
                        pilimg.save(
                            f'{self.detection_frames_path_1}\\{counter}.png')
                        with open(f'{self.detection_targets_path_1}\\{counter}.txt', 'w+') as write_file:
                            write_file.write(f'{target[0]};{target[1]}\n')

                    if target[2] != '#':
                        pilimg.save(
                            f'{self.detection_frames_path_2}\\{counter}.png')
                        with open(f'{self.detection_targets_path_2}\\{counter}.txt', 'w+') as write_file:
                            write_file.write(f'{target[2]};{target[3]}\n')
