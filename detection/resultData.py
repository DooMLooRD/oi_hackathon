import json
import torch
import numpy as np


class ResultData:
    def __init__(self, filename):
        self.filename = filename

        self.train_history = []
        self.val_history = []
        self.val_best = 1000
        self.time = 0.0

    def print_best(self):
        print('Best val Loss: {:4f}'.format(self.val_best))

    @classmethod
    def from_json(cls, data):
        return cls(**data)

    def save(self):
        with open(f'{self.filename}.json', 'w+') as json_file:
            data = json.dumps(self, default=lambda o: o.__dict__,
                              sort_keys=True, indent=4)
            json_file.write(data)

        with open(f'{self.filename}.txt', 'w+') as text_file:
            text_file.write(
                '\n Epoch {} - Time summary: {:.0f}m {:.0f}s'.format(len(self.val_history),
                                                                     self.time // 60, self.time % 60))
            text_file.write('-' * 10)
            text_file.write('\nTrain Loss: {:.4f} Val Loss: {:.4f}'.format(
                self.train_history[-1], self.val_history[-1]))
