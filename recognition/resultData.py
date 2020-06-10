import json
import torch
import numpy as np


class SingleEpochData:
    def __init__(self, acc, loss, cm):
        self.acc = acc
        self.loss = loss
        self.cm = cm


class ResultData:
    def __init__(self, filename, n_classes):
        self.filename = filename

        self.train_history = []
        self.val_history = []
        self.time = 0.0
        self.val_best = SingleEpochData(
            0, 1, torch.zeros(n_classes, n_classes))

    def print_best(self):
        print('Best val Acc: {:4f}'.format(self.val_best.acc))
        print('Best val Loss: {:4f}'.format(self.val_best.loss))
        print('Best val conf matrix:')
        print(*self.val_best.cm.tolist(), sep="\n")

    @classmethod
    def from_json(cls, data):
        return cls(**data)

    def save(self):
        with open(f'{self.filename}.json', 'w+') as json_file:
            data = json.dumps(self, default=lambda o: o.__dict__,
                              sort_keys=True, indent=4)
            json_file.write(data)

        with open(f'{self.filename}.txt', 'a+') as text_file:
            text_file.write(
                '\n Epoch {} - Time summary: {:.0f}m {:.0f}s'.format(len(self.val_history),
                                                                     self.time // 60, self.time % 60))
            text_file.write('-' * 10)
            text_file.write('\nTrain Acc: {:.4f} Val Acc: {:.4f}'.format(
                self.train_history[-1].acc, self.val_history[-1].acc))
            text_file.write('\nTrain Loss: {:.4f} Val Loss: {:.4f}'.format(
                self.train_history[-1].loss, self.val_history[-1].loss))
            text_file.write('\nTrain Confusion Matrix:\n')
            text_file.write('\n'.join('{}'.format(k)
                                      for k in self.train_history[-1].cm.tolist()))
            text_file.write('\nVal Confusion Matrix:\n')
            text_file.write('\n'.join('{}'.format(k)
                                      for k in self.val_history[-1].cm.tolist()))
