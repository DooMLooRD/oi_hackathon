import time
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

from printHelper import printProgressBar
from detectionConvNetwork import DetectionConvNetwork
from detectionDataSet import DetectionDataSet


def init_model():
    model = DetectionConvNetwork()
    return model


def load_datasets(path, variant):
    train_transform = transforms.Compose([
        # transforms.Resize((640, 360)),
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_set = DetectionDataSet(
        root=path, variant=variant, transforms=train_transform)

    length = len(train_set)
    train_len = int(length*0.8)
    val_len = length - train_len
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_len, val_len])
    return train_set, val_set


def init_train_loaders(train_set, val_set, batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                               batch_size=batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_set, pin_memory=False,
                                             batch_size=batch_size,
                                             shuffle=False)

    return train_loader, val_loader


def log_result(model, result_data, best_model_wts, epoch_train, epoch_val):
    result_data.train_history.append(epoch_train)
    result_data.val_history.append(epoch_val)
    improved = False
    if(result_data.val_best > epoch_val):
        result_data.val_best = epoch_val
        best_model_wts = copy.deepcopy(model.state_dict())
        improved = True

    return best_model_wts, improved


def checkImprovement(improved, no_improv_epochs, max_no_improv_epochs):
    should_stop = False
    if(improved):
        no_improv_epochs = 0
    else:
        no_improv_epochs += 1

    if(no_improv_epochs == max_no_improv_epochs):
        should_stop = True

    return no_improv_epochs, should_stop


def train_model(model, training_loader, criterion, optimizer, progress_title):
    since = time.time()
    model.train()

    running_loss = 0.0
    running_corrects = 0

    printProgressBar(progress_title, 0, len(training_loader), length=50)

    for i, (inputs, labels) in enumerate(training_loader, 0):
        inputs, labels = inputs.cuda(), labels.cuda()
        labels = labels.view(labels.size(0), -1)

        inputs = inputs.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.FloatTensor).cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        printProgressBar(progress_title, i+1, len(training_loader), length=50)

    epoch_loss = np.sqrt(running_loss / len(training_loader.dataset))

    print()

    time_elapsed = time.time() - since
    print('Training Loss: {:.4f} Time: {:.0f}m {:.0f}s'.format(
        epoch_loss, time_elapsed // 60, time_elapsed % 60))

    return epoch_loss, time_elapsed


def val_model(model, val_loader, criterion):
    since = time.time()
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    improved = False

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            labels = labels.view(labels.size(0), -1)
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels.type(torch.FloatTensor).cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = np.sqrt(running_loss / len(val_loader.dataset))
    time_elapsed = time.time() - since

    print('Val Loss: {:.4f} Time: {:.0f}m {:.0f}s'.format(
        epoch_loss, time_elapsed // 60, time_elapsed % 60))
    print()

    return epoch_loss, time_elapsed
