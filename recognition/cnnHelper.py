import time
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from printHelper import printProgressBar
from statisticsHelper import confusion_matrix
from resultData import SingleEpochData
from sklearn.model_selection import train_test_split


def init_model(num_classes):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)
    model.num_classes = num_classes
    return model


def load_datasets(path, variant):
    # DATA AUGMENTATION
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=.20, contrast=.20, saturation=.20, hue=.20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.ImageFolder(
        root=path+f'classification_{variant}',  transform=train_transform)

    indices = list(range(len(train_set)))
    train_index, val_index, train_target, val_target = train_test_split(indices, train_set.targets,
                                                                        stratify=train_set.targets, test_size=0.2)
    splitted_train_set = torch.utils.data.Subset(train_set, train_index)
    splitted_val_set = torch.utils.data.Subset(train_set, val_index)

    return splitted_train_set, splitted_val_set, len(train_set.classes), train_set.classes


def init_train_loaders(train_set, val_set, batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=False,
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
    if(result_data.val_best.loss > epoch_val.loss):
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


def train_model(model, training_loader, criterion, optimizer, n_classes, progress_title):
    since = time.time()
    model.train()

    running_loss = 0.0
    running_corrects = 0
    conf_matrix = torch.zeros(n_classes, n_classes)

    printProgressBar(progress_title, 0, len(training_loader), length=50)

    for i, (inputs, labels) in enumerate(training_loader, 0):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        conf_matrix = confusion_matrix(outputs, labels, conf_matrix)
        printProgressBar(progress_title, i+1, len(training_loader), length=50)

    epoch_loss = running_loss / len(training_loader.dataset)
    epoch_acc = float(running_corrects) / \
        len(training_loader.dataset)

    data = SingleEpochData(epoch_acc, epoch_loss, conf_matrix)
    print()

    time_elapsed = time.time() - since
    print('Training Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(
        epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60))

    return data, time_elapsed


def val_model(model, val_loader, criterion, n_classes):
    since = time.time()
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    conf_matrix = torch.zeros(n_classes, n_classes)
    improved = False

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            conf_matrix = confusion_matrix(outputs, labels, conf_matrix)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = float(running_corrects) / \
        len(val_loader.dataset)

    data = SingleEpochData(epoch_acc, epoch_loss, conf_matrix)
    time_elapsed = time.time() - since

    print('Val Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(
        epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60))
    print()

    return data, time_elapsed
