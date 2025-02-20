import torch
import torch.optim as optim
import torch.nn as nn
import copy
from pathlib import Path

from cnnHelper import train_model, val_model, init_train_loaders
from cnnHelper import init_model, load_datasets, checkImprovement, log_result
from statisticsHelper import save_plots, extract_conf_matrix
from resultData import ResultData

from dataService import DataService

import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


def main():
    torch.cuda.empty_cache()

    data_path = 'train\\'
    num_epochs = 200
    batch_size = 64
    variant = 2
    filedirectory = f'results_{variant}'
    filename = 'result'
    max_no_improv_epochs = 20

    Path(f'{filedirectory}/images_best').mkdir(parents=True, exist_ok=True)

    dataService = DataService(data_path)

    downloadData = input('Do you want to extract data? [Y/N]: ')
    if downloadData.lower() == 'y':
        dataService.getData()
    augmentData = input('Do you want to augment the data? [Y/N]: ')
    if augmentData.lower() == 'y':
        dataService.augmentData()

    train_set, val_set, num_classes, classes = load_datasets(
        data_path, variant)

    train_loader, val_loader = init_train_loaders(
        train_set, val_set, batch_size)

    model = init_model(num_classes)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=1, rho=0.8)

    result_data = ResultData(
        f'{filedirectory}/{filename}', num_classes)
    best_model_wts = copy.deepcopy(model.state_dict())

    no_improv_epochs = 0
    current_epoch = 0
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        current_epoch += 1

        train_result, train_time = train_model(model, train_loader, criterion, optimizer,
                                               num_classes, 'Training')
        val_result, val_time = val_model(model, val_loader, criterion,
                                         num_classes)

        result_data.time += train_time + val_time

        print('-' * 10)
        print()

        best_model_wts, improved = log_result(
            model, result_data, best_model_wts, train_result, val_result)
        result_data.save()
        torch.save(model.state_dict(),
                   F'{filedirectory}/{epoch+1}_model.pt')

        # Early stopping
        no_improv_epochs, should_stop = checkImprovement(
            improved, no_improv_epochs, max_no_improv_epochs)

        if(should_stop):
            print('Early stopping... No improvement')
            model.load_state_dict(best_model_wts)
            break

    print('Finished Training, Final time: {:.0f}m {:.0f}s'.format(
        result_data.time // 60, result_data.time % 60))

    torch.save(model.state_dict(),
               f'{filedirectory}/best_model.pt')
    result_data.print_best()
    extract_conf_matrix(result_data.val_best.cm, num_classes, classes)

    save_plots(result_data.train_history, result_data.val_history,
               current_epoch, 1, filedirectory)


if __name__ == '__main__':
    main()
