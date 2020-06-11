import torch
import torch.optim as optim
import torch.nn as nn
import copy
from pathlib import Path

from cnnHelper import train_model, val_model, init_train_loaders
from cnnHelper import init_model, load_datasets, checkImprovement, log_result
from statisticsHelper import save_plots
from resultData import ResultData

from dataService import DataService

import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


def main():
    torch.cuda.empty_cache()

    data_path = 'train\\'
    num_epochs = 300
    batch_size = 32
    variant = 1
    filedirectory = f'results_{variant}'
    filename = 'result'
    max_no_improv_epochs = 20

    Path(f'{filedirectory}/images_best').mkdir(parents=True, exist_ok=True)

    dataService = DataService(data_path)

    extractData = input('Do you want to extract data? [Y/N]: ')
    if extractData.lower() == 'y':
        dataService.getData()

    train_set, val_set = load_datasets(data_path, variant)

    train_loader, val_loader = init_train_loaders(
        train_set, val_set, batch_size)

    model = init_model()
    model.cuda()

    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adadelta(model.parameters(), lr=1, rho=0.8)

    result_data = ResultData(
        f'{filedirectory}/{filename}')
    best_model_wts = copy.deepcopy(model.state_dict())

    no_improv_epochs = 0
    current_epoch = 0
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        current_epoch += 1
        epoch_train_results = []
        epoch_val_results = []

        train_result, train_time = train_model(
            model, train_loader, criterion, optimizer, 'Training')
        val_result, train_time = val_model(
            model, val_loader, criterion)

        result_data.time += train_time

        print()

        best_model_wts, improved = log_result(
            model, result_data, best_model_wts, train_result, val_result)
        result_data.save()
        torch.save(model.state_dict(),
                   F'{filedirectory}/{epoch+1}_model.pt')

        # Early stopping
        no_improv_epochs, shold_stop = checkImprovement(
            improved, no_improv_epochs, max_no_improv_epochs)

        if(shold_stop):
            print('Early stopping... No improvement')
            model.load_state_dict(best_model_wts)
            break

    print('Finished Training, Final time: {:.0f}m {:.0f}s'.format(
        result_data.time // 60, result_data.time % 60))

    torch.save(model.state_dict(),
               f'{filedirectory}/best_model.pt')
    result_data.print_best()

    save_plots(result_data.train_history, result_data.val_history,
               current_epoch, int(current_epoch/10), filedirectory)


if __name__ == '__main__':
    main()
