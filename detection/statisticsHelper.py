import torch
import torchvision
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt


def save_plots(train_history, val_history, num_epochs, ticks, filedirectory):
    if(ticks == 0):
        ticks = 1
    display_plot_loss(train_history, val_history,
                      num_epochs, ticks, filedirectory)

    display_val_plot_loss(val_history, num_epochs, ticks, filedirectory)

    display_train_plot_loss(
        train_history, num_epochs, ticks, filedirectory, "epoch")


def display_plot_loss(train_history, val_history, num_epochs, ticks, filedirectory):
    ohist = list(map(lambda x: x.loss, train_history[::int(ticks)]))
    shist = list(map(lambda x: x.loss, val_history[::int(ticks)]))

    plt.title("Loss vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(arange(1, num_epochs+1, ticks), ohist, label="Train")
    plt.plot(arange(1, num_epochs+1, ticks), shist, label="Val")
    plt.xticks(np.arange(1, num_epochs+1, int(num_epochs/10)))
    plt.legend()
    plt.savefig(f'{filedirectory}/plot_loss_{ticks}.png')
    plt.close()


def display_train_plot_loss(train_history, num_epochs, ticks, filedirectory, label):
    ohist = list(map(lambda x: x.loss, train_history[::int(ticks)]))

    plt.title("Loss vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(arange(1, num_epochs+1, ticks), ohist, label="Train")
    plt.xticks(np.arange(1, num_epochs+1, int(num_epochs/10)))
    plt.legend()
    plt.savefig(f'{filedirectory}/plot_train_loss_{ticks}_{label}.png')
    plt.close()


def display_val_plot_loss(val_history, num_epochs, ticks, filedirectory):
    ohist = list(map(lambda x: x.loss, val_history[::int(ticks)]))

    plt.title("Loss vs. Number of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(arange(1, num_epochs+1, ticks), ohist, label="Val")
    plt.xticks(np.arange(1, num_epochs+1, int(num_epochs/10)))
    plt.legend()
    plt.savefig(f'{filedirectory}/plot_val_loss_{ticks}.png')
    plt.close()
