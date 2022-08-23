import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def read_data(path_to_data):
    X = []
    y = []
    for image_name in os.listdir(path_to_data):
        o_n = os.path.splitext(image_name)[0]
        r = h5py.File(os.path.join(path_to_data, image_name), 'r')
        X.append(int(o_n))
        y.append(r["label"][()])


def get_statistics(dataloader, selected_channels):
    nmb_channels = len(selected_channels)

    statistics = dict()
    statistics["min"] = torch.zeros(nmb_channels)
    statistics["p01"] = torch.zeros(nmb_channels)
    statistics["p05"] = torch.zeros(nmb_channels)
    statistics["p25"] = torch.zeros(nmb_channels)
    statistics["p50"] = torch.zeros(nmb_channels)
    statistics["p75"] = torch.zeros(nmb_channels)
    statistics["p95"] = torch.zeros(nmb_channels)
    statistics["p99"] = torch.zeros(nmb_channels)
    statistics["max"] = torch.zeros(nmb_channels)

    statistics["mean"] = torch.zeros(nmb_channels)
    statistics["std"] = torch.zeros(nmb_channels)

    for _, data_l in enumerate(tqdm(dataloader), 0):
        image, _ = data_l
        for n in range(nmb_channels):
            statistics["min"][n] += image[:, n, :, :].min()
            statistics["p01"][n] += torch.quantile(image[:, n, :, :], 0.01)
            statistics["p05"][n] += torch.quantile(image[:, n, :, :], 0.05)
            statistics["p25"][n] += torch.quantile(image[:, n, :, :], 0.25)
            statistics["p50"][n] += torch.quantile(image[:, n, :, :], 0.50)
            statistics["p75"][n] += torch.quantile(image[:, n, :, :], 0.75)
            statistics["p95"][n] += torch.quantile(image[:, n, :, :], 0.95)
            statistics["p99"][n] += torch.quantile(image[:, n, :, :], 0.99)
            statistics["max"][n] += image[:, n, :, :].max()

            statistics["mean"][n] += image[:, n, :, :].mean()
            statistics["std"][n] += image[:, n, :, :].std()

    # averaging
    for k in statistics:
        statistics[k] = statistics[k].div_(len(dataloader))

    print('statistics used: %s' % (str(statistics)))

    return statistics


def calculate_weights(y_train):
    class_sample_count = np.array([len(np.where(np.asarray(y_train) == t)[0]) \
                                   for t in np.unique(y_train)])
    weights = len(y_train) / class_sample_count
    return weights


def train_validation_test_split(index,
                                y,
                                validation_size=0.20,
                                test_size=0.20,
                                random_state=None):
    train_index, test_index, y_train, _ = train_test_split(index,
                                                           y,
                                                           test_size=test_size,
                                                           stratify=y,
                                                           random_state=random_state)

    train_index, validation_index, _, _ = train_test_split(train_index,
                                                           y_train,
                                                           test_size=validation_size,
                                                           stratify=y_train,
                                                           random_state=random_state)
    return train_index, validation_index, test_index


def plot_heatmap_3_channels(heatmap):
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 15))
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.imshow(heatmap[0])
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.imshow(heatmap[1])
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    ax3.imshow(heatmap[2])
