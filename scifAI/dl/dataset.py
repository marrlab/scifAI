import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from skimage.util import crop
import copy
import os

seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)


def get_image(h5_file_path, scaling_factor):
    h5_file = h5py.File(h5_file_path, 'r')
    image_original =h5_file.get('image')[()] / scaling_factor
    h5_file.close()
    h5_file = None
    return image_original

def crop_pad_h_w(image_dummy, reshape_size):
    if image_dummy.shape[0] < reshape_size:
        h1_pad = (reshape_size - image_dummy.shape[0]) / 2
        h1_pad = int(h1_pad)
        h2_pad = reshape_size - h1_pad - image_dummy.shape[0]
        h1_crop = 0
        h2_crop = 0
    else:
        h1_pad = 0
        h2_pad = 0
        h1_crop = (reshape_size - image_dummy.shape[0]) / 2
        h1_crop = abs(int(h1_crop))
        h2_crop = image_dummy.shape[0] - reshape_size - h1_crop

    if image_dummy.shape[1] < reshape_size:
        w1_pad = (reshape_size - image_dummy.shape[1]) / 2
        w1_pad = int(w1_pad)
        w2_pad = reshape_size - w1_pad - image_dummy.shape[1]
        w1_crop = 0
        w2_crop = 0
    else:
        w1_pad = 0
        w2_pad = 0
        w1_crop = (reshape_size - image_dummy.shape[1]) / 2
        w1_crop = abs(int(w1_crop))
        w2_crop = image_dummy.shape[1] - reshape_size - w1_crop

    h = [h1_crop, h2_crop, h1_pad, h2_pad]
    w = [w1_crop, w2_crop, w1_pad, w2_pad]
    return h, w


class DatasetGenerator(Dataset):

    def __init__(self,
                metadata,
                label_map,
                selected_channels,
                reshape_size=64,
                scaling_factor = 4095.,
                task="classification",
                transform=None):

        self.metadata = metadata.copy().reset_index(drop = True)
        self.selected_channels = selected_channels
        self.num_channels = len(self.selected_channels)
        self.task = task
        self.reshape_size = reshape_size
        self.label_map = label_map
        self.transform = transform
        self.scaling_factor = scaling_factor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        h5_file_path = self.metadata.loc[idx,"file"]
        image_original= get_image(h5_file_path, self.scaling_factor)
        label = self.metadata.loc[idx,"label"]

        ## creating the image
        h, w = crop_pad_h_w(image_original, self.reshape_size)
        h1_crop, h2_crop, h1_pad, h2_pad = h
        w1_crop, w2_crop, w1_pad, w2_pad = w
        image = np.zeros((  self.num_channels,
                            self.reshape_size,
                            self.reshape_size),
                            dtype=np.float64)

        # filling the image with selected channels
        for i, ch in enumerate(self.selected_channels):
            image_dummy = crop( image_original[:, :, ch],
                                ((h1_crop, h2_crop),
                                (w1_crop, w2_crop)))
            image_dummy = np.pad(image_dummy,
                                ((h1_pad, h2_pad),(w1_pad, w2_pad)),
                                "constant",
                                constant_values = image_dummy.mean())
            image[i, :, :] = image_dummy
            image_dummy = None

        image_original = None

        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))

        if self.transform:
            image = self.transform(image)

        if self.task == "classification":
            # preparing the label part
            label = self.label_map[label]
            label = torch.tensor(label).long()
            return image.float(),  label
        elif self.task == "autoencoder":
            # preparing the label part
            label = image
            return image.float(),  image.float()
        else:
            raise KeyError("%s is not a correct task" % self.task)