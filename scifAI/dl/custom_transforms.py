import torch
from pathlib import Path
from typing import Sequence, Union
import numpy as np
from monai.transforms.compose import Transform
from PIL import Image
import os
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
from scipy.stats import kurtosis, skew
from scipy.spatial import distance as dist
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from skimage.measure import label, regionprops
from skimage.measure import moments_hu, inertia_tensor, inertia_tensor_eigvals
from skimage.measure import moments


class LoadTensor(object):
    """
    Load the test samples from already preprocessed dataset
    """

    def __init__(self, part=1.0) -> None:
        self.part = part

    def __call__(self, name: Union[Sequence[Union[Path, str]], Path, str]):
        sample_tensor = torch.load(name)
        return sample_tensor[0], sample_tensor[1]


class ShuffleChannel(object):
    """
    Shuffle the chosen channel
    """

    def __init__(self, channels_to_shuffle=[], perturb=False) -> None:
        self.channels_to_shuffle = channels_to_shuffle
        self.perturb = perturb

    def __call__(self, image):
        if len(self.channels_to_shuffle) > 0:
            for channel in self.channels_to_shuffle:
                channel_shape = image[channel].shape
                if self.perturb:
                    image[channel] = torch.full(channel_shape, torch.mean(image[channel]))
                else:
                    image[channel] = image[channel].flatten()[
                        torch.randperm(len(image[channel].flatten()))].reshape(
                        channel_shape)
        return image


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.5):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class LoadImage(Transform):
    """
    Load common 2D image format (PNG, JPG, etc. using PIL) file or files from provided path.
    If loading a list of files, stack them together and add a new dimension as first dimension,
    and use the meta data of the first image to represent the stacked result.
    It's based on the Image module in PIL library:
    https://pillow.readthedocs.io/en/stable/reference/Image.html
    """

    def __init__(self, only_channels=[]) -> None:
        self.only_channels = only_channels

    def __call__(self, name: Union[Sequence[Union[Path, str]], Path, str]):
        if isinstance(name, (np.ndarray, np.generic)):
            name = name[0]
        img_array = list()
        class_dir = os.path.dirname(name)
        sample_id = os.path.basename(name) + "_Ch"
        channels = [np.asarray(Image.open(os.path.join(class_dir, file))) / 255. for file in os.listdir(class_dir) if
                    file.startswith(sample_id)]
        image = np.stack(channels, axis=2)
        image_RGB = image * 255.
        return Image.fromarray(np.uint8(image_RGB))


class ToTensorCustom(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply the transform to `img` and make it contiguous.
        """
        # img = np.asarray(img)
        # image = torch.from_numpy(copy.deepcopy(img))
        if torch.is_tensor(img):
            return img.contiguous().T
        # Tracer()()
        return torch.as_tensor(np.ascontiguousarray(img) / 255.).T


class RandomCrop3D():
    def __init__(self, img_sz, crop_sz):
        c, h, w, d = img_sz
        assert (h, w, d) > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)

    def __call__(self, x):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        images = []
        for image in X:
            images.append(image[self.key].cpu().numpy())
        return np.asarray(images)
