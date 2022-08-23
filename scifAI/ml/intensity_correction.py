import numpy as np
from skimage.exposure import rescale_intensity
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class IntensityCorrection(BaseEstimator, TransformerMixin):
    """maps the image to specific range
    
    Based on the in_range and out_range, it changes the range of the image per channel:
    https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity

    Parameters
    ----------
    in_range : list (size C) where each element is a tuple of (min_input, max_range)
 
    out_range : list (size C) where each element is a tuple of (min_output, max_output)

    Returns
    -------
    image : rescaled intensity image
    mask: the mask of the same image (no change applied) 

    """
    def __init__(self, in_range, out_range):
        self.in_range = in_range
        self.out_range = out_range
        
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy()
        mask = X[1].copy()

        if len(self.in_range) != image.shape[2]:
            raise IndexError("Length of the input ranges \
                and number of channels are inconsisntent")
        
        elif len(self.out_range) != image.shape[2]:
            raise IndexError("Length of the output ranges \
                and number of channels are inconsisntent")


        for ch in range(image.shape[2]):
            image[:,:,ch] = rescale_intensity(  image[:,:,ch], 
                                                in_range=self.in_range[ch], 
                                                out_range=self.out_range[ch])
 
        return [image, mask]
            
