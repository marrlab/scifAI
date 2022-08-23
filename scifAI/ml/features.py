import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from scipy.stats import kurtosis, skew
from scipy.spatial import distance as dist
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from skimage.measure import regionprops_table
from skimage.filters import  sobel
from skimage.feature import hog
from skimage.transform import resize
from skimage.metrics import structural_similarity,hausdorff_distance

class MaskBasedFeatures(BaseEstimator, TransformerMixin):
    """
    mask based features
    """
    def __init__(self):
        self.properties = [ "area", 
                            "bbox_area", 
                            "convex_area",
                            "eccentricity", 
                            "equivalent_diameter", 
                            "euler_number", 
                            "extent", 
                            "feret_diameter_max",
                            "filled_area", 
                            "major_axis_length", 
                            "max_intensity",
                            "mean_intensity",
                            "min_intensity",
                            "minor_axis_length", 
                            "moments_hu",
                            "orientation",
                            "perimeter",
                            "perimeter_crofton",
                            "solidity",
                            "weighted_moments_hu"] 
        
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy()
        mask = X[1].copy()
        feature_names = regionprops_table(mask[:,:,0].astype(int),
                                       image[:,:,0],
                                       properties = self.properties).keys()
        # storing the feature values
        features = dict()
        for ch in range(image.shape[2]):
            calculated_features = regionprops_table(mask[:,:,ch].astype(int), 
                                                            image[:,:,ch],
                                                            properties = self.properties)
            
            for p in feature_names:
                deignated_name = "mask_based_" +  p + "_Ch" + str(ch+1)
                try:
                    features[deignated_name] = calculated_features[p][0]   
                except: 
                    features[deignated_name] = -1    
        return features
            

class GLCMFeatures(BaseEstimator, TransformerMixin):
    """calculates the glcm features 
    
    Calculates the features per channel using glcm features including
    contrast, dissimilarity, homogeneity, ASM, energy and correlation.
    For more info please refer to:
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels. 

    Returns
    -------
    features :  dict  
        dictionary including 'contrast_Chx', 'dissimilarity_Chx', 'homogeneity_Chx'
        'ASM_Chx', 'energy_Chx' and 'correlation_Chx' per channel where 
        x will be substituted by the channel number starting from 1. 

    """

    def __init__(   self, 
                    distances=[5],
                    angles=[0],
                    levels=256):
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy()
        features = dict()
        for ch in range(image.shape[2]):
            # create a 2D temp image 
            temp_image = image[:,:,ch].copy()
            temp_image = (temp_image/temp_image.max())*255 # use 8bit pixel values for GLCM
            temp_image = temp_image.astype('uint8') # convert to unsigned for GLCM

            # calculating glcm
            glcm = greycomatrix(temp_image,
                                distances=self.distances,
                                angles=self.angles,
                                levels=self.levels)

            # storing the glcm values
            features["contrast_Ch" + str(ch+1)] = greycoprops(glcm, prop='contrast')[0,0]
            features["dissimilarity_Ch" + str(ch+1)] = greycoprops(glcm, prop='dissimilarity')[0,0]
            features["homogeneity_Ch" + str(ch+1)] = greycoprops(glcm, prop='homogeneity')[0,0]
            features["ASM_Ch" + str(ch+1)] = greycoprops(glcm, prop='ASM')[0,0]
            features["energy_Ch" + str(ch+1)] = greycoprops(glcm, prop='energy')[0,0]
            features["correlation_Ch" + str(ch+1)] = greycoprops(glcm, prop='correlation')[0,0]
        return features


class GradientRMS(BaseEstimator, TransformerMixin):
    """calculates the RMS of the gradient of the image
    
    calculates the RMS of the gradient of the image. It can be used to check whether
    an image is focused or not.

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels. 

    Returns
    -------
    features :  dict  
        dictionary including 'RMS_Chx' 

    """
    def __init__(self):
        self.RMS = "RMS"

    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy()
        features = dict()
        for ch in range(image.shape[2]):
            features["gradient_RMS_Ch" + str(ch+1)] = np.sqrt(np.mean(np.square(sobel(image[:,:,ch])))) 
        return features

class BackgroundMean(BaseEstimator, TransformerMixin):
    """calculates average background pixels
    
    calculates average background pixels. It can be used as a sanity check for bleeding through

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels. 

    Returns
    -------
    features :  dict  
        dictionary including 'RMS_Chx' 
    """
    def __init__(self):
        self.BackgroundMean = "BackgroundMean"

    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy()
        mask = X[1].copy()
        background_mask = (mask == 0).astype(int)
        background_image = image*background_mask 
        # storing the feature values
        features = dict()
        for ch in range(image.shape[2]):
            background = background_image[:,:,ch].copy()
            indx = background > 0
            features["background_mean_Ch" + str(ch+1)] = background[indx].mean()
        return features

class CellShape(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.BackgroundMean = "BackgroundMean"

    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy()
        mask = X[1].copy()    
        segmented_cell = image.copy() * mask.copy()    
        # storing the feature values
        features = dict()
        for ch in range(image.shape[2]):
            indx = segmented_cell[:,:,ch] > 0.
            if indx.sum() > 0:
                features["Area_Ch" + str(ch+1)] = mask[:,:,ch].sum()
                features["sum_intensity_Ch" + str(ch+1)] = segmented_cell[:,:,ch].sum()
                features["mean_intensity_Ch" + str(ch+1)] = segmented_cell[:,:,ch][indx].mean()
                features["std_intensity_Ch" + str(ch+1)] = segmented_cell[:,:,ch][indx].std()
                features["kurtosis_intensity_Ch" + str(ch+1)] = kurtosis(segmented_cell[:,:,ch][indx]) 
                features["skew_intensity_Ch" + str(ch+1)] = skew(segmented_cell[:,:,ch][indx])  
                features["min_intensity_Ch" + str(ch+1)] = segmented_cell[:,:,ch][indx].min()  
                features["max_intensity_Ch" + str(ch+1)] = segmented_cell[:,:,ch][indx].max()
                features["shannon_entropy_Ch" + str(ch+1)] = shannon_entropy(segmented_cell[:,:,ch][indx])
            else:
                features["Area_Ch" + str(ch+1)] = 0.
                features["sum_intensity_Ch" + str(ch+1)] = 0.
                features["mean_intensity_Ch" + str(ch+1)] = 0.
                features["std_intensity_Ch" + str(ch+1)] = 0.
                features["kurtosis_intensity_Ch" + str(ch+1)] = 0.
                features["skew_intensity_Ch" + str(ch+1)] = 0. 
                features["min_intensity_Ch" + str(ch+1)] = 0.
                features["max_intensity_Ch" + str(ch+1)] = 0. 
                features["shannon_entropy_Ch" + str(ch+1)] = 0.
        return features


class Collocalization(BaseEstimator, TransformerMixin):
    """calculates the cross channel distance features 
    
    Calculates the distances across channels 

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels. 

    Returns
    -------
    features :  dict  
        dictionary including different distances across channels

    """

    def __init__(self):
        self.Collocalization = "Collocalization"

    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy()
        mask = X[1].copy()      
        features = dict()
        for ch1 in range(mask.shape[2]):
            for ch2 in range(ch1+1,mask.shape[2]):
                # rehaping the channels to 1D
                channel1 = mask[:,:,ch1].ravel()
                channel2 = mask[:,:,ch2].ravel()

                # creating the suffix name for better readability
                suffix = "_Ch" + str(ch1 + 1) + "_Ch" + str(ch2 + 1)

                # storing the distance values
                features["dice_distance" + suffix] = dist.dice(channel1,channel2)
                features["jaccard_distance" + suffix] = dist.jaccard(channel1,channel2)

        for ch1 in range(image.shape[2]):
            for ch2 in range(image.shape[2]):
                if ch1 != ch2:
                    # rehaping the channels to 1D
                    channel1 =  (image[:,:,ch1].copy() * mask[:,:,ch1].copy() ) 
                    channel2 =  (image[:,:,ch2].copy() * mask[:,:,ch2].copy() ) 

                    # creating the suffix name for better readability
                    suffix = "_R" + str(ch1 + 1) +"_Ch" + str(ch1 + 1) + "_R" + str(ch2 + 1) +"_Ch" + str(ch2 + 1)
                    features["correlation_distance" + suffix] = dist.correlation(   channel1.ravel(),
                                                                                    channel2.ravel())
                    features["euclidean_distance" + suffix] = dist.euclidean(   channel1.ravel(),
                                                                                channel2.ravel())
                    features["manders_overlap_coefficient" + suffix] = (channel1.sum()*channel2.sum())/(np.power(channel1,2).sum()*np.power(channel2,2).sum())
                    features["intensity_correlation_quotient" + suffix] = ((channel1>channel1.mean())*(channel2>channel2.mean())).sum()/(channel1.shape[0]) - 0.5 
                    features["structural_similarity" + suffix] = structural_similarity(channel1,channel2)
                    features["hausdorff_distance" + suffix] = hausdorff_distance(channel1,channel2)   
                
                
        return features
 

class PercentileFeatures(BaseEstimator, TransformerMixin):
    """calculates the percentile features per channel 
    
    Calculates the percentile for different channels
    For more info please refer to:
    https://scikit-image.org/docs/dev/api/skimage.exposure.html

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.
    n_bins : positive int 
        number of bins

    Returns
    -------
    features :  dict  
        dictionary including hist_0_Ch1, hist_1_Ch1 ...

    """  

    def __init__(self,  cuts= range(10,100,10)):
        self.cuts = cuts

    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy() 
        features = dict()
        for ch in range(image.shape[2]):
            for i, cut in enumerate(self.cuts):
                perc  = np.percentile(image[:,:,ch].ravel(), cut)
                features["percentile_" + str(cut) +  "_Ch" + str(ch+1)] = perc
                
        return features

class HogFeatures(BaseEstimator, TransformerMixin):
    """calculates the set of hog features 
    
    Calculates the hog features with
    For more info please refer to:
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.

    Returns
    -------
    features :  dict  
        dictionary including hog_1, hog_2 ...

    """

    def __init__(self , size = (64,64)):
        self.size = size

    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy() 
        
        features = dict()
        for ch in range(image.shape[2]):
            temp_image = resize(image[:,:,ch].copy(), self.size)
            # calculating the pixels per cells 
            hog_features= hog(  temp_image, 
                                orientations=8, 
                                pixels_per_cell=(12, 12),
                                cells_per_block=(1, 1), 
                                visualize=False, 
                                multichannel=False)
        
            for i in range(len(hog_features)):
                features["Hog_" + str(i) + "_Ch" + str(ch+1)] = hog_features[i]
                
        return features


class IntersectionProperties(BaseEstimator, TransformerMixin):
    """Properties in the intersection of the cells
    
    Properties in the intersection of the cells 

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.
    mask : 3D array, shape (M, N, C)
        The input mask with multiple channels.

    Returns
    -------
    features :  dict  
        dictionary including sum_intensity_ratio, mean_intensity_ratio ...

    """

    def __init__(self , channels = None, eps = 1e-12):
        self.eps = eps
        self.channels = channels
            
    def fit(self, X = None, y = None):        
        return self

    def transform(self,X):
        image = X[0].copy() 
        mask = X[1].copy()
        segmented_cell = image.copy() * mask.copy()
        n_channels = image.shape[2]

        if self.channels is None:
            self.channels = range(n_channels)

        features = dict()
        for ch1 in range(0,n_channels):
            for ch2 in range(ch1+1,n_channels): 
                intersection_mask = mask[:,:,ch1].copy() * mask[:,:,ch2].copy()    
                for ch in self.channels:
                    suffix = "_Ch" + str(ch + 1) + "_R" + str(ch1 + 1) + "_R" + str(ch2 + 1)
                    intersected_cell = segmented_cell[:,:,ch] * intersection_mask
                    features["sum_intensity_ratio" + suffix] = intersected_cell.sum() / (segmented_cell[:,:,ch].sum() + self.eps)
                    features["mean_intensity_ratio" + suffix] = intersected_cell.mean() / (segmented_cell[:,:,ch].mean() + self.eps) 
                    features["max_intensity_ratio" + suffix] = intersected_cell.max() / (segmented_cell[:,:,ch].mean() + self.eps)        
        return features


class CenterOfCellsDistances(BaseEstimator, TransformerMixin):
    """Distances of the cells
    
    Distances of the cells

    Parameters
    ----------
    image : 3D array, shape (M, N, C)
        The input image with multiple channels.
    mask : 3D array, shape (M, N, C)
        The input mask with multiple channels.

    Returns
    -------
    features :  dict  
        dictionary including hog_1, hog_2 ...

    """
    def __init__(self):
        self.properties = [ "centroid",
                            "weighted_centroid" ] 
        
    
    def fit(self, X = None, y = None):        
        return self
    
    def transform(self,X):
        image = X[0].copy()
        mask = X[1].copy()
 
        # storing the feature values
        features = dict()
        n_channels = image.shape[2]
        for ch1 in range(0,n_channels):
            for ch2 in range(ch1+1,n_channels): 
                channel1 = regionprops_table(mask[:,:,ch1].astype(int), 
                                                image[:,:,ch1], 
                                                properties = self.properties )  
                    
                channel2 = regionprops_table(mask[:,:,ch2].astype(int), 
                                                image[:,:,ch2], 
                                                properties = self.properties ) 

                ## distance feature
                feature_name = "cell_distance_Ch" + str(ch1 + 1) + "_Ch" +  str(ch2 + 1)
                try:
                    features[feature_name] = (channel2['centroid-0'][0] - channel1['centroid-0'][0])**2
                    features[feature_name] += (channel2['centroid-1'][0] - channel1['centroid-1'][0])**2
                    features[feature_name] = np.sqrt(features[feature_name])
                except IndexError:
                    features[feature_name] = -1.

                ## weighted distance feature  
                feature_name = "weighted_cell_distance_Ch" + str(ch1 + 1) + "_Ch" +  str(ch2 + 1)
                try:
                    features[feature_name] = (channel2['weighted_centroid-0'][0] - channel1['weighted_centroid-0'][0])**2
                    features[feature_name] += (channel2['weighted_centroid-1'][0] - channel1['weighted_centroid-1'][0])**2
                    features[feature_name] = np.sqrt(features[feature_name])
                except IndexError:
                    features[feature_name] = -1.

        return features