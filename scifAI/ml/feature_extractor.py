import pandas as pd
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
from scifAI.ml.segmentation import segment_all_channels
from scifAI.utils import list_of_dict_to_dict

class FeatureExtractor(object):
    def __init__(self, feature_unions):
        self.feature_unions = feature_unions

    def extract_(self, f):
        try:
            h5_file = h5py.File(f, "r")      
            image = h5_file.get("image")[()]*1.
            try:
                mask  = h5_file.get("mask")[()]
            except TypeError:
                mask = segment_all_channels(image)
            h5_file.close()
            features = self.feature_unions.transform([image,mask]).copy()
            features = list_of_dict_to_dict(features)
        except:
            print("corrupted file", f)
            features = []
        return features

    def extract_features(self, metadata, n_jobs = -1):
        file_list = metadata["file"].tolist()
        results = Parallel(n_jobs=n_jobs)(delayed(self.extract_)(f) \
            for f in tqdm(file_list, position=0, leave=True) )
        return results

    def get_image_mask(self, metadata, i = 0):
        h5_file = h5py.File(metadata.loc[i,"file"], "r")      
        image = h5_file.get("image")[()]*1.
        try:
            mask  = h5_file.get("mask")[()]
        except TypeError:
            mask = segment_all_channels(image)
        h5_file.close()
        return image, mask