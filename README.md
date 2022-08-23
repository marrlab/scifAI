# scifAI: Explainable machine learning for profiling the immunological synapse and functional characterization of therapeutic antibodies

Here, we present `scifAI`, a machine learning framework for the efficient and explainable analysis of high-throughput imaging data based on a modular open-source implementation. We also publish the largest publicly available multi-channel IFC data set with over 2.8 million images of primary human T-B cell conjugates from multiple donors, and demonstrate how scifAI can be used to detect patterns and build predictive models. We showcase the potential of our framework for (i) the prediction of immunologically relevant cell class frequencies, (ii) the systematic morphological profiling of the immunological synapse, (iii) the investigation of inter donor and inter and intra-experiment variability, as well as (iv) the characterization of the mode of action of therapeutic antibodies and (v) the prediction of their functionality in vitro. Combining high-throughput imaging of the immunological synapse using IFC with rigorous data preprocessing and machine learning enables researchers in pharma to screen for novel antibody candidates and improved evaluation of lead molecules in terms of functionality, mode-of-action insights and antibody characteristics such as affinity, avidity and format.

Note: this repository only includes the main python package and multiple jupyter notebooks on publicly available datasets. For following on how this package is used in the main publication, please refer to https://github.com/marrlab/scifAI-notebooks 

## Data structure

For using the package, you need to download the data from IDEAS software and save each image (all the channels) as an `.h5` file. The `.h5` file should include at least these keys: `image`, `mask`. In case there is label available, also the `label` should be provided as `str`.

In addition, each file should be saved with the object number as the last part in the name. For example, for a random image with object number of 1000 this is the correct name: `random_file_1000.h5`. This is important as you can use the object numbers to come back to files and use the IDEAS software as well.

Apart from each file, we assume that the data comes from different experiments, donors and conditions. For example, in case we have N experiments, M donors and K conditions, the data path folder should look like this:

```bash
data_path/Experiment_1/Donor_1/condition_1/*.h5
data_path/Experiment_1/Donor_1/condition_2/*.h5
.
.
.
data_path/Experiment_1/Donor_2/condition_1/*.h5
data_path/Experiment_1/Donor_2/condition_2/*.h5
.
.
.
data_path/Experiment_N/Donor_M/condition_K/*.h5
```

## How to install the package

For installing the package, you can simply clone the repository and run the following command:

```bash
pip -q install <PATH TO THE FOLDER>
```

## How to use the package

For the feature extraction, you first need to calcalate the `metadata` dataframe with providing the correct data path. 

```python
import scifAI

data_path = <PATH TO THE DATA FOLDER>
metadata = scifAI.metadata_generator(data_path)
```

After that, you need to defined the feature union from `sklearn` based on the desired features. For example:

```python
from sklearn.pipeline import  FeatureUnion
from scifAI.ml import features

feature_union = FeatureUnion([
                                ("MaskBasedFeatures", features.MaskBasedFeatures()), 
                                ("GLCMFeatures", features.GLCMFeatures()),  
                                ("GradientRMS", features.GradientRMS()),  
                                ("BackgroundMean", features.BackgroundMean()), 
                                ("PercentileFeatures", features.PercentileFeatures()), 
                                ("CellShape", features.CellShape()),  
                                ("Collocalization", features.Collocalization()),    
                                ("IntersectionProperties", features.IntersectionProperties()),
                                ("CenterOfCellsDistances", features.CenterOfCellsDistances())
]
)
```

Finally you can pass the feature union to the `FeatureExtractor` as a `sklearn` pipeline:

```python
from sklearn.pipeline import Pipeline
from scifAI.ml import FeatureExtractor 

pipeline = Pipeline([("features", feature_union)])

feature_extractor = FeatureExtractor(pipeline)
list_of_features = feature_extractor.extract_features(metadata)
```
The output of `extract_features` would be a list, where each element is a dictionary of features for every row in the `metadata`. Finally, you can transoform the `list_of_features` to a DataFrame by simply running:

```python
df_features = pd.DataFrame(list_of_features)
```

where every row in the `df_features` contains the corresponding features from the same row in `metadata`. 

Considering that there are many features, we suggest to reduce the features with no variance. In addition, imputing with `0.` is the best option as it follows the biological assumptions for the feature extraction process.

```python
df_features = df_features.fillna(0.)
df_features = df_features.loc[:, df_features.std() > 0.]
```

For different examples, you can follow our examples in the [docs](docs) folder.


