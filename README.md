# scifAI: An explainable AI python framework for the analysis of multi-channel imaging flow cytometry data

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7bdb3a3cb3a44cfda5e7bd0d7b2096de)](https://app.codacy.com/gh/marrlab/scifAI?utm_source=github.com&utm_medium=referral&utm_content=marrlab/scifAI&utm_campaign=Badge_Grade_Settings)

Here, we present `scifAI`, a machine learning framework for the efficient and explainable analysis of high-throughput imaging data based on a modular open-source implementation. The open-source framework was developed in python, leveraging functionality from state-of-the-art modules, such as scikit-learn, SciPy, NumPy and pandas, allowing for smooth integration and extension of existing analysis pipelines. Universally applicable for single-cell imaging projects, the framework provides functionality for import and preprocessing of input data, several feature engineering pipelines including the implementation of a set of biologically motivated features and autoencoder-generated features, as well as methodology for efficient and meaningful feature selection. Moreover, the framework implements several machine learning and deep learning models for training supervised image classification models, e.g. for the prediction of cell configurations such as the immunological synapse. Following the principle of multi-instance learning, the framework also implements functionality to regress a set of selected images, against a downstream continuous readout such as cytokine production. Extensive documentation, as well as example code in the form of Jupyter notebooks is provided.

Note: this repository only includes the main python package and multiple jupyter notebooks on publicly available datasets. For following on how this package is applied in the main publication, please refer to https://github.com/marrlab/scifAI-notebooks 

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

# How to cite this work

coming soon

