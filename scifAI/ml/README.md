# Biologically driven features

In this folder, classes and methods are maintained that can be used for feature extraction from the fluorescent images and their corresponding masks. 

The files include:

-   `intensity_correction`: includes a `sklearn.transform` for pixel intensity transform
-   `segmentation`: provies two `sklearn.transform` for brightfield and fluorescent cell segmentation
-   `features`: includes multiple feature classes which can be used with a `sklearn.pipeline.FeatureUnion`
-   `feature_extractor`: includes a class to read the `metadata` and create the features
-   `auto_feature_selection`: a feature selection `sklearn.transform` which can be used before a classifier in a `sklearn.pipeline.Pipeline` before the estimator.