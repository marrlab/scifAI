import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import  SelectFromModel , SelectKBest
from sklearn.feature_selection import mutual_info_classif
from scipy.cluster import hierarchy
from collections import defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

class AutoFeatureSelection(BaseEstimator, TransformerMixin):
    """
    Auto feature selection transformer
    """
    def __init__(self, top_k = 20, correlation_method = "spearman",correlation_threshold = 0.95, distance_threshold = 2., verbose =False):
        self.top_k = top_k 
        self.correlation_method = correlation_method
        self.correlation_threshold = correlation_threshold
        self.distance_threshold = distance_threshold
        self.verbose = verbose
        self.selected_features = []
    
    def fit(self, X,y):
        X_ = X.copy()
        
        if self.verbose:
            print("Step 1: Find highly correlated features")
        cor_matrix = pd.DataFrame(X_).corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

        correlation_support = []
        for col in upper_tri.columns:
            if any(upper_tri[col] < self.correlation_threshold):
                correlation_support.append(True)
            else:
                correlation_support.append(False)
        correlation_support = np.array(correlation_support)


        if self.verbose: 
            print("Step 2: wrapper methods")

        if self.verbose:
            print('Calculating mutual information')

        selector0 = SelectKBest(    mutual_info_classif,
                                        k = self.top_k)
        selector0 = selector0.fit(X_, y)

        if self.verbose:
            print('Calculating SVC')

        selector1 = SelectFromModel(    SVC(kernel="linear"),
                                        max_features = self.top_k)
        selector1 = selector1.fit(X_, y)

        if self.verbose:
            print('Calculating random forest')

        selector2 = SelectFromModel(    RandomForestClassifier(),  
                                        max_features = self.top_k)
        selector2 = selector2.fit(X_, y) 

        if self.verbose:
            print('Calculating l1 logistic regression')

        selector3 = SelectFromModel(    LogisticRegression(penalty="l1", 
                                            solver="liblinear"),
                                        max_features = self.top_k)
        selector3 = selector3.fit(X_, y) 

        if self.verbose:
            print('Calculating l2 logistic regression')

        selector4 = SelectFromModel(    LogisticRegression(penalty="l2"),
                                        max_features = self.top_k)
        selector4 = selector4.fit(X_, y)

        
        if self.verbose:
            print('Calculating xgb')
        
        selector5 = SelectFromModel(    XGBClassifier(n_jobs = -1, 
                                            eval_metric='logloss'),
                                        max_features = self.top_k)
        selector5 = selector5.fit(X_, y)


        importances = [ selector0.get_support(),
                        selector1.get_support(), 
                        selector2.get_support(), 
                        selector3.get_support(),  
                        selector4.get_support(),
                        selector5.get_support() ]

        intersection_of_features = correlation_support
        for _, im in enumerate(importances):
            intersection_of_features = intersection_of_features*im

        if self.verbose:
            print("number of similar features among all the methods:",intersection_of_features.sum())

        for _, im in enumerate(importances):
            self.selected_features =    self.selected_features + \
                                        np.where(im*correlation_support)[0].tolist()
            self.selected_features = sorted(list(set(self.selected_features)))
        
        if self.verbose:
            print(  "From", 
                X.shape[1] ,
                "initial features Selected (multicolinear):", 
                len(self.selected_features))

        if self.verbose:
            print("Step 3: clustering over correlation of features")
        
        corr_spearman = pd.DataFrame(X_[:,self.selected_features]).corr(self.correlation_method).to_numpy()
        corr_spearman = 1 - np.multiply(corr_spearman,corr_spearman) 
        corr_spearman = hierarchy.ward(corr_spearman)    

        cluster_ids = hierarchy.fcluster(corr_spearman,
                                        self.top_k, 
                                        criterion='maxclust')

        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features_spearman = [v[0] for v in cluster_id_to_feature_ids.values()]

        self.selected_features = np.array(self.selected_features)[selected_features_spearman]
        if self.verbose:
            print(  "From", 
                X.shape[1] ,
                "initial features Selected (uncorrelated):", 
                len(self.selected_features))
        return self
    
    def transform(self,X):
        X_ = X.copy()
        X_ = X_[:,self.selected_features]
        return X_