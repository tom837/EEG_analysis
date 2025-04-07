from sklearn.decomposition import PCA
import numpy as np
from skrebate import ReliefF
from sklearn.model_selection import train_test_split
import pandas as pd
import pymrmr
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR


def rfe(features,labels):
    # Initialize the model
    features=features.reshape(len(features), -1)
    model = SVR(kernel="linear")
    # Initialize RFE with the model and the number of features to select
    rfe = RFE(estimator=model, n_features_to_select=15, step=1)
    # Fit RFE to the training data
    rfe.fit(features[400:500], labels[400:500])
    print("done fitting")
    selected_features = np.where(rfe.support_)[0]  # Get the indices of selected features
    # Transform the data to keep only the selected features
    X_train_selected = rfe.transform(features)
    return np.array(X_train_selected)

def mrmr(features,labels):
    # Generate a sample dataset
    # Convert the dataset to a DataFrame
    features=features.reshape(len(features), -1)
    df = pd.DataFrame(features[400:500], columns=[f'feature_{i}' for i in range(features[400:500].shape[1])])
    df['target'] = labels[400:500]  # Add the target column

    # Apply mRMR feature selection
    selected_features = pymrmr.mRMR(df, 'MIQ', 15)  # 'MIQ' is a strategy, 15 is the number of features to select
    selected_indices = [df.columns.get_loc(feature) for feature in selected_features if feature in df.columns]
    return np.array(features[:,selected_indices])

def relief(features,labels):
    # Generate a sample dataset
    # Split the data into training and testing sets
    # Initialize the ReliefF feature selection model
    features=features.reshape(len(features), -1)
    relief = ReliefF(n_neighbors=15)  # Adjust the number of neighbors as needed
    # Fit the model to the training data
    relief.fit(features[400:500], labels[400:500])
    # Get feature scores
    feature_scores = relief.feature_importances_
    # Select features based on a threshold or rank
    selected_features = np.argsort(feature_scores)[::-1][:15]  # Top 15 features
    return features[:,selected_features]


def pca(features,labels=None, n=15):
    features=features.reshape(len(features), -1)
    pca=PCA(n_components=n)
    return np.array(pca.fit_transform(features))

if __name__ == "__main__":
    X_train=np.load("X_train_rt_ftr.npy")
    X_test=np.load("X_test_rt_ftr.npy")
    y_train=np.load("y_train_rt.npy")
    y_test=np.load("y_test_rt.npy")
    train=pca(X_train,y_train)
    test=pca(X_test,y_test)
    print(train.shape,X_train.shape)
    print(test.shape,X_test.shape)
