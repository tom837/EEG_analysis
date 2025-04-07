import numpy as np
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from lib.classifier import NaiveBayesClassifier
import undersample as ud


data=np.load('data_cleaned.npy')
labels=np.load('y_train_rt.npy')
data_eval=np.load('data_cleaned_eval.npy')
labels_eval=np.load('y_train_rt_eval.npy')


for i in range(len(data)):
    X_train,y_train =ud.undersample_data(data[i],labels[i])
    X_test,y_test=ud.undersample_data(data_eval[i],labels_eval[i])
nbmodel = NaiveBayesClassifier(bandwidth=1,kernel='radial')
nbmodel.fit(X_train, y_train)
print("Performance of radial kernel with bandwidth '1' for Test set: %.4f"%nbmodel.score(X_test,y_test))