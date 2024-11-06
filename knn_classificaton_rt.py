from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.decomposition import PCA
import wavelet_preprocessing

# Load the training data (see rt_dataset to see how it was created)
X_train=np.load("X_train_rt.npy")
labels=np.load("y_train_rt.npy")
X_test=np.load("X_test_rt.npy")
y_test=np.load("y_test_rt.npy")
# Extracting features from the data
train_ftr=wavelet_preprocessing.dwt_feature_extraction(X_train)
test_ftr=wavelet_preprocessing.dwt_feature_extraction(X_test)
X_test=X_test.transpose(0,2,1)
X_train=X_train.transpose(0, 2, 1) # Shape: (n_samples, n_channels, n_timepoints)
n=15
# Using pca to make the data less complexe
pca=PCA(n_components=n)
tmp=[]
for sample in X_train:
    tmp.append(pca.fit_transform(sample))
X_train=np.array(tmp)
tmp=[]
for sample in X_test:
    tmp.append(pca.fit_transform(sample))
X_test=np.array(tmp)
tmp=[]
for sample in train_ftr:
    tmp.append(pca.fit_transform(sample))
train_ftr=np.array(tmp)
tmp=[]
for sample in test_ftr:
    tmp.append(pca.fit_transform(sample))
test_ftr=np.array(tmp)

# Combine the uncomplex training data and features
X_test=np.concatenate([X_test,test_ftr],axis=2)
X_train=np.concatenate([X_train,train_ftr],axis=2)
# Reshape the data
X_train=X_train.reshape(-1,22*2*n)
X_test=X_test.reshape(-1,22*2*n)


knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(X_train, labels)

predicted_labels = knn.predict(X_test)
# Count how many correct labels the model got
n_correct=0
for i in range(len(predicted_labels)):
    if predicted_labels[i]==y_test[i]:
        n_correct+=1
# Print the ratio of correctness of the model in the evaluation dataset
print("ratio correctness: ", n_correct/len(predicted_labels))