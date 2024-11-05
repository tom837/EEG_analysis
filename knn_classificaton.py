from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.decomposition import PCA

X_train=np.load("X_train_rtfeatures.npy")
labels=np.load("y_train_rtfeatures.npy")
# Uc stands for uncleaned (doubles the dataset)
"""X_uc=np.load("X_train_rta_uc.npy")
y_uc=np.load("y_train_rta_uc.npy")"""
X_test=np.load("X_test_rtfeatures.npy")
y_test=np.load("y_test_rtfeatures.npy")
train=np.load("X_train_rtfeatures.npy")
test=np.load("X_test_rtfeatures.npy")
n=15
pca=PCA(n_components=n)
tmp=[]
for sample in train:
    tmp.append(pca.fit_transform(sample))
X_train=np.array(tmp)
tmp=[]
for sample in test:
    tmp.append(pca.fit_transform(sample))
X_test=np.array(tmp)

print(X_train.shape)
print(X_test.shape)
print(X_train.shape)
X_train=X_train.reshape(-1,22*n)
X_test=X_test.reshape(-1,22*n)
print(X_train.shape)


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