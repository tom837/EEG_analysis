import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import mne
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import undersample as ud


def undersample_data(X, y):
    """
    Undersample each class to have the same number of samples as the minority class.

    Parameters:
        X (numpy array): Feature data of shape (n_samples, n_features).
        y (numpy array): Labels of shape (n_samples,).

    Returns:
        X_balanced, y_balanced: Undersampled feature data and labels.
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_samples = min(class_counts)

    X_balanced = []
    y_balanced = []

    for cls in unique_classes:
        # Indices of all samples in the class
        indices = np.where(y == cls)[0]
        # Randomly select min_samples from the class
        selected_indices = np.random.choice(indices, min_samples, replace=False)
        X_balanced.append(X[selected_indices])
        y_balanced.append(y[selected_indices])

    return np.vstack(X_balanced), np.concatenate(y_balanced)
mne.set_log_level('WARNING')
data=np.load('data_cleaned.npy')
labels=np.load('y_train_rt.npy')
data=np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
data_eval=np.load('data_cleaned_eval.npy')
labels_eval=np.load('y_train_rt_eval.npy')
data=data.transpose(0,1,3,2)
data_eval=data_eval.transpose(0,1,3,2)
#data = data.reshape(data.shape[0],data.shape[1],-1)
print(data.shape)
models=[]
for i in range(1):
    X_train,y_train = ud.undersample_data(data[i], labels[i])
    #X_train,y_train= ud.undersample_data(data[i], labels[i])
    X_test,y_test=ud.undersample_data(data_eval[i],labels_eval[i])
    n_samples, n_channels, n_timepoints = X_train.shape
    # FFT for each channel

    X_train=np.abs(np.fft.rfft(X_train,axis=-1))
    X_train=X_train.reshape(X_train.shape[0],-1)
    print(X_train.shape)
    n_samples, n_channels, n_timepoints = X_test.shape
    # FFT for each channel

    X_test=np.abs(np.fft.rfft(X_test,axis=-1))
    X_test=X_test.reshape(X_test.shape[0],-1)
    print(X_test.shape)
    tot=[]
    for j in range(10):
        c=(j*10)+1.0
        #model = make_pipeline(StandardScaler(), SVC(kernel='sigmoid', C=c))
        model = SVC(kernel='rbf', C=0.001)  # 'linear', 'rbf', 'poly', 'sigmoid' are common kernels
        model.fit(X_train, y_train)
        models.append(model)
        predicted_labels = model.predict(X_test)
        # Get the label of the class with the highest probabilityindex
        #predicted_labels = np.argmax(predictions, axis=1)
        # Count how many correct labels the model got
        # Calculate accuracy
        n_correct=0
        for i in range(len(predicted_labels)):
            if predicted_labels[i]==y_test[i]:
                n_correct+=1
        n_cor=0
        conf_matrix = confusion_matrix(y_test, predicted_labels)
        # Get the label of the class with the highest probabilityindex
        predicted = model.predict(X_train)
        for i in range(len(predicted)):
            if predicted[i]==y_train[i]:
                n_cor+=1
        print(n_correct/len(y_test), n_cor/len(y_train), 'c being:',c)
        tot.append(n_correct/len(y_test))
        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=True)

        # Customize the plot
        plt.title("Prediction-Accuracy Table: Type")
        plt.xlabel("Predicted")
        plt.ylabel("Observed")
        plt.show()
    print(tot)
