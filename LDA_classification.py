import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from mne import Epochs, pick_types
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data=np.load('data_cleaned.npy')
labels=np.load('y_train_rt.npy')
data=np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
data=data.transpose(0,1,3,2)

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

def reshaping(data):
    n_samples, n_channels, n_timepoints = data.shape
    return data.reshape(n_samples, -1)

for i in range(1):
    scores = []
    X_train,y_train= undersample_data(data[i], labels[i])
    n_samples, n_channels, n_timepoints = X_train.shape
    # FFT for each channel
    #X_train=np.abs(np.fft.rfft(X_train,axis=-1))
    #X_train = X_train.reshape(n_samples, -1)
    print(X_train.shape,y_train.shape)
    n=15
    pca=PCA(n_components=n)
    tmp=[]
    for sample in X_train:
        tmp.append(pca.fit_transform(sample))
    X_train=np.array(tmp)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(X_train)
    #X_train, X_test, y_train, y_test = train_test_split(X_train, labels[i], test_size=0.2, random_state=42)
    for i in range(5):
        print(len(np.where(y_test==i)[0]))
    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=50 , reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp),("LDA", lda)])
    scores = cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=None)
    # Printing the results
    # Printing the results
    """class_balance = np.mean(y_train == y_train[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")"""

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(X_train, y_train)
    clf.fit_transform(X_train,y_train)
    predicted_labels=clf.predict(X_test)
    n_correct=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==y_test[i]:
            #print(predicted_labels[i],y_test[i])
            n_correct+=1
    conf_matrix = confusion_matrix(y_test, predicted_labels)
    # Print the ratio of correctness of the model in the evaluation dataset
    print("ratio correctness: ", n_correct/len(predicted_labels))
    predicted_labels=clf.predict(X_train)
    n_correct=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==y_train[i]:
            #print(predicted_labels[i],y_test[i])
            n_correct+=1
    # Print the ratio of correctness of the model in the evaluation dataset
    print("ratio correctness trained: ", n_correct/len(predicted_labels))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=True)

    # Customize the plot
    plt.title("Prediction-Accuracy Table: Type")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.show()
    #csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
    """scores_windows = []
    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_tr, y_test = y_train[train_idx], y_train[test_idx]
        print("Shape of training data:", X_train[train_idx].shape)
        print("Shape of training labels:", y_tr.shape)
        print("Training data has NaN or Inf:", np.isnan(data[i][train_idx]).any() or np.isinf(data[i][train_idx]).any())
        print("Unique labels in y_train:", np.unique(y_train))
        X_tr = csp.fit_transform(X_train[train_idx], y_tr)
        if np.isnan(X_tr).any() or np.isinf(X_tr).any():
            raise ValueError("CSP output contains NaN or Inf values")
        X_test = csp.transform(X_train[test_idx])
        # fit classifier
        lda.fit(X_train, y_train)
        # running classifier: test classifier on sliding window
        score_this_window = []
        score=lda.score(X_test, y_test)
        score_this_window.append(score)
        print("final score:", score)
        scores_windows.append(score_this_window)
    print("step 5")
    # Plot scores over time
    #w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(250, np.mean(scores_windows, 0), label="Score")
    plt.axvline(0, linestyle="--", color="k", label="Onset")
    plt.axhline(0.5, linestyle="-", color="k", label="Chance")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()"""