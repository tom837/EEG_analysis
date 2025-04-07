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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,LSTM, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers 
import tensorflow as tf

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

def prediction(input,model_binary,model_task):
    output=model_binary.predict(input)
    indices=np.where(output!=0)[0]
    output[indices]=model_task.predict(input[indices])
    return output

def binary_reshaping(labels):
    labels[np.where(labels!=0)[0]]=1
    return labels

for i in range(len(data)):
    scores = []
    deb=data[i]
    X_train,y_train= undersample_data(data[i], labels[i])
    new_labels=binary_reshaping(labels[i])
    X_binary, y_binary=undersample_data(data[i],new_labels)
    #n_samples, n_channels, n_timepoints = X_train.shape
    # FFT for each channel
    #X_train=np.abs(np.fft.rfft(X_train,axis=-1))
    #X_train = X_train.reshape(n_samples, -1)
    n=15
    pca=PCA(n_components=n)
    tmp=[]
    for sample in X_train:
        tmp.append(pca.fit_transform(sample))
    X_train=np.array(tmp)
    tmp=[]
    for sample in X_binary:
        tmp.append(pca.fit_transform(sample))
    X_binary=np.array(tmp)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train=np.delete(X_train,np.where(y_train==0)[0],axis=0)
    y_train=np.delete(y_train,np.where(y_train==0)[0])
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(X_train)
    y_binary = to_categorical(y_binary, num_classes=5)
    #X_train, X_test, y_train, y_test = train_test_split(X_train, labels[i], test_size=0.2, random_state=42)
    # Assemble a classifier
    lda_t = LinearDiscriminantAnalysis()
    csp_t = CSP(n_components=50 , reg=None, log=True, norm_trace=False)
    # Use scikit-learn Pipeline with cross_val_score function
    clf_task = Pipeline([("CSP", csp_t),("LDA", lda_t)])
    scores_task = cross_val_score(clf_task, X_train, y_train, cv=cv, n_jobs=None)
    # Printing the results
    # Printing the results
    """class_balance = np.mean(y_train == y_train[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")"""
    model = Sequential()
    # 1D Convolutional Layer (for spatial feature extraction across channels)
    model.add(Input(shape=((X_train.shape[1],X_train.shape[2]))))
    model.add(Conv1D(filters=12, kernel_size=5, activation='relu',kernel_regularizer=regularizers.l2(0.1)))
    model.add(BatchNormalization())


    # Adding these layers make the model better but increase overfiting
    model.add(Conv1D(filters=6, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(0.1)))
    model.add(BatchNormalization())
    # Max pooling
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    # Flatten the output
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.1)))
    #model.add(Dropout(0.5))


    # Output layer (multiclass classification for each task and no task)
    model.add(Dense(5, activation='softmax'))

    # Compile the model
    loss=tf.keras.losses.CategoricalCrossentropy()

    acc='categorical_crossentropy'
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,use_ema=True,ema_momentum=0.9)

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    # plot CSP patterns estimated on full data for visualization
    #X_binary,X_t,y_binary,y_t=train_test_split(X_binary,y_binary,test_size=0.2, random_state=42)
    csp_t.fit_transform(X_train, y_train)
    clf_task.fit_transform(X_train,y_train)
    predicted_labels=prediction(X_test,clf_binary,clf_task)
    n_correct=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==y_test[i]:
            #print(predicted_labels[i],y_test[i])
            n_correct+=1
    conf_matrix = confusion_matrix(y_test, predicted_labels)
    # Print the ratio of correctness of the model in the evaluation dataset
    print("ratio correctness: ", n_correct/len(predicted_labels))
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