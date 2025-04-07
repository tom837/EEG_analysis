# Creating and training model for real time classification
# It is still a work in progress and tweaks to the model and training still need to be done
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,LSTM, Input,GRU
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Attention,Reshape
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
<<<<<<< Updated upstream
from sklearn.decomposition import PCA
import wavelet_preprocessing
import ica_preprocessing
=======
>>>>>>> Stashed changes
import mne
from scipy.signal.windows import hann
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
import undersample as ud
from sklearn.decomposition import PCA
import pywt


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

# Load the training data (see rt_dataset to see how it was created)
mne.set_log_level('WARNING')
data=np.load('data_cleaned.npy')
labels=np.load('y_train_rt.npy')
data_eval=np.load('data_cleaned_eval.npy')
labels_eval=np.load('y_train_rt_eval.npy')
data=data.transpose(0,1,3,2)
data_eval=data_eval.transpose(0,1,3,2)
#print(train_ftr.shape)
# Transpose the the matrix to fit the input shape (it trains better like this)


# Using pca to make the data less complexe
"""tmp=[]
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
test_ftr=np.array(tmp)"""

"""X_test=np.concatenate([X_test,test_ftr],axis=2)
X_train=np.concatenate([X_train,train_ftr],axis=2)"""
# Encode the label in one hot encoding
#data = data.reshape(data.shape[0],data.shape[1],-1)
models=[]
for i in range(1):
<<<<<<< Updated upstream
    X_train,y_train= undersample_data(data[i], labels[i])
=======
    X_train,y_train= ud.undersample_data(data[i], labels[i])
    X_test,y_test=ud.undersample_data(data_eval[i],labels_eval[i])
    #print(X_test.shape,y_test.shape)
>>>>>>> Stashed changes
    features = []
    """for i in range(len(X_train)):
        hann_window = hann(250)
        X_train[i] = X_train[i] * hann_window[:, np.newaxis]
    X_train=X_train.transpose(0,2,1)
    print(X_train.shape)"""
    n_samples, n_channels, n_timepoints = X_train.shape
    # FFT for each channel

    """X_train=np.abs(np.fft.rfft(X_train,axis=-1))
    #X_train=X_train.reshape(X_train.shape[0],-1)
    print(X_train.shape)
    print(y_train.shape,y_val.shape)
    n_samples, n_channels, n_timepoints = X_test.shape
    # FFT for each channel

    X_test=np.abs(np.fft.rfft(X_test,axis=-1))"""
    X_tmp=[]
    print(X_train.shape)
    for k in range(len(X_train)):
        X_tmp.append([])
        for f in range(len(X_train[k])):
            X_tmp[k].append(pywt.dwt(X_train[k][f],'db1'))
    X_train=np.array(X_tmp)
    X_tmp=[]
    for k in range(len(X_test)):
        X_tmp.append([])
        for f in range(len(X_test[k])):
            X_tmp[k].append(pywt.dwt(X_test[k][f],'db1'))
    print(X_train.shape)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],-1)
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],-1)
    print(X_train.shape)
    n=15
    pca=PCA(n_components=n)
    tmp=[]
    for sample in X_train:
        tmp.append(pca.fit_transform(sample))
    X_train=np.array(tmp)
    y_train=to_categorical(y_train)
    X_train,X_val,y_train,y_val=train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    pca=PCA(n_components=n)
    tmp=[]
    for sample in X_test:
        tmp.append(pca.fit_transform(sample))
    X_test=np.array(tmp)
    #X_test=X_test.reshape(X_test.shape[0],-1)
    print(X_train.shape)
    #features=np.abs(features)
    # Create the CNN model
    model = Sequential()
    # 1D Convolutional Layer (for spatial feature extraction across channels)
    model.add(Input(shape=(X_train.shape[1],X_train.shape[2])))
    model.add(Conv1D(filters=12, kernel_size=5, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())


    # Adding these layers make the model better but increase overfiting
    model.add(Conv1D(filters=64, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=3,activation='relu'))
    model.add(BatchNormalization())

    # Max pooling
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    # Flatten the output
    model.add(Flatten())

    # Fully connected layer
    
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(LayerNormalization())
    model.add(Reshape((32,2)))
    model.add(GRU(units=32))
    #model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))


    # Output layer (multiclass classification for each task and no task)
    model.add(Dense(5, activation='softmax',kernel_regularizer=regularizers.l2(0.001)))

    # Compile the model
    loss=tf.keras.losses.CategoricalCrossentropy()

    acc='categorical_crossentropy'
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001,use_ema=True,ema_momentum=0.9)

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    # Perform K-fold cross-validation
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    # Testing this solution to avoid overfiting but it doesn't seem to help
    # K fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # Using learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
    """for train_index, test_index in kf.split(X_train):
        X, X_t = X_train[train_index], X_train[test_index]
        y, y_t = y_train[train_index], y_train[test_index]
        # Train the model
        history = model.fit(X, y, epochs=200, batch_size=64, validation_data=(X_t, y_t),callbacks=early_stopping)
        # Plot accuracy and loss curves
        plt.plot(history.history['accuracy'], label='train accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.legend()
        plt.show()
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()
        score = model.evaluate(X_t, y_t)
        print(f"Fold accuracy: {score[1]}")"""
    history=model.fit(X_train, y_train, epochs=5000,validation_data=(X_val, y_val), batch_size=64,callbacks=early_stopping)
    # Ploting the history of the training of the model
    plt.plot(history.history[acc], label='train accuracy')
    plt.plot(history.history['val_'+acc], label='val accuracy')
    plt.legend()
    plt.show()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    
    """score = model.evaluate(X_test, y_test)
    print(f"Fold accuracy: {score[1]}")"""
    # Summary of the model
    model.summary()
    # Save the model
    model.save('3rd_rt_cnn.keras')
    # Evaluating the model
    predictions = model.predict(X_test)
    #y_test=np.argmax(y_test,axis=1)
    # Get the label of the class with the highest probabilityindex
    predicted_labels = np.argmax(predictions, axis=1)
    # Count how many correct labels the model got
    n_correct=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==y_test[i]:
            #print(predicted_labels[i],y_test[i])
            n_correct+=1
    # Print the ratio of correctness of the model in the evaluation dataset
    print("ratio correctness: ", n_correct/len(predicted_labels))
    kappa = cohen_kappa_score(y_test,predicted_labels)
    print("Kappa value:", kappa)
    conf_matrix = confusion_matrix(y_test, predicted_labels)
    predictions = model.predict(X_train)
    # Get the label of the class with the highest probabilityindex
    predicted_labels = np.argmax(predictions, axis=1)
    y_train=np.argmax(y_train,axis=1)
    # Count how many correct labels the model got
    n_correct=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==y_train[i]:
            #print(predicted_labels[i],y_test[i])
            n_correct+=1
    # Print the ratio of correctness of the model in the evaluation dataset

    # Compute kappa value
    kappa = cohen_kappa_score(y_train,predicted_labels)
    print("Kappa value:", kappa)
    print("ratio correctness trained: ", n_correct/len(predicted_labels))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=True)

    # Customize the plot
    plt.title("Prediction-Accuracy Table: Type")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.show()
    
    
"""ratio correctness:  0.3163934426229508
Kappa value: 0.1454918032786885"""