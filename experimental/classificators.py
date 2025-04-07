import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import feature_selectors as fs
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


def cnn(data):
    model = Sequential()
    # 1D Convolutional Layer (for spatial feature extraction across channels)
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(data.shape[1],1)))
    model.add(BatchNormalization())
    # Adding these layers make the model better but increase overfiting
    model.add(Conv1D(filters=32, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=3,activation='relu'))
    model.add(BatchNormalization())

    # Max pooling
    model.add(MaxPooling1D(pool_size=2))

    # Flatten the output
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))


    # Output layer (multiclass classification for each task and no task)
    model.add(Dense(5, activation='softmax'))

    # Compile the model
    loss=tf.keras.losses.Huber()
    model.compile(optimizer='adamw', loss=loss, metrics=['accuracy'])
    return model


def svm(data):
    model = SVC(kernel='linear', C=1.0)  # 'linear', 'rbf', 'poly', 'sigmoid' are common kernels
    return model

def knn(data):
    model = KNeighborsClassifier(n_neighbors=5)
    return model

def lda(data):
    model=LinearDiscriminantAnalysis()
    return model


def rf(data):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def dnn(data):
    model = Sequential([
    Dense(128, activation='relu', input_shape=(data.shape[1],1)),  # Input layer with 128 neurons
    Dropout(0.3),  # Dropout for regularization
    Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    Dropout(0.3),
    Flatten(),
    Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    Dense(5, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer="adamw", loss='binary_crossentropy', metrics=['accuracy'])
    return model


def lstm(data):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(data.shape[1],1)),  # LSTM with 50 units
        Dropout(0.3),  # Dropout for regularization
        LSTM(50),  # Another LSTM layer
        
        Dense(5, activation='softmax')  # Output layer for binary classification; use 'softmax' for multi-class
    ])

    # Compile the model
    model.compile(optimizer='adamw', loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
    
if __name__ == "__main__":
    X_train=np.load("X_train_rt_ftr.npy")
    X_test=np.load("X_test_rt_ftr.npy")
    y_train=np.load("y_train_rt.npy")
    y_test=np.load("y_test_rt.npy")
    X_train=fs.pca(X_train,y_train)
    X_test=fs.pca(X_test,y_test)
    """X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_train=to_categorical(y_train, num_classes=5)"""

    model=knn(X_train)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train)
    predicted_labels = model.predict(X_test)
    # Get the label of the class with the highest probabilityindex
    #predicted_labels = np.argmax(predictions, axis=1)
    # Count how many correct labels the model got
    print(predicted_labels)
    n_correct=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==y_test[i]:
            n_correct+=1
    # Print the ratio of correctness of the model in the evaluation dataset
    print("ratio correctness: ", n_correct/len(predicted_labels))