# Creating and training model for real time classification
# It is still a work in progress and tweaks to the model and training still need to be done
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA
import wavelet_preprocessing

# Load the training data (see rt_dataset to see how it was created)
X_train=np.load("X_train.npy")
labels=np.load("y_train.npy")
X_test=np.load("X_test.npy")
y_test=np.load("y_test.npy")
# Extracting features from the data
train_ftr=wavelet_preprocessing.dwt_feature_extraction(X_train)
test_ftr=wavelet_preprocessing.dwt_feature_extraction(X_test)

# Transpose the the matrix to fit the input shape (it trains better like this)
X_test=X_test.transpose(0,2,1)
X_train=X_train.transpose(0, 2, 1) # Shape: (n_samples, n_channels, n_timepoints)

# Using pca to make the data less complexe
pca=PCA(n_components=15)
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

X_test=np.concatenate([X_test,test_ftr],axis=2)
X_train=np.concatenate([X_train,train_ftr],axis=2)
X_train, X_t, labels, y_t = train_test_split(X_train, labels, test_size=0.2, random_state=42)
# Encode the label in one hot encoding
labels = to_categorical(labels, num_classes=5)
y_t=to_categorical(y_t, num_classes=5)



# Create the CNN model
model = Sequential()
# 1D Convolutional Layer (for spatial feature extraction across channels)
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(22,30),kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
"""
# Adding these layers make the model better but increase overfiting
model.add(Conv1D(filters=32, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
"""
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Perform K-fold cross-validation
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
"""
# Testing this solution to avoid overfiting but it doesn't seem to help
# K fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Using learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
for train_index, test_index in kf.split(X_train):
    X, X_t = X_train[train_index], X_train[test_index]
    y, y_t = labels[train_index], labels[test_index]
    # Train the model
    history = model.fit(X, y, epochs=200, batch_size=64, validation_data=(X_t, y_t), callbacks=[lr_scheduler,early_stopping])
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

history=model.fit(X_train, labels, epochs=100, batch_size=32,validation_data=(X_t,y_t),callbacks=[early_stopping])
# Ploting the history of the training of the model
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
score = model.evaluate(X_t, y_t)
print(f"Fold accuracy: {score[1]}")
# Summary of the model
model.summary()
# Save the model
model.save('3rd_rt_cnn.keras')
# Evaluating the model
predictions = model.predict(X_test)
# Get the label of the class with the highest probabilityindex
predicted_labels = np.argmax(predictions, axis=1)
# Count how many correct labels the model got
n_correct=0
for i in range(len(predicted_labels)):
    if predicted_labels[i]==y_test[i]:
        n_correct+=1
# Print the ratio of correctness of the model in the evaluation dataset
print("ratio correctness: ", n_correct/len(predicted_labels))

# Compute kappa value
kappa = cohen_kappa_score(y_test,predicted_labels)
print("Kappa value:", kappa)