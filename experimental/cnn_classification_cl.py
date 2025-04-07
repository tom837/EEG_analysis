# Batch classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# load the training and evaluation datasets (see cl_dataset for more info)
X_train=np.load("X_train_cl.npy")
y_train=np.load("y_train_cl.npy")
X_test=np.load("X_test_cl.npy")
y_test=np.load("y_test_cl.npy")
# Convert the labels to one hot encoding
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=5)

# Create the model
model = Sequential()
# 1D Convolutional Layer (for spatial feature extraction across channels)
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(22, 750)))
model.add(MaxPooling1D(pool_size=2))
# Flatten the output
model.add(Flatten())
# Fully connected layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Output layer (binary classification)
model.add(Dense(5, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=70)
# Summary of the model
model.summary()
model.save('3rd_cnn.keras')



# Predict the values of evaluation data
predictions = model.predict(X_test)
# Get the index of the class with the highest probability
predicted_labels = np.argmax(predictions, axis=1)
# Calculate the ratio of correct classification
n_correct=0
# Print the predicted class labels
for i in range(len(predicted_labels)):
    if predicted_labels[i]==y_test[i]:
        n_correct+=1
print("ratio correctness: ", n_correct/len(y_test))