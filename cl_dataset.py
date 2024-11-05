import numpy as np
import data_extraction
import wavelet_preprocessing
from random import randint


# Initialising the training and evaluation lists
X_train=[]
y_train=[]
X_test=[]
y_test=[]

# Mapping the labels of the tasks
label_mapping = {
        'left': 0,
        'right': 1,
        'feet': 2,
        'tongue': 3,
        'fixation': 4
    }

# Go through every user in the dataset (note i am only using 6 as oposed to 9 because the last 3 got corrupt but I will fix this)
for i in range(6):
    file=f"BCICIV_2a_gdf/A0{i+1}T.npz"
    data=data = np.load(file)
    signal = data['s']
    # Get list of 3sec intervals for each task
    left,right,feet,tongue,fixation= data_extraction.get_data(file,signal)
    eeg_data={}
    eeg_data["left"]=left
    eeg_data["right"]=right
    eeg_data["feet"]=feet
    eeg_data["tongue"]=tongue
    eeg_data["fixation"]=fixation[:len(left)]
    # Clean the data
    eeg_cleaned_data=wavelet_preprocessing.wavelet(eeg_data)
    # Flatten the dictionary into X_train and y_train
    for task in eeg_cleaned_data:
        # Convert the task's data into a NumPy array directly (skip inner loops)
        task_data = np.array(eeg_cleaned_data[task])  # Convert to NumPy array
        X_train.append(task_data)  # Append the entire task data to X_train
        # Create a label array for the task, one label for each sample
        task_labels = np.full((task_data.shape[0],), label_mapping[task])
        y_train.append(task_labels)

# Convert the lists to NumPy arrays for better performance
X_train = np.vstack(X_train)  # Shape: (n_samples, n_channels, n_timepoints)
y_train = np.concatenate(y_train)
# Get 150 random elements from the training data to make evaluation data
for i in range(150):
    num=randint(0,len(X_train))
    X_test.append(X_train[num])
    y_test.append(y_train[num])
    np.delete(X_train,num)
    np.delete(y_train,num)
np.save('X_train_cl.npy',X_train)
np.save('y_train_cl.npy',y_train)
np.save('X_test_cl.npy',X_test)
np.save('y_test_cl.npy',y_test)