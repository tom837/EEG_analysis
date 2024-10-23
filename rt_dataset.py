import numpy as np
import data_extraction
import wavelet_preprocessing

# Creating the training and validation data for the real time classification

def create_windows(eeg_data, window_size=250,step_size=125):
    # Create window_size sample windows with step_size samples overlap between each window
    n_windows = (eeg_data.shape[0] - window_size) // step_size + 1
    # Create the overlapping windows
    overlapping_windows = np.array([eeg_data[i:i+window_size] for i in range(0, n_windows * step_size, step_size)])
    # Reshape each window to include the channels
    # This will result in shape (number of windows, window size, number of channels)
    overlapping_windows = overlapping_windows.reshape(n_windows, window_size, eeg_data.shape[1])
    return overlapping_windows

def reshape_positions(positions,window_size=250,step_size=125):
    # Create labels list for the windows of the data
    # Calculate the number of windows based on the step size
    # Minimum step_size is 1
    n_windows = (positions.shape[0] - window_size) // step_size + 1
    # Create overlapping windows for the labels (positions)
    position_windows = np.array([positions[i:i+window_size] for i in range(0, n_windows * step_size, step_size)])
    # Take the label from the middle of each window
    middle_index = window_size // 2
    reshaped_positions = position_windows[:, middle_index]
    return reshaped_positions

# The lists the data will go to
X_train=[]
y_train=[]
X_test=[]
y_test=[]
# Mapping the labels of the tasks
label_mapping = {
        'fixation': 0,
        'left': 1,
        'right': 2,
        'feet': 3,
        'tongue': 4
    }

# Go through all the test subjects (note I am only using 6 as oposed to 9 because the last 3 got corrupt but I will fix this)
for i in range(8):
    # Load each file
    file=f"BCICIV_2a_gdf/A0{i+1}T.npz"
    data=data = np.load(file)
    # Differentiate between training data (signal) and evaluation data(X) ~90/10
    # The evaluation data is going to be a continuous stream as oposed to the training data that is going to be sequential
    signal = data['s'][:-60000]
    X=data['s'][-60000:]
    # Getting only the needed seccions from the data
    left,right,feet,tongue,fixation= data_extraction.get_data(file,signal)
    eeg_data={}
    eeg_data["left"]=np.array(left).transpose(0, 2, 1).reshape(-1, 22)
    eeg_data["right"]=np.array(right).transpose(0, 2, 1).reshape(-1, 22)
    eeg_data["feet"]=np.array(feet).transpose(0, 2, 1).reshape(-1, 22)
    eeg_data["tongue"]=np.array(tongue).transpose(0, 2, 1).reshape(-1, 22)
    eeg_data["fixation"]=np.array(fixation).transpose(0, 2, 1).reshape(-1, 22)[:len(eeg_data["left"])]
    eeg_cleaned_data={}
    # Cleaning the data
    for task in eeg_data:
        eeg_cleaned_data[task]=wavelet_preprocessing.wavelet_rt(np.array([eeg_data[task].T])).T
    X=data_extraction.cleaning_eog_rt(X)
    X=wavelet_preprocessing.wavelet_rt(X)
    # Reshaping the data to match what we need it for
    X=np.squeeze(X, axis=0)
    X=X.T
    # Creating the labels for evaluation data
    positions=np.zeros(len(data['s']))
    positions= data_extraction.extract_data_rt(769,data['etyp'],positions,data['epos'],1)
    positions= data_extraction.extract_data_rt(770,data['etyp'],positions,data['epos'],2)
    positions= data_extraction.extract_data_rt(771,data['etyp'],positions,data['epos'],3)
    positions= data_extraction.extract_data_rt(772,data['etyp'],positions,data['epos'],4)
    positions=positions[-len(X):]
    X_test.append(X)
    y_test.append(positions)
    # Go through every task and every reading do add them to the training list
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
X_test=np.array(X_test)
y_test=np.array(y_test)
# Reshape the arrays to the correct shape
X_test=X_test.reshape(-1,22)
y_test=y_test.reshape(-1)
X_train=np.squeeze(X_train, axis=2)
# Create the 1sec windows for the data
X_train=create_windows(X_train, step_size=250) # Shape: (n_samples, n_timepoints, n_channels)
y_train=reshape_positions(y_train, step_size=250)
X_test=create_windows(X_test, step_size=250)
y_test=reshape_positions(y_test, step_size=250)
# Save the data
np.save('X_train_rt.npy',X_train)
np.save('y_train_rt.npy',y_train)
np.save('X_test_rt.npy',X_test)
np.save('y_test_rt.npy',y_test)