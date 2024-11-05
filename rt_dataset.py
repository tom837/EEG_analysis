import numpy as np
import data_extraction
import wavelet_preprocessing
import time

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
whole=time.time()
for i in range(9):
    print(f"--- processing personne {i+1} ---")
    personne=time.time()
    # Load each file
    file=f"BCICIV_2a_gdf/A0{i+1}T.npz"
    data=data = np.load(file)
    # Differentiate between training data (signal) and evaluation data(X) ~90/10
    # The evaluation data is going to be a continuous stream as oposed to the training data that is going to be sequential
    if i+1 ==4:
        X=data['s'][750:]
    else:
        X=data['s'][60000:660000]
    # Cleaning the data
    X=create_windows(X,window_size=250, step_size=250)
    X=X.transpose(0, 2, 1)
    X=data_extraction.cleaning_eog(X)
    X=np.array(X).transpose(0, 2, 1)
    X=np.array(X).transpose(0, 2, 1).reshape(-1, np.size(X,axis=-1)).T
    X=wavelet_preprocessing.wavelet_rt(np.array([X]))
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
    # Get random section for evaluation data
    X_test.append(X[30000:40000])
    y_test.append(positions[30000:40000])
    X_train.append(np.concatenate([X[:30000],X[40000:]]))
    y_train.append(np.concatenate([positions[:30000],positions[40000:]]))
    print(f"Time spent on personne {i+1}: {time.time()-personne}")

# Convert the lists to NumPy arrays for better performance
X_test=np.array(X_test)
y_test=np.array(y_test)
X_train=np.array(X_train)
y_train=np.array(y_train)
# Reshape the arrays to the correct shape
X_test=X_test.reshape(-1, np.size(X_test,axis=-1))
y_test=y_test.reshape(-1)
X_train=X_train.reshape(-1, np.size(X_train,axis=-1))
y_train=y_train.reshape(-1)
# Create the 1sec windows for the data
X_test=create_windows(X_test)# Shape: (n_samples, n_timepoints, n_channels)
y_test=reshape_positions(y_test)
X_train=create_windows(X_train)
y_train=reshape_positions(y_train)
X_train=np.delete(X_train,np.where(y_train==0)[0][len(np.where(y_train==1)[0]):],0)
y_train=np.delete(y_train,np.where(y_train==0)[0][len(np.where(y_train==1)[0]):],0)

# Save the data
np.save('X_train.npy',X_train)
np.save('y_train.npy',y_train)
np.save('X_test.npy',X_test)
np.save('y_test.npy',y_test)

print(f"time spent for the whole data: {time.time()-whole}")