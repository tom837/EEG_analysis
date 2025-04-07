import numpy as np
import data_extraction
import wavelet_preprocessing
import time
import matplotlib.pyplot as plt
import ica_preprocessing
import mne
from mne.preprocessing import ICA
import ica
from imblearn.over_sampling import SMOTE
import undersample as ud

# Creating the training and validation data for the real time classification




def cr_win(eeg_data, labels, window_size=250,step_size=125):
    output=[]
    labs=[]
    i=[]
    while i*step_size+window_size<len(eeg_data):
        output.append(eeg_data[i*step_size:i*step_size+window_size])
        labs.append(labels[i*step_size+(window_size//2)])
    return output, labs
    

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
X_eval=[]
y_eval=[]
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
    print(f"Getting data from person {i+1}")
    # Load each file
    file=f"BCICIV_2a_gdf/A0{i+1}T.npz"
    data = np.load(file)
    # Differentiate between training data (signal) and evaluation data(X) ~90/10
    # The evaluation data is going to be a continuous stream as oposed to the training data that is going to be sequential
    if i+1 ==4:
        X=data['s'][750:600750]
    else:
        X=data['s'][60000:660000]
    # Cleaning the data
    """X=create_windows(X,window_size=250, step_size=250)
    X=X.transpose(0, 2, 1)
    X=data_extraction.cleaning_eog(X)
    X=np.array(X).transpose(0, 2, 1)
    X=np.array(X).transpose(0, 2, 1).reshape(-1, np.size(X,axis=-1)).T
    print(X.shape)
    X=wavelet_preprocessing.wavelet_rt(np.array([X]))
    # Reshaping the data to match what we need it for
    X=np.squeeze(X, axis=0)
    X=X.T
    print(X.shape)"""
    # Creating the labels for evaluation data
    positions=np.zeros(len(data['s']))
    positions= data_extraction.extract_data_rt(769,data['etyp'],positions,data['epos'],1)
    positions= data_extraction.extract_data_rt(770,data['etyp'],positions,data['epos'],2)
    positions= data_extraction.extract_data_rt(771,data['etyp'],positions,data['epos'],3)
    positions= data_extraction.extract_data_rt(772,data['etyp'],positions,data['epos'],4)
    if i+1 ==4:
        positions=positions[750:600750]
    else:
        positions=positions[60000:660000]
    # Get random section for evaluation data
    X_train.append(X)
    y_train.append(positions)
    # Evaluation Data (basicaly the same thing as the training)
    file=f"BCICIV_2a_gdf/A0{i+1}E.npz"
    file_cl=f"true_labels/A0{i+1}E.npz"
    data = np.load(file)
    data_cl=np.load(file_cl)
    # Differentiate between training data (signal) and evaluation data(X) ~90/10
    # The evaluation data is going to be a continuous stream as oposed to the training data that is going to be sequential
    X=data['s'][60000:600000]
    cl = data_cl['classlabel'].astype('int32')
    for j in range(4):
        cl[np.where(cl==j+1)[0]]=769+j
    type=data['etyp']
    type[np.where(type==[783])[0]]=cl
    # Creating the labels for evaluation data
    positions=np.zeros(len(data['s']))
    positions= data_extraction.extract_data_rt(769,type,positions,data['epos'],1)
    positions= data_extraction.extract_data_rt(770,type,positions,data['epos'],2)
    positions= data_extraction.extract_data_rt(771,type,positions,data['epos'],3)
    positions= data_extraction.extract_data_rt(772,type,positions,data['epos'],4)
    positions=positions[60000:600000]
    X_eval.append(X)
    y_eval.append(positions)
    print(f"done extracting person {i+1}")
X_train=np.array(X_train)
y_train=np.array(y_train)
X_eval=np.array(X_eval)
y_eval=np.array(y_eval)
X_eval=X_eval.transpose(0,2,1)
X_train=X_train.transpose(0,2,1)
<<<<<<< Updated upstream
print(X_train.shape)
"""
data_mean_removed = X_train - np.mean(X_train, axis=1, keepdims=True)

# Z-score normalization
X_train = data_mean_removed / np.std(data_mean_removed, axis=1, keepdims=True)


sfreq = 250  # Sampling frequency in Hz
n_channels, n_timepoints = X_train.shape

eeg_ch_names = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz', 'Pz', 'Fz', 'Oz','A1','A2']

# Define EOG channel names based on your setup
eog_ch_names = ['EOG 1', 'EOG 2', 'EOG 3']  # Replace with actual labels if known

# Combine EEG and EOG channel names
ch_names = eeg_ch_names + eog_ch_names

# Define channel types
ch_types = ['eeg'] * 22 + ['eog'] * 3
=======
>>>>>>> Stashed changes

tmp=[]
tmp1=[]
tmp_eval=[]
tmp1_eval=[]
for i in range(len(X_train)):
    print(f"--- processing person {i+1} ---")
    personne=time.time()
    data_cleaned,_= ica.clean(X_train[i])
    data_eval,_=ica.clean(X_eval[i])
    """data_cleaned=X_train[i].T
    data_eval=X_eval[i].T"""
    data_cleaned=data_cleaned[:,:22]
    data_eval=data_eval[:,:22]
    print(data_cleaned.shape)
    data_mean_removed = data_cleaned - np.mean(data_cleaned, axis=1, keepdims=True)
    # Z-score normalization
    data_cleaned = data_mean_removed / np.std(data_cleaned, axis=1, keepdims=True)

    data_mean_removed = data_eval - np.mean(data_eval, axis=1, keepdims=True)
    # Z-score normalization
    data_eval = data_mean_removed / np.std(data_eval, axis=1, keepdims=True)
    data_cleaned=create_windows(data_cleaned)# Shape: (n_samples, n_timepoints, n_channels)
    data_eval=create_windows(data_eval)
    tmp.append(reshape_positions(y_train[i]))
<<<<<<< Updated upstream
    print(tmp)
    for j in range(len(tmp)):
        print(len(tmp[j]))
    print("y_train shape:", tmp[i].shape)
    print("data shape:", data_cleaned.shape)
=======
    tmp_eval.append(reshape_positions(y_eval[i]))
>>>>>>> Stashed changes
    data_cleaned = np.nan_to_num(data_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    data_eval = np.nan_to_num(data_eval, nan=0.0, posinf=0.0, neginf=0.0)
    tmp1.append(data_cleaned)
    tmp1_eval.append(data_eval)
    print(f"Time spent on person {i+1}: {time.time()-personne}")
y_train=np.array(tmp)
X_train=np.array(tmp1)
y_eval=np.array(tmp_eval)
X_eval=np.array(tmp1_eval)
# Save the data

<<<<<<< Updated upstream
print(X_train.shape)
print(y_train.shape)
=======
print("eval shapes", X_eval.shape, y_eval.shape)
print("Train shapes", X_train.shape,y_train.shape)

>>>>>>> Stashed changes
np.save('data_cleaned.npy',X_train)
np.save('y_train_rt.npy',y_train)
np.save('data_cleaned_eval.npy',X_eval)
np.save('y_train_rt_eval.npy',y_eval)
#np.save('X_test.npy',X_test)
#np.save('y_test.npy',y_test)

print(f"time spent for the whole data: {time.time()-whole}")