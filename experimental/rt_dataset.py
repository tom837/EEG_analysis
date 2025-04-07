import numpy as np
import data_extraction
import wavelet_preprocessing
import time
import matplotlib.pyplot as plt
import ica_preprocessing
import mne
from mne.preprocessing import ICA
import ica

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
    print(f"Time spent on personne {i+1}: {time.time()-personne}")
X_train=np.array(X_train)
y_train=np.array(y_train)
X_train=X_train.transpose(0,2,1)
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

# Create an MNE Info object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create the RawArray object
raw = mne.io.RawArray(X_train, info)

# Set the montage for the EEG channels (10-20 system)
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)


# Plot the raw data (optional)
#raw.plot()


# Assuming `raw` is your MNE Raw object with EEG data
raw.filter(1., 40., fir_design='firwin')  # Apply a band-pass filter (optional but recommended)
print(X_train.shape)
ica = ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.plot_components() 
eog_indices, eog_scores = ica.find_bads_eog(raw)  # Detect EOG-related components

# Optionally, plot the time series of ICs for visual inspection
ica.plot_sources(raw, picks=eog_indices)

# Mark EOG components for exclusion
ica.exclude = eog_indices
ica.plot_sources(raw)

print("starting")
X_train=data_extraction.cleaning_eog_rt(np.array(X_train))
print("done")
print(np.array(X_train).shape)
X_train=wavelet_preprocessing.wavelet_rt(np.array(X_train))
X_train=np.array(X_train)
print(X_train.shape)
f=5
fig, axes = plt.subplots(f, 1, figsize=(10, 7), sharex=True)
time_axis = np.arange(X_train[0][:250].shape[0]) / 250
for i in range(f):
    # Only showing the 1st second for better visibility
    axes[i].plot(time_axis, X_train[i][i*250:i*250+250], color="blue")
    axes[i].set_title(f'cleaned signal')
    axes[i].set_ylabel('Amplitude (µV)')
    axes[i].grid(True)
plt.grid(True)
plt.tight_layout()
plt.show()

tmpx=X_train
tmpy=y_train

X_train=[]
y_train=[]
X_test=[]
y_test=[]
for i in range(9):
    X_test.append(tmpx[i*600000+30000:i*600000+40000])
    y_test.append(tmpy[i*600000+30000:i*600000+40000])
    X_train.append(np.concatenate([tmpx[:i*600000+30000],tmpx[i*600000+40000:]]))
    y_train.append(np.concatenate([tmpy[:i*600000+30000],tmpy[i*600000+40000:]]))
# Convert the lists to NumPy arrays for better performance
X_test=np.array(X_test)
y_test=np.array(y_test)
X_train=np.array(X_train)
y_train=np.array(y_train)
print(X_test.shape)
print(X_train.shape)
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

X_test=X_test.transpose(0,2,1)
X_train=X_train.transpose(0,2,1)"""
tmp=[]
tmp1=[]
for i in range(len(X_train)):
    data_cleaned,_ = ica.clean(X_train[i])
    data_cleaned=data_cleaned[:,:22]


    data_mean_removed = data_cleaned - np.mean(data_cleaned, axis=1, keepdims=True)
    # Z-score normalization
    data_cleaned = data_mean_removed / np.std(data_mean_removed, axis=1, keepdims=True)


    data_cleaned=create_windows(data_cleaned)# Shape: (n_samples, n_timepoints, n_channels)
    tmp.append(reshape_positions(y_train[i]))
    print(tmp)
    for j in range(len(tmp)):
        print(len(tmp[j]))
    print("y_train shape:", tmp[i].shape)
    print("data shape:", data_cleaned.shape)
    data_cleaned = np.nan_to_num(data_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    tmp1.append(data_cleaned)
y_train=np.array(tmp)
X_train=np.array(tmp1)
# Save the data

print(X_train.shape)
print(y_train.shape)
np.save('data_cleaned.npy',X_train)
np.save('y_train_rt.npy',y_train)
#np.save('X_test.npy',X_test)
#np.save('y_test.npy',y_test)

print(f"time spent for the whole data: {time.time()-whole}")