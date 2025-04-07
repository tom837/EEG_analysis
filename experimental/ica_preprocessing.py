import matplotlib.pyplot as plt
import data_extraction
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler



def ica(eeg_data):
    eeg_cleaned_data={}
    # Example: Assuming your EEG data is in a NumPy array with shape (n_channels, n_times)
    # Replace this with your actual NumPy array
    for task, recordings in eeg_data.items():
        eeg_cleaned_data[task] = []
        n_channels = 22  # Number of EEG channels
        # Transpose the data to shape (n_times, n_channels) for FastICA
        for i in range(len(eeg_data[task])):
            scaler = StandardScaler()
            eeg_data_T = eeg_data[task][i].T
            eeg_data_T_standardized = scaler.fit_transform(eeg_data_T)
            # Apply ICA using FastICA from scikit-learn
            ica = FastICA(n_components=n_channels, random_state=97, max_iter=1500,tol=0.008)
            sources = ica.fit_transform(eeg_data_T_standardized)  # This gives the independent components (ICs)
            # To remove artifacts, inspect components and choose which to remove (e.g., eye blinks, noise)
            # For this example, we assume the first component is an artifact (you'll need to identify these manually)
            components_to_remove = [0]  # Example: Mark the first component for removal
            # Reconstruct the signal without the artifact components
            # Set the unwanted components to zero
            sources[:, components_to_remove] = 0
            # Reconstruct the cleaned signal (still transposed)
            eeg_cleaned_T = ica.inverse_transform(sources)
            # Transpose back to original shape (n_channels, n_times)
            eeg_cleaned_data[task].append(eeg_cleaned_T.T)
        # eeg_cleaned now contains the cleaned EEG signal
    return eeg_cleaned_data

def ica_rt(eeg_data):
    eeg_data=eeg_data.T
    n_channels = 22  # Number of EEG channels
    # Transpose the data to shape (n_times, n_channels) for FastICA
    # Apply ICA using FastICA from scikit-learn
    ica = FastICA(n_components=n_channels)
    sources = ica.fit_transform(eeg_data)  # This gives the independent components (ICs)
    # To remove artifacts, inspect components and choose which to remove (e.g., eye blinks, noise)
    # Transpose back to original shape (n_channels, n_times)
    # eeg_cleaned now contains the cleaned EEG signal
    sources=np.array(sources).T
    print(sources.shape)
    return sources

if __name__ == "__main__":
    file=f"BCICIV_2a_gdf/A01T.npz"
    data=data = np.load(file)
    signal = data['s']
    left,right,feet,tongue,fixation= data_extraction.get_data(file,signal)
    eeg_data={}
    eeg_data["left"]=np.array(left)
    eeg_data["right"]=np.array(right)
    eeg_data["feet"]=np.array(feet)
    eeg_data["tongue"]=np.array(tongue)
    eeg_data["fixation"]=np.array(fixation)
    eeg_cleaned_data=ica(eeg_data)
    # Plot an example before and after wavelet denoising
    # Assuming 'left' task and first trial, first EEG channel (replace with your real data)
    original_signal = eeg_data['left'][0]  # First channel of the first trial (before denoising)
    cleaned_signal = eeg_cleaned_data['left'][0] # Same channel after denoising
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    time_axis = np.arange(original_signal[1].shape[0]) / 250
    for i in range(22):
            axes[0].plot(time_axis, original_signal[i], color="blue")
            axes[0].set_title(f'original signal')
            axes[0].set_ylabel('Amplitude (µV)')
            axes[0].grid(True)
    for i in range(22):
            axes[1].plot(time_axis, cleaned_signal[i], color="blue")
            axes[1].set_title(f'cleaned signal')
            axes[1].set_ylabel('Amplitude (µV)')
            axes[1].grid(True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
