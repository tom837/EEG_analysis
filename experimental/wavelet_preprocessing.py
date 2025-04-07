import numpy as np
import pywt  # PyWavelets library for wavelet transform
import matplotlib.pyplot as plt
import data_extraction
import numpy as np
import pywt


def wavelet_denoise(signal, wavelet='sym5', level=1, threshold_method='soft'):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Apply thresholding to detail coefficients (all but the first approximation coefficient)
    threshold = np.sqrt(2 * np.log(len(signal)))  # Example threshold
    coeffs[1:] = [pywt.threshold(c, threshold, mode=threshold_method) for c in coeffs[1:]]
    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal

def wavelet(eeg_data):
    # Applying wavelet preprocessing to the dataset
    # Create a dictionary to store the cleaned EEG data
    eeg_cleaned_data = {}
    # Apply wavelet denoising to each component (left, right, etc.) and each EEG channel
    for task, recordings in eeg_data.items():
        eeg_cleaned_data[task] = []  # Initialize list for each task (e.g., 'left')
        # Iterate over each trial in the task
        for trial in recordings:
            cleaned_trial = []
            # Iterate over each EEG channel (22 channels in total)
            for channel in range(trial.shape[1]):
                eeg_channel_data = trial[:, channel]
                # Denoise the channel using wavelet denoising
                denoised_channel = wavelet_denoise(eeg_channel_data)
                cleaned_trial.append(denoised_channel)
            # Convert cleaned trial back to 2D array and store it in the cleaned data dictionary
            cleaned_trial = np.array(cleaned_trial).T  # Transpose to match original structure
            eeg_cleaned_data[task].append(cleaned_trial)
    return eeg_cleaned_data


def wavelet_rt(eeg_data):
    # Similar to the other function but is used to processes differently shaped data for real time classification
    # Apply wavelet denoising to each component (left, right, etc.) and each EEG channel
    eeg_cleaned_data = []  # Initialize list for each task (e.g., 'left')
    # Iterate over each trial in the task
    cleaned_trial = []
    # Iterate over each EEG channel (22 channels in total)
    for channel in range(eeg_data.shape[1]):
        eeg_channel_data = eeg_data[:, channel]
        # Denoise the channel using wavelet denoising
        denoised_channel = wavelet_denoise(eeg_channel_data)
        cleaned_trial.append(denoised_channel)
    # Convert cleaned trial back to 2D array and store it in the cleaned data dictionary
    cleaned_trial = np.array(cleaned_trial).T  # Transpose to match original structure
    return cleaned_trial


def dwt_feature_extraction(data, wavelet='db4', level=None):
    n_samples, n_timepoints, n_channels = data.shape
    # Initialize a list to store coefficients
    all_coeffs = []
    for sample in range(n_samples):
        sample_coeffs = []
        for channel in range(n_channels):
            samples=[]
            signal = data[sample, :, channel]  # Extract the signal for the current sample and channel
            # Apply wavelet decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            samples=coeffs[0]
            for i in range(1,len(coeffs)):
                samples=np.concatenate((samples,coeffs[i]))
            sample_coeffs.append(samples)
        all_coeffs.append(sample_coeffs)  # Append the coefficients for this sample
    return np.array(all_coeffs)








if __name__ == "__main__":
    file=f"BCICIV_2a_gdf/A01T.npz"
    data=data = np.load(file)
    signal = data['s']
    left,right,feet,tongue,fixation= data_extraction.get_data(file,signal,False)
    original_signal = left[0]  # First channel of the first trial (before denoising)
    left,right,feet,tongue,fixation= data_extraction.get_data(file,signal)
    eeg_data={}
    eeg_data["left"]=left
    eeg_data["right"]=right
    eeg_data["feet"]=feet
    eeg_data["tongue"]=tongue
    eeg_data["fixation"]=fixation
    eeg_cleaned_data=wavelet(eeg_data)
    # Plot an example before and after wavelet denoising
    # Assuming 'left' task and first trial, first EEG channel (replace with your real data)
    cleaned_signal = eeg_cleaned_data['left'][0] # Same channel after denoising
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    time_axis = np.arange(original_signal[0][:250].shape[0]) / 250
    # Only showing the 1st second for better visibility
    axes[0].plot(time_axis, original_signal[0][:250], color="blue")
    axes[0].set_title(f'original signal')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].grid(True)
    axes[1].plot(time_axis, cleaned_signal[0][:250], color="blue")
    axes[1].set_title(f'cleaned signal')
    axes[1].set_ylabel('Amplitude (µV)')
    axes[1].grid(True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
