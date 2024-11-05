"""
We will be evaluating different preprocessing algorithms based on 7 metrics:
 -Mutual Information
 -Signal to Noise Ratio
 -Signal to Artifact Gain Coefficient
 -Mean Absolute Error in δ-band PSD
 -Percentage Improvement in Autocorrelation
 -Average Percentage Improvement in Coherence
 -Execution Time
"""
import wavelet_preprocessing
import data_extraction
import numpy as np
from scipy.signal import welch
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import coherence
import time
import ica_preprocessing


file=f"BCICIV_2a_gdf/A01T.npz"
data=data = np.load(file)
signal = data['s']
left,right,feet,tongue,fixation= data_extraction.get_data(file,signal)
eeg_data={}
eeg_data["left"]=left
eeg_data["right"]=right
eeg_data["feet"]=feet
eeg_data["tongue"]=tongue
eeg_data["fixation"]=fixation

start=time.time()
eeg_cleaned_data=wavelet_preprocessing.wavelet(eeg_data)
# To evaluate another preprocessing algorithm chose that algorithm
# eeg_cleaned_data=ica_preprocessing.ica(eeg_data)
end=time.time()

def compute_snr(original_signal, cleaned_signal):
    signal_power = np.var(cleaned_signal)
    noise_power = np.var((original_signal - cleaned_signal))
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_sar(original, clean):
    original_power = np.var(original)
    clean_power = np.var(clean)
    sar = clean_power / original_power
    return sar

# Compute Signal-to-Artifact Gain Coefficient (SAGC)
def sar(original_signal, cleaned_signal):
    sar_before = compute_sar(original_signal, cleaned_signal)
    return 10*np.log10(sar_before)


# Function to compute Power Spectral Density (PSD) using Welch's method
def compute_psd(signal, fs=250, nperseg=256):
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd

# Function to extract δ-band (0.5-4 Hz) PSD
def extract_delta_band_psd(freqs, psd, delta_range=(0.5, 4)):
    delta_mask = (freqs >= delta_range[0]) & (freqs <= delta_range[1])
    delta_psd = psd[delta_mask]
    return delta_psd

# Function to compute Mean Absolute Error (MAE) in δ-band PSD
def compute_mae_delta_psd(original_signal, cleaned_signal, fs=250, nperseg=256, delta_range=(0.5, 4)):
    # Compute PSD for the original signal
    freqs_orig, psd_orig = compute_psd(original_signal, fs=fs, nperseg=nperseg)
    # Compute PSD for the cleaned signal
    freqs_clean, psd_clean = compute_psd(cleaned_signal, fs=fs, nperseg=nperseg)
    # Extract δ-band PSD for both original and cleaned signals
    delta_psd_orig = extract_delta_band_psd(freqs_orig, psd_orig, delta_range)
    delta_psd_clean = extract_delta_band_psd(freqs_clean, psd_clean, delta_range)
    # Compute Mean Absolute Error (MAE)
    mae_delta_psd = np.mean(np.abs(delta_psd_orig - delta_psd_clean))
    return mae_delta_psd

# Function to compute autocorrelation
def autocorrelation(signal):
    result = np.correlate(signal, signal, mode='full')
    return result[result.size // 2:]  # Return only the second half (positive lags)

# Function to calculate percentage improvement
def calculate_percentage_improvement(before, after):
    return ((after - before) / abs(before)) * 100

def corrolation(original_signal, cleaned_signal):
    # Compute autocorrelation
    autocorr_before = np.max(autocorrelation(original_signal))  # Max autocorrelation of original signal
    autocorr_after = np.max(autocorrelation(cleaned_signal))  # Max autocorrelation of cleaned signal

    # Calculate percentage improvement in autocorrelation
    percentage_improvement = calculate_percentage_improvement(autocorr_before, autocorr_after)
    return percentage_improvement



# Function to compute percentage improvement in coherence
def calculate_percentage_improvement(coherence_before, coherence_after):
    return ((coherence_after - coherence_before) / abs(coherence_before)) * 100


# Function to compute percentage improvement in coherence
def coherence_clac(original_signal,cleaned_signal, fs=250):
    t = np.linspace(0, 10, fs * 3)  # 10-second signal
    improvements=[]
    for i in range(len(original_signal)):
        for j in range(i+1,len(original_signal)):
            f_before, coherence_before = coherence(original_signal[i], original_signal[j], fs=fs)
            # Compute coherence between channels after processing
            f_after, coherence_after = coherence(cleaned_signal[i], cleaned_signal[j], fs=fs)
            percentage_improvement = calculate_percentage_improvement(coherence_before, coherence_after)
            improvements.append(percentage_improvement)
    # Calculate percentage improvement in coherence at each frequency
    percentage_improvement_coherence=np.mean(improvements, axis=0)
    return percentage_improvement_coherence





# Example usage:
fs = 250  # Sampling frequency (adjust based on your data)
nperseg = 256  # Length of each segment for Welch's method
# Creating a list for metric
all_snr=[]
all_sagc=[]
all_maepsd=[]
all_mi=[]
all_cor=[]
all_coh=[]
# Calculating every metric for every sample of every task
for task in eeg_data:
    for j in range(len(eeg_data[task])):
        all_coh.append(coherence_clac(eeg_data[task][j],eeg_cleaned_data[task][j]))
        for i in range(len(eeg_data[task][j])):
            all_snr.append(compute_snr(eeg_data[task][j][i], eeg_cleaned_data[task][j][i]))
            all_sagc.append(sar(eeg_data[task][j][i], eeg_cleaned_data[task][j][i]))
            all_maepsd.append(compute_mae_delta_psd(eeg_data[task][j][i], eeg_cleaned_data[task][j][i]))
            all_mi.append(mutual_info_regression(eeg_data[task][j][i].reshape(-1, 1), eeg_cleaned_data[task][j][i]))
            all_cor.append(corrolation(eeg_data[task][j][i], eeg_cleaned_data[task][j][i]))
# Computing the mean of every metric
snr_value = np.mean(all_snr)
sagc=np.mean(all_sagc)
mae_delta_psd= np.mean(all_maepsd)
mi_value=np.mean(all_mi)
percentage_improvement=np.mean(all_cor)
coh_improvements=np.mean(all_coh)
# Finaly printing every performance metric to then compare with different preprocessing algorithms
print(f"Mutual Information: {mi_value}")
print(f'SNR after denoising: {snr_value} dB')
print(f'Signal-to-Artifact Gain Coefficient (SAGC): {sagc}')
print(f"Mean Absolute Error in δ-band PSD (MAEδ_PSD): {mae_delta_psd}")
print(f"Percentage Improvement in Autocorrelation: {percentage_improvement:.2f}%")
print(f"Average Percentage Improvement in Coherence: {coh_improvements}%")
print("The time of execution of above program is :",(end-start) * 10**3, "ms")