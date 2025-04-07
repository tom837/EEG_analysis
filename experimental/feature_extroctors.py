import numpy as np
from scipy.signal import welch
import pywt

def psd_feature_extraction(data, fs=250, nperseg=250):
    psd_features = []
    for channel in data:
        feat=[]
        for signal in channel:
            _, psd = welch(signal, fs=fs, nperseg=nperseg)
            feat.append(psd)
        psd_features.append(feat)
    return np.array(psd_features)

from scipy.stats import skew, kurtosis, entropy



def fft_feature_extraction(data, fs=250):
    features = []
    for channel in data:
        feat=[]
        for signal in channel:
            # Compute the FFT
            fft_result = np.fft.fft(signal)
            fft_magnitude = np.abs(fft_result)  # Magnitude of the FFT
            freqs = np.fft.fftfreq(len(signal), d=1/fs)  # Frequency bins
            # Only return the positive half of the spectrum
            positive_freq_indices = freqs > 0
            feat.append(freqs[positive_freq_indices]+fft_magnitude[positive_freq_indices])
        features.append(feat)
    return np.array(features)

def dwt_feature_extraction(data, wavelet='db4', level=None):
    n_samples, _, n_channels = data.shape
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
    X_train=np.load("X_train_rt.npy")
    X_test=np.load("X_test_rt.npy")
    X_test=X_test.transpose(0,2,1)
    X_train=X_train.transpose(0, 2, 1)
    train=fft_feature_extraction(X_train)
    test=fft_feature_extraction(X_test)
    print(train.shape, X_train.shape)
    print(test.shape, X_test.shape)

