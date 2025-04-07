import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib
matplotlib.matplotlib_fname()
import matplotlib.pyplot as plt
from mne.baseline import rescale
from mne.datasets import somato
from mne.stats import bootstrap_confidence_interval
from mne.io import RawArray
from padasip.filters import FilterLMS
from scipy.signal import butter, lfilter
from mne.decoding import CSP
from sklearn.feature_selection import mutual_info_classif
import undersample as ud

def convert_signal(data):
    sfreq = 250  # Sampling frequency in Hz
    n_channels, n_timepoints = data.shape
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
    raw=mne.io.RawArray(data,info)

    # Set the montage for the EEG channels (10-20 system)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    return raw





def clean_using_bfre(data,data_eval, data_bfre,labels):
    mne.viz.set_browser_backend("qt")
    # Convert both before and during signals to mne objects
    raw=convert_signal(data)
    raw_eval=convert_signal(data_eval)
    raw.pick_types(eeg=True, eog=False)
    raw_eval.pick_types(eeg=True,eog=False)
    #raw.plot(scalings='auto',title="not cleaned")
    raw_bfre=convert_signal(data_bfre)
    raw_bfre.pick_types(eeg=True, eog=False)
    raw= raw.copy().filter(l_freq=1.0, h_freq=None)
    raw_bfre= raw_bfre.copy().filter(l_freq=1.0, h_freq=None)
    # Create a single ICA and train only on during signal
    components=20
    ica = mne.preprocessing.ICA(n_components=components, random_state=42)
    ica.fit(raw)
    # Get both before and during components based on the same ICA
    before_sources = ica.get_sources(raw_bfre).get_data()
    during_sources = ica.get_sources(raw).get_data()
    # Compute correlation matrix between before and during ICA components
    cors=[]
    threshold = 0.12
    for i in range(4):
        labe=np.where(labels==i+1)[0][0]
        bf=i*250*60
        # take a small section of each label at random to compare with a random section of the data before
        get_data_bfr = ica.get_sources(convert_signal(data_bfre[:,bf+250:bf+750])).get_data()
        get_data_during = ica.get_sources(convert_signal(data[:,labe+250:labe+750])).get_data()
        correlations=[]
        for j in range(len(get_data_bfr)):
            cov_matrix = np.cov(get_data_during[j], get_data_bfr[j])
            covariance = cov_matrix[0, 1]

            # Compute standard deviations using np.cov() for consistency
            std_during = np.sqrt(cov_matrix[0, 0])
            std_before = np.sqrt(cov_matrix[1, 1])

            # Compute Pearson correlation coefficient
            correlation = covariance / (std_during * std_before)
            correlations.append(correlation)
        matched_components = np.where(np.abs(correlations) >= threshold)[0]
        cors.append(matched_components)
        print("correlations", correlations)
        print("matched_components", matched_components)
    # Identify components with correlation higher or equal to threshold (needs tunning)
    # Components to remove
    components_to_remove=[]
    print(cors)
    for i in range(components):
        k=0
        for j in range(len(cors)):
            if i in cors[j]:
                k+=1
        if k>=3:
            components_to_remove.append(i)
    print(f"Components to remove: {components_to_remove}")
    # Remove the matched components from the during-classification signal
    ica.exclude = components_to_remove

    # Apply ICA cleaning to remove these components
    raw_during_cleaned = ica.apply(raw)
    raw_eval_cleaned= ica.apply(raw_eval)
    #raw_during_cleaned.plot(scalings='auto',block=True,title="cleaned")
    # Convert back to numpy array
    output=raw_during_cleaned.get_data()
    output_eval=raw_eval_cleaned.get_data()
    return output.T, output_eval.T
    
    
    
    
def clean(data):
    mne.viz.set_browser_backend("qt")
    raw=convert_signal(data)
    raw=extr_band(raw)
    """print(raw)
    print(raw.info)
    print(raw.ch_names)
    print("Data min:", raw._data.min())
    print("Data max:", raw._data.max())
    print("Data mean:", raw._data.mean())"""
    #raw.plot(scalings='auto',block=True)

    regexp =r'(.)'
    artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
    #raw.plot(scalings='auto', order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False,block=True)

    eog_evoked = mne.preprocessing.create_eog_epochs(raw).average()
    eog_evoked.apply_baseline()



    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)


    ica = mne.preprocessing.ICA(n_components=22, max_iter="auto", random_state=97)
    ica.fit(filt_raw)
    ica

    explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
    """for channel_type, ratio in explained_var_ratio.items():
        print(f"Fraction of {channel_type} variance explained by all components: {ratio}")"""
    raw.load_data()
    print("sources")
    #ica.plot_sources(raw,picks=[0,1,2,3,4],show_scrollbars=False,title="main sources")

    #ica.plot_components(title="components")


    """ica.plot_overlay(raw,exclude=[0], picks="eeg",title="overlay 0")
    ica.plot_overlay(raw,exclude=[1], picks="eeg",title="overlay 1")
    ica.plot_overlay(raw,exclude=[3], picks="eeg",title="overlay 3")
    ica.plot_overlay(raw,exclude=[0,1,3], picks="eeg",title="overlay 0,1,3")
    ica.plot_properties(raw, picks=[0,1,2,3,4])"""
    
    #ica.exclude = [0, 1,4,5,7,8,10,16,17,18,20,21]
    # find which ICs match the EOG pattern
    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)
    # barplot of ICA component "EOG match" scores
    """ica.plot_scores(eog_scores)

    # plot diagnostics
    ica.plot_properties(raw, picks=eog_indices)

    # plot ICs applied to raw data, with EOG matches highlighted
    ica.plot_sources(raw, show_scrollbars=False,title='og')

    # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    ica.plot_sources(eog_evoked,title='artifacts')"""
    """reconst_raw = raw.copy()
    ica.apply(raw)
    """

    """raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False,scalings='auto',block=True,title='clean')
    reconst_raw.plot(
        order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False,scalings='auto',block=True,title='og'
    )"""
    #reconst_raw.plot(scalings='auto',block=True,title="Cleaned")
    eeg_data = reconst_raw.get_data()  # Clean method to extract the data
    # Get the source activations (all ICA components)
    sources = ica.get_sources(raw)

    # Exclude the components in `ica.exclude`
    remaining_sources = [i for i in range(ica.n_components_) if i not in ica.exclude]

    # Extract the data of the remaining components as a NumPy array
    sources_data = sources.get_data()  # Shape: (n_components, n_times)
    remaining_data = sources_data[remaining_sources, :]  # Keep only the remaining components

    # If needed, transpose the array to shape it as (n_times, n_components)
    remaining_data = remaining_data.T  # Shape: (n_times, n_remaining_components)
    return eeg_data.T, remaining_data



def extr_band(data):
    # Example: Assume raw is your mne.Raw object
    raw_beta = data.copy().filter(l_freq=8., h_freq=30., picks='eeg')  # Beta band filter
    return raw_beta


def band(data):
    raw=convert_signal(data)
    raw=extr_band(raw)
    return raw.get_data(), "blank"


def bandpass_filter(data, low_freq, high_freq, fs, order=4):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=-1)

def filter_data_by_band(eeg_data, frequency_bands, fs):
    filtered_data = []
    for band in frequency_bands:
        low, high = band
        filtered = bandpass_filter(eeg_data, low, high, fs)
        filtered_data.append(filtered)
    return filtered_data  # List of filtered data arrays, one per band

def apply_csp_to_bands(filtered_data, labels,eval_data, n_components=4):
    csp = CSP(n_components=n_components)  # Initialize CSP
    csp_features = []
    csp_features_eval=[]
    for i in range(len(filtered_data)):
        print(labels.shape)
        tr,trys=ud.undersample_data(np.array(filtered_data[i]),np.array(labels))
        csp.fit(tr, trys)  # Fit CSP and transform data
        features= csp.transform(filtered_data[i])
        features_eval=csp.transform(eval_data[i])
        csp_features.append(features)  # Collect CSP features for this band
        csp_features_eval.append(features_eval)
    return np.hstack(csp_features), np.hstack(csp_features_eval) # Concatenate features across bands



def apply_fbcsp_with_mibif(filtered_data, labels, eval_data, n_components=4, num_features=4):
    """
    Apply FBCSP with multiclass extension using OVR and MIBIF for feature selection.
    :param filtered_data: List of band-pass filtered EEG data for training (list of [n_samples, n_channels, n_times]).
    :param labels: Class labels for the training data (numpy array of shape [n_samples]).
    :param eval_data: List of band-pass filtered EEG data for evaluation (list of [n_samples, n_channels, n_times]).
    :param n_components: Number of CSP components to extract.
    :param num_features: Number of best features to select using MIBIF.
    :return: Selected CSP features for training and evaluation data.
    """
    num_classes = len(np.unique(labels))  # Determine the number of classes
    csp_features_train = []
    csp_features_eval = []

    # Apply OVR for each class
    for class_label in np.unique(labels):
        ovr_labels = (labels == class_label).astype(int)  # Convert to binary labels for OVR
        band_features_train = []
        band_features_eval = []

        # Apply CSP for each band
        print("not trained",np.array(filtered_data).shape)
        for i in range(len(filtered_data)):
            csp = CSP(n_components=n_components)  # Initialize CSP
            csp.fit(filtered_data[i], ovr_labels)  # Fit CSP for this band
            features_train = csp.transform(filtered_data[i])  # Transform training data
            features_eval = csp.transform(eval_data[i])  # Transform evaluation data
            band_features_train.append(features_train)
            band_features_eval.append(features_eval)

        # Combine CSP features from all bands
        print("combined",np.array(band_features_train).shape)
        combined_features_train = np.hstack(band_features_train)
        combined_features_eval = np.hstack(band_features_eval)
        print("staked",np.array(combined_features_train).shape)

        # Select best features using MIBIF
        mi_scores = mutual_info_classif(combined_features_train, ovr_labels)  # Calculate mutual information scores
        best_features_idx = np.argsort(mi_scores)[-num_features:]  # Get indices of best features

        selected_train = combined_features_train[:, best_features_idx]
        selected_eval = combined_features_eval[:, best_features_idx]

        csp_features_train.append(selected_train)
        csp_features_eval.append(selected_eval)

    # Stack features for all classes
    final_features_train = np.hstack(csp_features_train)
    final_features_eval = np.hstack(csp_features_eval)

    return final_features_train, final_features_eval