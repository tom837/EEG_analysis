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

