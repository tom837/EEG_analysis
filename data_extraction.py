from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge


def extract_data(task,eeg,data):
    # Exatrcting only the critical parts of the data
    if len(np.where(data['epos']>=len(eeg[0]))[0])>0:
        event_types = data['etyp'][:np.where(data['epos']>=len(eeg[0]))[0][0]-2]
        positions = data['epos'][:np.where(data['epos']>=len(eeg[0]))[0][0]-2]
    else:
        event_types = data['etyp']
        positions = data['epos']
    # Step 1: Extract positions for task motor imagery tasks
    indexs = np.where(event_types == task)
    positions = positions[indexs]
    data_clean=[]
    for i,pos in enumerate(positions):
        tmp=[]
        for j in range (len (eeg)):
            if task== 768: # fixation=768
                # Get the data just before the subjects thinks about the task to be able to differentiate between the tasks and no task
                # Subject starts performing action 3secs after cross apears (Check desc_2a)
                tmp.append(eeg[j][pos:pos + 750])
            else:
                # Extract the time when the subject performs the task
                # Subject starts performing action 1sec after they see the task on screen and end 3 secs after (Check desc_2a)
                tmp.append(eeg[j][pos+250:pos + 1000])
        data_clean.append(tmp)
    return data_clean


def remove_eog(eeg_data):
    # Remove EOG channels from the data
    eeg_cleaned = []  # To store EEG data for each recording
    for recording in eeg_data:
            # Split the EEG and EOG data
        eeg_channels = np.array(recording[:22])  # EEG channels
        eeg_cleaned.append(eeg_channels)
    return eeg_cleaned

def cleaning_eog(eeg_data):
    # Removing EOG artifact from the eeg signals
    # Technicaly preprocessing but was one of the requirements of the competition
    # Assuming eeg_data is a list of 3-second recordings where each element has both EEG and EOG signals
    # First 22 elements: EEG channels
    # Last 3 elements: EOG channels
    eeg_cleaned = []  # To store cleaned EEG data for each recording
    for recording in eeg_data:
        # Split the EEG and EOG data
        eeg_channels = np.array(recording[:22])  # EEG channels
        eog_channels = np.array(recording[22:25])  # EOG channels
        # Initialize Ridge regressor
        ridge_regressor = Ridge(alpha=1.0)
        # Cleaned EEG recording (same shape as EEG channels)
        cleaned_recording = np.zeros_like(eeg_channels)
        # Regress out EOG from each EEG channel
        for i in range(eeg_channels.shape[0]):
            # Fit the model: EOG channels are predictors, EEG channel i is the target
            ridge_regressor.fit(eog_channels.T, eeg_channels[i])
            # Predict the EOG-related artifacts
            eog_artifact = ridge_regressor.predict(eog_channels.T)
            # Subtract the predicted EOG artifact from the EEG signal
            cleaned_recording[i] = eeg_channels[i] - eog_artifact
        # Store the cleaned EEG recording
        eeg_cleaned.append(cleaned_recording)
    return eeg_cleaned

def get_data(file,signal, eog_filter=True):
    # Exatrcting only the critical parts of the data and seperating them into each task
    data = np.load(file)
    eegchannels=[]
    for i in range(25):
        # Creating list of channels
        eegchannels.append(signal[:, i])
    # Calling the number for each task as per the desc_2a file
    left = 769
    right= 770
    feet= 771
    tongue = 772
    fixation = 768
    # Getting the critical 3sec readings for each task
    left_data= extract_data(left,eegchannels,data)
    right_data= extract_data(right,eegchannels,data)
    feet_data= extract_data(feet,eegchannels,data)
    tongue_data= extract_data(tongue,eegchannels,data)
    fixation_data = extract_data(fixation,eegchannels,data)
    if eog_filter:
        # Cleaning the EOG from the EEG
        left_data= cleaning_eog(left_data)
        right_data= cleaning_eog(right_data)
        feet_data= cleaning_eog(feet_data)
        tongue_data= cleaning_eog(tongue_data)
        fixation_data = cleaning_eog(fixation_data)
    else:
        # Removing the EOG channels
        """left_data=remove_eog(left_data)
        right_data=remove_eog(right_data)
        feet_data=remove_eog(feet_data)
        tongue_data=remove_eog(tongue_data)
        fixation_data=remove_eog(fixation_data)"""
    return left_data,right_data,feet_data,tongue_data,fixation_data

def cleaning_eog_rt(eeg_data):
    # Assuming eeg_data is a list of 3-second recordings where each element has both EEG and EOG signals
    # First 22 elements: EEG channels
    # Last 3 elements: EOG channels
    eeg_cleaned = []  # To store cleaned EEG data for each recording
        # Split the EEG and EOG data
    eeg_channels = np.array(eeg_data[:22])  # EEG channels
    eog_channels = np.array(eeg_data[22:25])  # EOG channels
    # Initialize Ridge regressor
    ridge_regressor = Ridge(alpha=1.0)
    # Cleaned EEG recording (same shape as EEG channels)
    cleaned_recording = np.zeros_like(eeg_channels)
    # Regress out EOG from each EEG channel
    for i in range(eeg_channels.shape[0]):
        # Fit the model: EOG channels are predictors, EEG channel i is the target
        ridge_regressor.fit(eog_channels.T, eeg_channels[i])
        # Predict the EOG-related artifacts
        eog_artifact = ridge_regressor.predict(eog_channels.T)
        # Subtract the predicted EOG artifact from the EEG signal
        cleaned_recording[i] = eeg_channels[i] - eog_artifact
    # Store the cleaned EEG recording
    eeg_cleaned.append(cleaned_recording)
    out=np.array(eeg_cleaned)
    out=np.squeeze(out, axis=0)
    return out

def extract_data_rt(task,event_types,positions,posi,t):
    # Extracts data used in real time classification training
    indexs = np.where(event_types == task)
    posi = posi[indexs]
    # Goes through every position where a task is performed and gets the data
    for i,pos in enumerate(posi):
        positions[pos+250:pos + 1000]=t
    return positions





if __name__ == "__main__":
    # Example of how to use get_data
    file=f"BCICIV_2a_gdf/A01T.npz"
    data=data = np.load(file)
    signal = data['s']
    left,right,feet,tongue,fixation= get_data(file,signal)
    # Ploting the first performance for each task
    fig, axes = plt.subplots(5, 1, figsize=(10, 7), sharex=True)
    time_axis = np.arange(left[0][1].shape[0]) / 250
    for i in range(22):
        # Superposing each channel on top of each other
        axes[0].plot(time_axis, fixation[0][i], color="blue")
        axes[0].set_title(f'fixation')
        axes[0].set_ylabel('Amplitude (µV)')
        axes[0].grid(True)
    for i in range(22):
        axes[4].plot(time_axis, tongue[0][i], color="blue")
        axes[4].set_title(f'tongue')
        axes[4].set_ylabel('Amplitude (µV)')
        axes[4].grid(True)
    for i in range(22):
        axes[1].plot(time_axis, left[0][i], color="blue")
        axes[1].set_title(f'left')
        axes[1].set_ylabel('Amplitude (µV)')
        axes[1].grid(True)
    for i in range(22):
        axes[2].plot(time_axis, right[0][i], color="blue")
        axes[2].set_title(f'right')
        axes[2].set_ylabel('Amplitude (µV)')
        axes[2].grid(True)
    for i in range(22):
        axes[3].plot(time_axis, feet[0][i], color="blue")
        axes[3].set_title(f'feet')
        axes[3].set_ylabel('Amplitude (µV)')
        axes[3].grid(True)
    plt.grid(True)
    plt.show()