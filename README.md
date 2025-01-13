# EEG_analysis

This is the code for my eeg classification thesis project. This project focuses on the analysis and classification of EEG (Electroencephalography) signals to distinguish between task-related brain activity. The pipeline includes preprocessing steps such as bandpass filtering, and Independent Component Analysis (ICA) to clean the data. The goal is to extract meaningful features from the EEG signals and classify brain activity during motor tasks using machine learning techniques. This project has applications in brain-computer interfaces (BCI), neuroscience research, and real-time signal processing.



# How to use:

Clone the repo and download the full dataset, which you can find at https://github.com/orvindemsy/BCICIV2a-FBCSP.git (this step is optional as I provided a partial dataset already preprocessed in the file data_cleaned.npy). If you want to use the whole dataset, you must run the rt_dataset.py file to preprocess and clean the data. You can run the cnn_classification_rt.py, LDA_classification.py or the svm_classification_rt.py files they will output the accuracy and prediction-accuracy table for each person. 
Note that the classes are:
0: Nothing
1: left hand
2: right hand 
3: feet
4: tongue

This project is still a work in progress files are susceptible to change or deletion. The best results we got were using LDA and got an accuracy of 55%. 
