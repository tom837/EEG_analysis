# EEG_analysis

This is the code for my eeg classification thesis project.

# How to use:

In the files there are two different types of classifications. the first one is task by task classification, it is distinguished with cl in the names of the files. To create and train the cnn model symply run the cnn_classification_cl.py file, it automaticaly evaluates the model and saves it. The second type of classification is real time classification. The data is setup to be a "continus" stream of 1second windows with 0.5seconds overlaping between each window. To train the model run the cnn_classification_rt.py or the knn_classification_rt.py. rt_dataset and cl_dataset are not usefull here as the dataset has already been processed in the X_train and X_test files but you can find the full dataset at https://github.com/orvindemsy/BCICIV2a-FBCSP.git. 
