import numpy as np
import data_extraction
import wavelet_preprocessing

file="BCICIV_2a_gdf/A01T.npz"
data=data = np.load(file)
# Differentiate between training data (signal) and evaluation data(X) ~90/10
# The evaluation data is going to be a continuous stream as oposed to the training data that is going to be sequential
X=data['s'][-60000:]
# Getting only the needed seccions from the data
X=data_extraction.cleaning_eog_rt(X)
X=wavelet_preprocessing.wavelet_rt(X)
# Reshaping the data to match what we need it for
X=np.squeeze(X, axis=0)
X=X.T
print(len(X))
# Creating the labels for evaluation data
positions=np.zeros(len(data['s']))
#positions= data_extraction.extract_data_rt(769,data['etyp'],positions,X,data['epos'],1)
"""positions= data_extraction.extract_data_rt(770,data['etyp'],positions,X,data['epos'],2)
positions= data_extraction.extract_data_rt(771,data['etyp'],positions,X,data['epos'],3)
positions= data_extraction.extract_data_rt(772,data['etyp'],positions,X,data['epos'],4)"""