import feature_extroctors as fe
import feature_selectors as fs
import classificators as cl
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import feature_selectors as fs
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

ftr_ext=[fe.dwt_feature_extraction,fe.psd_feature_extraction,fe.fft_feature_extraction]
ftr_sl=[fs.pca,fs.mrmr,fs.relief,fs.rfe]
classif=[cl.cnn,cl.dnn,cl.knn,cl.lda,cl.lstm,cl.rf]

X_train=np.load("X_train_rt.npy")
X_test=np.load("X_test_rt.npy")
y_train=np.load("y_train_rt.npy")
y_test=np.load("y_test_rt.npy")
X_test=X_test.transpose(0,2,1)
X_train=X_train.transpose(0, 2, 1)
results={}

print(len(ftr_ext))

for f in range(len(ftr_ext)):
    X_train_ftr=ftr_ext[f](X_train)
    X_test_ftr=ftr_ext[f](X_test)
    for j in range(len(ftr_sl)):
        X_train_sl=ftr_sl[j](X_train_ftr,y_train)
        X_test_sl=ftr_sl[j](X_test_ftr,y_test)
        for k in range(len(classif)):
            name = f"{f}_{j}_{k}"
            print("taking care of", name)
            if classif[k]==cl.cnn or classif[k]==cl.dnn or classif[k]==cl.lstm:
                tmp_train = X_train_sl.reshape((X_train_sl.shape[0], X_train_sl.shape[1], 1))
                tmp_test = X_test_sl.reshape((X_test_sl.shape[0], X_test_sl.shape[1], 1))
                tmpy_train=to_categorical(y_train, num_classes=5)
                X_tr, X_te, y_tr, y_te = train_test_split(tmp_train, tmpy_train, test_size=0.3, random_state=42)
                model=classif[k](tmp_train)
                early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                model.fit(X_tr, y_tr, epochs=1000, batch_size=32,validation_data=(X_te,y_te),callbacks=early_stopping)
                predictions = model.predict(tmp_test)
                # Get the label of the class with the highest probabilityindex
                predicted_labels = np.argmax(predictions, axis=1)
                # Count how many correct labels the model got
                n_correct=0
                for i in range(len(predicted_labels)):
                    if predicted_labels[i]==y_test[i]:
                        n_correct+=1
                n_cor=0
                predictions = model.predict(tmp_train)
                # Get the label of the class with the highest probabilityindex
                predicted = np.argmax(predictions, axis=1)
                for i in range(len(predicted)):
                    if predicted[i]==y_train[i]:
                        n_cor+=1
                # Print the ratio of correctness of the model in the evaluation dataset
                results[name]=[n_correct/len(predicted_labels),n_cor/len(predicted)]
            else:
                model=classif[k](X_train_sl)
                model.fit(X_train_sl, y_train)
                predicted_labels = model.predict(X_test_sl)
                # Get the label of the class with the highest probabilityindex
                #predicted_labels = np.argmax(predictions, axis=1)
                # Count how many correct labels the model got
                n_correct=0
                for i in range(len(predicted_labels)):
                    if predicted_labels[i]==y_test[i]:
                        n_correct+=1
                n_cor=0
                # Get the label of the class with the highest probabilityindex
                predicted = model.predict(X_train_sl)
                for i in range(len(predicted)):
                    if predicted[i]==y_train[i]:
                        n_cor+=1
                # Print the ratio of correctness of the model in the evaluation dataset
                results[name]=[n_correct/len(predicted_labels),n_cor/len(predicted)]
            print(results[name])
print(results)
