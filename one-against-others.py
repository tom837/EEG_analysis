import numpy as np
import undersample as ud
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from imblearn.over_sampling import SMOTE


data=np.load('data_cleaned.npy')
labels=np.load('y_train_rt.npy')
data_eval=np.load('data_cleaned_eval.npy')
labels_eval=np.load('y_train_rt_eval.npy')
#data=np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
data=data.transpose(0,1,3,2)
data_eval=data_eval.transpose(0,1,3,2)

def binary(labels,classe):
    labs=np.zeros_like(labels)
    labs[np.where(labels==classe)[0]]=1
    return labs


def pred(data,models,multiclass):
    pre=[]
    for model in models:
        predicted=model.predict_proba(data)
        predicted=predicted[:,1]
        pre.append(predicted)
    pre=np.array(pre)
    pre=pre.T
    result=multiclass.predict(pre)
    return result




kapa_values=[]
for i in range(len(data)):
    sm = SMOTE(random_state=42)
    X_train,y_train = ud.undersample_data(data[i], labels[i])
    #X_train,y_train= ud.undersample_data(data[i], labels[i])
    X_test,y_test=ud.undersample_data(data_eval[i],labels_eval[i])
    print("Training data shape", X_train.shape)
    print("Test data shape",X_test.shape)
    n_samples, n_channels, n_timepoints = X_train.shape
    # FFT for each channel

    X_train=np.abs(np.fft.rfft(X_train,axis=-1))
    X_train=X_train.reshape(X_train.shape[0],-1)
    print(X_train.shape)
    n_samples, n_channels, n_timepoints = X_test.shape
    # FFT for each channel

    X_test=np.abs(np.fft.rfft(X_test,axis=-1))
    X_test=X_test.reshape(X_test.shape[0],-1)
    print(X_test.shape)
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    cv_split = cv.split(X_train)
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=20)
    classes=5
    models=[]
    for i in range(5):
        """y=binary(y_train,i)
        X,y=ud.undersample_data(X_train,y)
        trainx,testx,trainy,testy=train_test_split(X,y, test_size=0.2, random_state=42)"""
        lda = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
        #csp = CSP(n_components=10 , reg=None, log=True, norm_trace=False)
        #clf = Pipeline([("CSP", csp),("LDA", lda)])
        #lda.fit(trainx, trainy)
        #score=lda.score(testx,testy)
        #scores[i]=score
        models.append((f'lda{i}', lda))
    classif=StackingClassifier(models,CalibratedClassifierCV())
    scores_windows = []

    """for train_idx, test_idx in cv_split:
        y_tr, y_tst = y_train[train_idx], y_train[test_idx]
        X_tr = classif.fit(X_train[train_idx], y_tr)
        X_tstt = classif.transform(X_train[test_idx])
        # fit classifier
        # running classifier: test classifier on sliding window
        score_this_window = []
        score=classif.score(X_train[test_idx], y_tst)
        print("final score:", score)
        scores_windows.append(score)"""
    print("here")
    classif.fit(X_train,y_train)
    print("Here?")
    predicted_labels=classif.predict(X_test)
    score=classif.score(X_train,y_train)
    print("WTF")
    print(score)
    """train=[]
    for model in models:
        predicted=model.predict_proba(X_train)
        predicted=predicted[:,1]
        train.append(predicted)
    train=np.array(train).T
    multiclass=LogisticRegression()
    multiclass.fit(train,y_train)
    predicted_labels=pred(X_test,models,multiclass)"""
    n_correct=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==y_test[i]:
            n_correct+=1
    conf_matrix = confusion_matrix(y_test, predicted_labels)
    class_report=classification_report(y_test, predicted_labels)
    # Print the ratio of correctness of the model in the evaluation dataset
    print("ratio correctness: ", n_correct/len(predicted_labels))
    kappa = cohen_kappa_score(y_test,predicted_labels)
    kapa_values.append(kappa)
    print("Kappa value:", kappa)
    """plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=True)
    # Customize the plot
    plt.title("Prediction-Accuracy Table: Type")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.show()"""
    """print(conf_matrix)
    print(class_report)"""
print(kapa_values)
print(np.mean(kapa_values))