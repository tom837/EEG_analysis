import numpy as np
import pywt
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input,GRU,Conv2D,MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization, LayerNormalization,Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.layers import LeakyReLU
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.decomposition import PCA




def accs(X_test,y_test,model):
    predictions = model.predict(X_test)
    # Get the label of the class with the highest probabilityindex
    predicted_labels = np.argmax(predictions, axis=1)
    n_correct=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==y_test[i]:
            #print(predicted_labels[i],y_test[i])
            n_correct+=1
    # Print the ratio of correctness of the model in the evaluation dataset
    accuracy= n_correct/len(predicted_labels)
    kappa = cohen_kappa_score(y_test,predicted_labels)
    return accuracy,kappa


def acc2(X_test,y_test,model):
    predicted_labels=model.predict(X_test)
    n_correct=0
    for i in range(len(predicted_labels)):
        if predicted_labels[i]==y_test[i]:
            n_correct+=1
    # Print the ratio of correctness of the model in the evaluation dataset
    accuracy= n_correct/len(predicted_labels)
    kappa = cohen_kappa_score(y_test,predicted_labels)
    return accuracy,kappa


def FFT(data):
    data=np.abs(np.fft.rfft(data,axis=-1))
    return data


def DWT(data):
    X_tmp=[]
    for k in range(len(data)):
        X_tmp.append([])
        for f in range(len(data[k])):
            X_tmp[k].append(pywt.dwt(data[k][f],'db1'))
    data=np.array(X_tmp)
    return data

def principal_component(data,eval,labels):
    n=15
    pca=PCA(n_components=n)
    tmp=[]
    for sample in data:
        tmp.append(pca.fit_transform(sample))
    data=np.array(tmp)
    pca=PCA(n_components=n)
    tmp=[]
    for sample in eval:
        tmp.append(pca.fit_transform(sample))
    eval=np.array(tmp)
    return data,eval

def csp(data,eval,labels):
    n_components=4
    csp = CSP(n_components=n_components)
    csp.fit(data, labels)  # Fit CSP and transform data
    data= csp.transform(data)
    eval=csp.transform(eval)
    return data,eval


def svm(X_train,y_train,X_test,y_test):
    X_train=X_train.reshape(X_train.shape[0],-1)
    X_test=X_test.reshape(X_test.shape[0],-1)
    model=SVC(kernel='rbf', C=0.001)
    model.fit(X_train,y_train)
    accuracy,kappa=acc2(X_test,y_test,model)
    accuracy_train,_=acc2(X_train,y_train,model)
    return [accuracy,kappa,accuracy_train]

def lda(X_train,y_train,X_test,y_test):
    X_train=X_train.reshape(X_train.shape[0],-1)
    X_test=X_test.reshape(X_test.shape[0],-1)
    model=LinearDiscriminantAnalysis()
    model.fit(X_train,y_train)
    accuracy,kappa=acc2(X_test,y_test,model)
    accuracy_train,_=acc2(X_train,y_train,model)
    return [accuracy,kappa,accuracy_train]


def cnn(X_train,y_train,X_test,y_test):
    y_train=to_categorical(y_train)
    X_train,X_val,y_train,y_val=train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    if len(X_train.shape)<=3:
        model = Sequential()
        # 1D Convolutional Layer (for spatial feature extraction across channels)
        if len(X_train.shape)==3:
            model.add(Input(shape=(X_train.shape[1],X_train.shape[2])))
        else:
            model.add(Input(shape=(X_train.shape[1],)))
        model.add(Conv1D(filters=12, kernel_size=5, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())


        # Adding these layers make the model better but increase overfiting
        model.add(Conv1D(filters=64, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=3,activation='relu'))
        model.add(BatchNormalization())

        # Max pooling
        model.add(MaxPooling1D(pool_size=3))
        model.add(Dropout(0.5))
        model.add(Conv1D(filters=32, kernel_size=3,activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))
        # Flatten the output
        model.add(Flatten())

        # Fully connected layer
        
        model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
        model.add(LayerNormalization())
        model.add(Reshape((32,2)))
        model.add(GRU(units=32))
        #model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))


        # Output layer (multiclass classification for each task and no task)
        model.add(Dense(5, activation='softmax',kernel_regularizer=regularizers.l2(0.001)))
    else:
        model = Sequential()
        # 1D Convolutional Layer (for spatial feature extraction across channels)
        print(X_train.shape[1:])
        model.add(Input(shape=(X_train.shape[1:])))


        # Adding these layers make the model better but increase overfiting
        # First Conv2D Layer
        model.add(Conv2D(filters=64, kernel_size=(3,2), padding="same", kernel_regularizer=regularizers.l2(0.001)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        
        # Second Conv2D Layer (Fixed kernel size)
        model.add(Conv2D(filters=32, kernel_size=(3,1),padding="same"))  # ✅ Prevents shrinking too fast
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())


        # Max pooling
        model.add(MaxPooling2D(pool_size=(3,1)))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=32, kernel_size=(3,1),padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,1)))
        model.add(Dropout(0.5))
        # Flatten the output
        model.add(Flatten())

        # Fully connected layer
        
        model.add(Dense(64, activation=LeakyReLU(alpha=0.1),kernel_regularizer=regularizers.l2(0.001)))
        model.add(LayerNormalization())
        model.add(Reshape((32,2)))
        model.add(GRU(units=32))
        #model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.5))


        # Output layer (multiclass classification for each task and no task)
        model.add(Dense(5, activation='softmax',kernel_regularizer=regularizers.l2(0.001)))
        
    # Compile the model
    loss=tf.keras.losses.CategoricalCrossentropy()

    acc='categorical_crossentropy'
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001,use_ema=True,ema_momentum=0.9)

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    # Perform K-fold cross-validation
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    # Testing this solution to avoid overfiting but it doesn't seem to help
    # K fold cross validation
    model.fit(X_train, y_train, epochs=5000,validation_data=(X_val, y_val), batch_size=64,callbacks=early_stopping)
    y_train=np.argmax(y_train,axis=1)
    accuracy,kappa=accs(X_test,y_test,model)
    accuracy_train,_=accs(X_train,y_train,model)
    return [accuracy,kappa,accuracy_train]


def ovr(X_train,y_train,X_test,y_test):
    X_train=X_train.reshape(X_train.shape[0],-1)
    X_test=X_test.reshape(X_test.shape[0],-1)
    classes=5
    models=[]
    for i in range(classes):
        lda = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
        models.append((f'lda{i}', lda))
    model=StackingClassifier(models,CalibratedClassifierCV())
    model.fit(X_train,y_train)
    accuracy,kappa=acc2(X_test,y_test,model)
    accuracy_train,_=acc2(X_train,y_train,model)
    return [accuracy,kappa,accuracy_train]


def a(data):
    return data

def b(data,eval,labels):
    return data,eval

def undersample_data(X, y):
    """
    Undersample each class to have the same number of samples as the minority class.

    Parameters:
        X (numpy array): Feature data of shape (n_samples, n_features).
        y (numpy array): Labels of shape (n_samples,).

    Returns:
        X_balanced, y_balanced: Undersampled feature data and labels.
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_samples = min(class_counts)

    X_balanced = []
    y_balanced = []

    for cls in unique_classes:
        # Indices of all samples in the class
        indices = np.where(y == cls)[0]
        # Randomly select min_samples from the class
        selected_indices = np.random.choice(indices, min_samples, replace=False)
        X_balanced.append(X[selected_indices])
        y_balanced.append(y[selected_indices])

    return np.vstack(X_balanced), np.concatenate(y_balanced)


preprocessing1=["None","FFT","DWT"]
preprocessing2=["None","PCA","CSP"]

classifiers=["SVM","LDA","CNN","OVR"]

preprocessing1_act=[a,FFT,DWT]
preprocessing2_act=[b,principal_component,csp]
classifiers_act=[svm,lda,cnn,ovr]


data=np.load('data_cleaned.npy')
labels=np.load('y_train_rt.npy')
data_eval=np.load('data_cleaned_eval.npy')
labels_eval=np.load('y_train_rt_eval.npy')
data=data.transpose(0,1,3,2)
data_eval=data_eval.transpose(0,1,3,2)

X_tr_og,y_train= undersample_data(data[0], labels[0])
X_te_og,y_test=undersample_data(data_eval[0],labels_eval[0])


X_t,_=csp(X_tr_og,X_te_og,y_train)

print(X_t.shape)


accuracys=[]
for pre1 in range(len(preprocessing1)):
    print("computing preprocessing1:",preprocessing1[pre1])
    X_tr_1=preprocessing1_act[pre1](X_tr_og)
    X_te_1=preprocessing1_act[pre1](X_te_og)
    for pre2 in range(len(preprocessing2)):
        print("Computing preprocessing2:",preprocessing2[pre2])
        if pre2!=0 and pre1==2:
            X_tr_1=X_tr_1.reshape(X_tr_1.shape[0],X_tr_1.shape[1],-1)
            X_te_1=X_te_1.reshape(X_te_1.shape[0],X_te_1.shape[1],-1)
        X_tr_2, X_te_2=preprocessing2_act[pre2](X_tr_1,X_te_1,y_train)
        for classif in range(len(classifiers)):
            if pre2==2 and classif==2:
                print("Ignore",preprocessing2[pre2],classifiers[classif])
            else:
                print("Training classifiers:",classifiers[classif])
                accss=classifiers_act[classif](X_tr_2,y_train,X_te_2,y_test)
                name=preprocessing1[pre1]+'_'+preprocessing2[pre2]+'_'+classifiers[classif]
                print("accuracies:",[name]+accss)
                accuracys.append([name]+accss)
                np.save('accuracies_noica.npy',np.array(accuracys))



