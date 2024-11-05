import numpy as np
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.models import load_model


# Testing out the model on the evaluation dataset
X_test=np.load("X_test_rta.npy")
y_test=np.load("y_test_rta.npy")
model = load_model('3rd_rt_cnn.keras')
X_test=X_test.transpose(0, 2, 1)
# Format the data correctly
predictions = model.predict(X_test)
# Get the label of the class with the highest probabilityindex
predicted_labels = np.argmax(predictions, axis=1)
# Print the predicted class labels
print(predicted_labels)
# Count how many correct labels the model got
n_correct=0
new=0
for i in range(len(predicted_labels)):
    if predicted_labels[i]==y_test[i]:
        n_correct+=1
# Print the ratio of correctness of the model in the evaluation dataset
print("ratio correctness: ", n_correct/len(predicted_labels))

# Compute kappa value
kappa = cohen_kappa_score(y_test,predicted_labels)
print("Kappa value:", kappa)