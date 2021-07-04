import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from time import time

# Loading the training data from csv file
data = pd.read_csv("train.csv")

# Extracting the feature columns
feature_columns = list(data.columns[1:])

# Extract target column 'label'
target_column = data.columns[0]

# Separate the data into feature data and target data (X and y, respectively)
X = data[feature_columns]
y = data[target_column]

# Apply CPA by fitting the data with only 60 dimensions
pca = PCA(n_components=60).fit(X)
# Transform the data using the PCA fit above
X = pca.transform(X)
y = y.values

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Fitting a K-nearest neighbor classifier on the training set with k=2
knneighbor_classifier = KNeighborsClassifier(n_neighbors=2, p=2)
knneighbor_classifier.fit(X_train, y_train)

# Initializing the array of predicted labels
y_predict = np.empty(len(y_test), dtype=int)

startTime = time()

# Find the nearest neighbors indices for each sample in the test set
kneighbors = knneighbor_classifier.kneighbors(X_test, return_distance=False)

def same_label(items):
    return len(set(items)) == 1
# For each set of neighbors indices
for idx, indices in enumerate(kneighbors):
    # Finding the actual training samples & their labels
    neighbors = [X_train[i] for i in indices]
    neighbors_labels = [y_train[i] for i in indices]

    # if all labels are the same, use it as the prediction and store in y_predict
    if same_label(neighbors_labels):
        y_predict[idx] = neighbors_labels[0]
    else:
        # else fitting a SVM classifier using the neighbors, and labelling the test samples
        #Radial base function is being implemented because of non-linearity of data
        svm_clf = svm.SVC(C=0.5, kernel='rbf', decision_function_shape='ovo', random_state=42)
        #fitting the SVM of neighbors
        svm_clf.fit(neighbors, neighbors_labels)
        #test samples are being labelled
        label = svm_clf.predict(X_test[idx].reshape(1, -1))

        y_predict[idx] = label

# accuracy in percentage

TotalTime = time() - startTime

print("Accuracy Of SVM-kNN Model is: ", end="")
print(accuracy_score(y_test, y_predict)*100)
print("Total Execution time is: ", TotalTime)
