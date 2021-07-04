import numpy as np
from numpy.lib.type_check import imag
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from time import time
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

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
knneighbor_classifier = KNeighborsClassifier(n_neighbors=3, p=2)
model = knneighbor_classifier.fit(X_train, y_train)

# Initializing the array of predicted labels

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
        print("Knn will classifiy")
    else:
        # else fitting a SVM classifier using the neighbors, and labelling the test samples
        #Radial base function is being implemented because of non-linearity of data
        svm_clf = svm.SVC(C=0.5, kernel='rbf', decision_function_shape='ovo', random_state=42)
        #fitting the SVM of neighbors
        model = svm_clf.fit(neighbors, neighbors_labels)
        #test samples are being labelled
        
        
# accuracy in percentage

TotalTime = time() - startTime


def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    
    res = model.predict([img])[0]
    return np.argmax(res), max(res)




class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command =self. self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit)+', '+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()
