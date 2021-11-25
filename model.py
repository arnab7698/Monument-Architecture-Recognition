import os, numpy
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import faiss
list = ['Ancient','British','Indo-Islamic','Maratha','Not a Monument', 'Sikh']
with open('images.npy', 'rb') as f:
        X= np.load(f)
with open('labels.npy', 'rb') as g:     
        Y= np.load(g)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
C_score = []
grid = np.arange(0.01, 3, 0.1)
for K in grid:
    clf = SVC(C=K)
    # Calculate the mean scores for each value of hyperparameter C
    scores = cross_val_score(clf, X_train, Y_train, cv=5)
    #print(scores.mean())
    C_score.append(scores.mean())

    # Display the maximum score achieved at which hyperparameter value
    #print(" max score is ", max(C_score), " at C = ", grid[C_score.index(max(C_score))])
clf = SVC(C=grid[C_score.index(max(C_score))])
clf.fit(X_train, Y_train)
#y_pred = clf.predict(X_test)
filename= "C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\SVM.sav"
joblib.dump(clf, filename)
print('Done')


