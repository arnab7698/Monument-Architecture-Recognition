from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask import Flask, render_template, redirect, request, flash
import joblib
# from flask_mysqldb import MySQL,MySQLdb 
# from werkzeug.utils import secure_filename
import os
#import magic
# import urllib.request
from datetime import datetime
import numpy as np
import pandas as pd

app = Flask(__name__)
##database





@ app.route('/',methods=['GET', 'POST'])
def home():
    title = 'Monument Architecture Recognition - Home'
    return render_template('home.html', title=title, architecture= "")

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect('/admin')
    return render_template('login.html', error=error)

import os
from werkzeug.utils import secure_filename
app.config['PATH'] = 'C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Test images'

@app.route('/predict', methods= ['GET', 'POST'])
def predict():
    title = 'Monument Architecture Recognition - Home'
    message = None
    print('Inside post')
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     return redirect(request.url)
        f = request.files['file']
        # arch= str(request.form.get("Architecture"))
        if not f:
            return render_template('home.html', title=title)
        else:
            # img = file.read()
            # when saving the file
            print('model prediction')
            filename = secure_filename(f.filename)
            filepath= os.path.join(app.config['PATH'], filename)
            print(filepath)
            f.save(filepath)
            arch = model(filepath)
            return render_template('home.html', title=title, message= message, architecture= arch)
    return render_template('home.html', title=title, architecture ="")   
            






app.config['UPLOAD_PATH'] = 'C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Git Project\\Data'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
  
def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/admin', methods= ['GET', 'POST'])
def admin():
    title = 'Monument Architecture Recognition - Admin'
    message = None
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        arch= str(request.form.get("Architecture"))
        #print(files)
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_PATH'],arch, filename))
    classes = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('admin.html', title=title, architectures= classes)  
    

@app.route('/folder', methods= ['GET', 'POST'])
def folder_upload():
    title = 'Monument Architecture Recognition - Admin'
    
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        folder= str(request.form.get("foldername"))
        os.makedirs(os.path.join(app.config['UPLOAD_PATH'],folder))
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_PATH'],folder, filename))
    
    classes = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('admin.html', title=title, architectures= classes)  
            

@app.route('/upload/<directory>/<filename>')
def send_dataset(directory, filename):
    return send_from_directory(f"Data/{directory}", filename)

@app.route('/dataset', methods= ['GET', 'POST'] )
def get_dataset():
    if request.method == 'POST':
        name= request.form['submit_button']
        
        image_names = os.listdir(os.path.join(app.config['UPLOAD_PATH'],name))
        # print(image_names)
    return render_template("dataset.html", image_names=image_names, folder= name)



# # #######Ancient ######
# @app.route('/upload/ancient/<filename>')
# def send_ancient(filename):
#     return send_from_directory("Data/Ancient", filename)

# @app.route('/ancient')
# def get_ancient():
#     image_names = os.listdir('C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Git Project\\Data\\Ancient')
#     print(image_names)
#     return render_template("ancient.html", image_names=image_names)

# ##############British#######
# @app.route('/upload/british/<filename>')
# def send_british(filename):
#     return send_from_directory("Data/British", filename)

# @app.route('/british')
# def get_british():
#     image_names = os.listdir('C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Git Project\\Data\\British')
#     print(image_names)
#     return render_template("british.html", image_names=image_names)
# # #######Indo-Islamic ######
# @app.route('/upload/indo/<filename>')
# def send_indo(filename):
#     return send_from_directory("Data/IndoIslamic", filename)

# @app.route('/indo')
# def get_indo():
#     image_names = os.listdir('C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Git Project\\Data\\IndoIslamic')
#     print(image_names)
#     return render_template("indo.html", image_names=image_names)

# # #######Maratha ######
# @app.route('/upload/maratha/<filename>')
# def send_maratha(filename):
#     return send_from_directory("Data/Maratha", filename)

# @app.route('/maratha')
# def get_maratha():
#     image_names = os.listdir('C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Git Project\\Data\\Maratha')
#     print(image_names)
#     return render_template("maratha.html", image_names=image_names)

# # #######Not a Monument ######
# @app.route('/upload/notmonument/<filename>')
# def send_notmonument(filename):
#     return send_from_directory("Data/Not a Monument", filename)

# @app.route('/notmonument')
# def get_notmonument():
#     image_names = os.listdir('C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Git Project\\Data\\Not a Monument')
#     print(image_names)
#     return render_template("notmonument.html", image_names=image_names)

# # #######Sikh ######
# @app.route('/upload/sikh/<filename>')
# def send_sikh(filename):
#     return send_from_directory("Data/Sikh", filename)

# @app.route('/sikh')
# def get_sikh():
#     image_names = os.listdir('C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Git Project\\Data\\Sikh')
#     print(image_names)
#     return render_template("sikh.html", image_names=image_names)

#============================
#Preprocessing and model

import os, numpy
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# def preprocess_data(path):
#     files = os.listdir(path)
#     orb = cv2.ORB_create()
#     #sift = cv2.xfeatures2d.SIFT_create()
#     X=numpy.zeros(shape=(1,500*32))
#     Y=numpy.zeros(shape=(1,))
#     count=-1
#     for f in files:
#              if not (f.startswith('.')):
#                 folder = os.path.join(path, f)
#                 filenames = os.listdir(folder)
#                 count+=1
#                 for fi in filenames:
#                     if not (fi.startswith('.')):
#                         filepath = os.path.join(folder, fi)
#                         #print filepath
#                         image = cv2.imread(filepath, 0)
#                         #print(filepath.split('/')[5])
#                         kp, features = orb.detectAndCompute(image,None)
#                         #print features.shape
#                         features=features.reshape(1,500*32)
#                         #print(features.shape)
#                         X = numpy.vstack((X, features))
#                         Y=numpy.vstack((Y,numpy.asarray([count])))
#         #print X.shape
#     Y=Y.reshape(Y.shape[0],)
#     return X,Y



list = ['Ancient','British','Indo-Islamic','Maratha','Not a Monument', 'Sikh']
# def model_svm(X,y,a):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#     C_score = []
#     grid = np.arange(0.01, 3, 0.1)
#     for K in grid:
#         clf = SVC(C=K)
#         # Calculate the mean scores for each value of hyperparameter C
#         scores = cross_val_score(clf, X_train, Y_train, cv=5)
#         #print(scores.mean())
#         C_score.append(scores.mean())

#         # Display the maximum score achieved at which hyperparameter value
#         #print(" max score is ", max(C_score), " at C = ", grid[C_score.index(max(C_score))])
#     clf = SVC(C=grid[C_score.index(max(C_score))])
#     clf.fit(X_train, Y_train)
#     #y_pred = clf.predict(X_test)
#     y_pred = clf.predict(a)
#     #print("accuracy is {}".format(accuracy_score(Y_test, y_pred)))
#     print(y_pred)
#     ind = int(y_pred[5])

#     return list[ind]
        #print(X_test.shape)

def model_KNN(X,Y, a):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    clf = neighbors.KNeighborsClassifier(10)
    clf.fit(X_train, Y_train)
    #y_pred = clf.predict(X_test)
    #print(X_test.shape)
    y_pred = clf.predict(a)
    #print(accuracy_score(Y_test, y_pred))
    # print(+y_pred)
    y_pred = clf.predict(a)
    ind = int(y_pred[5])
    return list[ind]

def model_randomforest(X,Y,a):
    #print("printing data shape {}".format(X.shape))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    K=1000
    clf = RandomForestClassifier(n_estimators=K)
    # Calculate the mean scores for each value of hyperparameter C
    #scores = cross_val_score(clf, X, Y, cv=5)
    #print scores.mean()
    clf.fit(X_train, Y_train)
    #y_pred = clf.predict(X_test)
    #print(features.shape, type(features))
    #print(accuracy_score(Y_test, y_pred))
    y_pred = clf.predict(a)
    ind = int(y_pred[5])
    return list[ind]

def model(filepath):
	
    image = cv2.imread(filepath, 0)
    # if np.isnan(image).any():
    #     val= 'Invalid image'
    #     return val
    # else:
    reimg = cv2.resize(image,(500,500))
    orb = cv2.ORB_create()
    kp, features = orb.detectAndCompute(reimg,None)
    #print features.shape
    features = features.reshape(1,500*32)
    a = np.zeros([1,16000])
    for i in range(54):
        a = np.vstack((a, features))

# with open('images.npy', 'rb') as f:
#     X= np.load(f)
# with open('labels.npy', 'rb') as g:     
#     Y= np.load(g)

# archi = model_svm(X,Y,a)
    f = "C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Git Project\\SVM.sav"
    loaded_model=joblib.load(f)
    
    y_pred = loaded_model.predict(a)
    #print("accuracy is {}".format(accuracy_score(Y_test, y_pred)))
    print(y_pred)
    ind = int(y_pred[5])
    #model_KNN(X,Y,a)
    #model_randomforest(X,Y,a)
    return list[ind]
        
       
    
# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
