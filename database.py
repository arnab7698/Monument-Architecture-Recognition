import os
import numpy
import cv2



def preprocess_data(path):
    files = os.listdir(path)
    orb = cv2.ORB_create()
    #sift = cv2.xfeatures2d.SIFT_create()
    X=numpy.zeros(shape=(1,500*32))
    Y=numpy.zeros(shape=(1,))
    count=-1
    for f in files:
             if not (f.startswith('.')):
                folder = os.path.join(path, f)
                filenames = os.listdir(folder)
                count+=1
                for fi in filenames:
                    if not (fi.startswith('.')):
                        filepath = os.path.join(folder, fi)
                        #print filepath
                        image = cv2.imread(filepath, 0)
                        #print(filepath.split('/')[5])
                        kp, features = orb.detectAndCompute(image,None)
                        shape = features.shape
                        if (shape[0]==500 and shape[1]==32):
                            features=features.reshape(1,500*32)
                            X = numpy.vstack((X, features))
                            Y=numpy.vstack((Y,numpy.asarray([count])))
                        else:
                            continue
        #print X.shape
    Y=Y.reshape(Y.shape[0],)
    return X,Y

X,Y= preprocess_data("C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\DPM\\Git Project\\Data")
with open('images.npy', 'wb') as f:
    numpy.save(f, X)
with open('labels.npy', 'wb') as g:
    numpy.save(g, Y)

print('Done')