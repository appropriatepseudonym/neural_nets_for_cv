import cv2
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import knn
"""
max_k is the highest k value for k nearest neighbors
"""
max_k = 20

"""
Create a list of all images recursively through working directory
This assumes the working directory has all images and images are seperated
by directories containing the class name
https://docs.python.org/2/library/glob.html
"""
img_list = list(glob.glob('**/*.jpg', recursive=True))

"""
#This section used in testing code outside of for loop
image=cv2.imread(img_list[0])
small = cv2.resize(image,(32,32)).flatten()
"""

#initialize dataset and label lists
data_set = [] 
label = []

#for loop to iterate through img_list
print("adding images...")
for i in img_list:

    #import image for processing
    image = cv2.imread(i)

    #data_set.append() adds the image to dataset (this done last)
    #cv2.resize().flatten will resize the image to 32x32x3 and then flatten the array to 3072x1         
    data_set.append(cv2.resize(image,(32,32)).flatten())
    """
    set label for image. inner os.path.split(i)[-2] removes image name. outter removes everythiing before
    https://docs.python.org/3/library/os.path.html
    """
    label.append(os.path.split(os.path.split(i)[-2])[-1])
    #print to let me know that the image was added to the dataset
    #print(i + ' was added to the dataset')
print("splitting data into train, validation, and test")
"""
split data into training, validation, and test. test is 20% (600 imgs), validation is 10% (300 imgs),
and training is 70% (2100 imgs). the first line seperates test out. since training and test is further split
the _a is added to train iot differentiate data. test_size for validation is 12.5% of the remaining imgs 
to give the 300 imgs we need. => the second line splits train_a into train and validation
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""
(trainX_a, testX, trainY_a, testY) = train_test_split(data_set, label, test_size=.20, random_state=0)
(trainX, valX ,trainY, valY) = train_test_split(trainX_a, trainY_a, test_size=.125, random_state=0)

"""
initialize class knn.KNN with k=1
Find the best k-value
this is a for loop to go k=1:max_k

"""

Ypred_val = []
for i in range(1, max_k+1):
    print("running validation vs. test for k =", i)
    #load class knn to find best value for k=i
    knn_val = knn.KNN(i)
    
    #load validation-set as training data into model
    knn_val.train(np.asarray(trainX), np.asarray(trainY))
    
    #get the prediction
    Ypred_val.append(knn_val.predict(np.asarray(valX)))

"""
This will evaluate the different k values for l1 and l2 to determine the most accurate value for k
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
"""

best_k = np.zeros((max_k, 2), dtype=float)
report_val_l1 = []
report_val_l2 = []

"""
i is the k value
get the accuracy reports for each k value
this uses micro averaging
"""
print("testing for the best k value from 1 to", max_k)
for i in range(0,max_k):
    report_val_l1.append(precision_recall_fscore_support(valY, Ypred_val[i][:, 0], labels=np.unique(label), average='micro'))
    report_val_l2.append(precision_recall_fscore_support(valY, Ypred_val[i][:, 1], labels=np.unique(label), average='micro'))

"""
set f-score for each k value; l1 is col 0, l2 is col 1
wanted to do argmax(report_val_lx[: , 2]) but couldn't quite get it to work
so instead created array using the f-score values. since the index is the same,
will return the correct value
"""
for i in range(0,max_k):
    best_k[i][0]=report_val_l1[i][2]
    best_k[i][1]=report_val_l2[i][2]

"""
This will predict l1/l2 for the best k
uses argmax() to find higest k-index of f-score 
"""

print("running test vs train for l1 - manhattan for k =", np.argmax(best_k[:, 0])+1)
knn_train_l1 = knn.KNN(np.argmax(best_k[:, 0]))
knn_train_l1.train(np.asarray(trainX), np.asarray(trainY))
Ypred_train_l1 = knn_train_l1.predict_l1(np.asarray(testX))

print("running test vs train for l2 - euclidean for k =", np.argmax(best_k[:, 1])+1)
knn_train_l2 = knn.KNN(np.amax(best_k[:, 1]))
knn_train_l2.train(np.asarray(trainX), np.asarray(trainY))
Ypred_train_l2 = knn_val.predict_l2(np.asarray(testX))

print("l1 - manhattan report with k =", np.argmax(best_k[:, 0])+1)
report_train_l1 = classification_report(testY, Ypred_train_l1, target_names=np.unique(label))
print(report_train_l1)

print("l2 - euclidean report with k =", np.argmax(best_k[:, 1])+1)
report_train_l2 = classification_report(testY, Ypred_train_l2, target_names=np.unique(label))
print(report_train_l2)