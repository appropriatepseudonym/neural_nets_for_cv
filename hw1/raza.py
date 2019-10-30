from PIL import Image
import os, sys, math,random, cv2,operator
from sklearn.model_selection import train_test_split
import numpy as npy
import knn as knn

img_dataset = []
img_labels = []


def resizeImage(dir, output_dir="", size=(32,32,3)):
	infile=""
	for file in os.listdir(dir):
		imageFile = (os.path.join(dir, file))
		infile=file
		im =  Image.open(infile)
		imgPath=dir+os.sep+infile
		img = cv2.resize(cv2.imread(imageFile), (32,32))
		imgdata = npy.array(img)
		imgflat = imgdata.flatten()
		img_dataset.append(imgflat)
		img_labels.append(imgPath.split(os.path.sep)[-2])

	
if __name__=="__main__":
	rootdir = os.getcwd()
	rootdir=rootdir+os.sep+'animals'
	animals = ["cats", "dogs", "panda"]
	for anm in animals:
		os.chdir(rootdir)
		chgdir = os.getcwd()+os.sep+anm
		os.chdir(chgdir)
		dir = os.getcwd()
		output_dir = dir+os.sep+"resized"
		resizeImage(dir,output_dir=output_dir)
	
	(trainX_a, testX, trainY_a, testY) = train_test_split(img_dataset, img_labels, test_size=.20, random_state=0)
	(trainX, valX ,trainY, valY) = train_test_split(trainX_a, trainY_a, test_size=.125, random_state=0)
	
	max_k=2
	print("running test vs. validation")
	Ypred_val = []
	for i in range(1, max_k+1):
	    #load class knn to find best value for k
		knn_val = knn.KNN(i)
		 
		#load training data into model
		knn_val.train(npy.asarray(valX), npy.asarray(valY))
		
		#get the prediction for validation
		Ypred_val.append(knn_val.predict(npy.asarray(testX)))
		print("ypred val", Ypred_val)