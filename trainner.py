import numpy as np
from PIL import Image
import os, cv2, pickle

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

mainDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(mainDir,"img")

current_id =0
label_dec = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(imageDir):
	for file in files:
		path = os.path.join(root,file)
		labels = os.path.basename(root).replace(" ","-").lower()
		if labels in label_dec:
			pass
		else:
			label_dec[labels] = current_id
			current_id = current_id+1
		labelID = label_dec[labels]
		#print(label_dec)
		pilImage = Image.open(path).convert("L")#gray scale 
		#size = (300,300)
		#finalImage = pilImage.resize(size, Image.ANTIALIAS)#resize the image
		#image_array = np.array(finalImage,"uint8")
		image_array = np.array(pilImage,"uint8")
		#print(image_array)
		faces = faceCascade.detectMultiScale(image_array, 1.3, 2)
		for (x,y,w,h) in faces:
			roi = image_array[y:y+h, x:x+w]
			x_train.append(roi)
			y_labels.append(labelID)


with open("labels.pkl","wb") as f:
	pickle.dump(label_dec, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
