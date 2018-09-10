# coding: utf-8

import numpy as np
import cv2, pickle, random


#Creat a face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

#List of Videos
videos = ["Crush.mp4","OverMyHead.mp4"]

#Load the haarcascade frontalface Classifier
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eye = cv2.CascadeClassifier("haarcascade_eye.xml")

#Read the trainner.yml model that generate using trainner.py
recognizer.read('trainner.yml')

#Declare an empty label dict
labels = {}

#load the names that in the video 
with open("labels.pkl","rb") as f:
    labelsH = pickle.load(f)
    labels = {v:k for k,v in labelsH.items()}

#Random Choice for the video
video = random.choice(videos)

############ Warning  #####################
cap = cv2.VideoCapture(str(video))
#for this line if you want use the input of the video is camera live steam 
#put this code :
#cap = cv2.VideoCapture(0)
#And if you wnat an other video just replace the argument in cap 
#like this:
#cap = cv2.VideoCapture("myVideo.mp4")
###########################################

#i = 0

#Creat a loop
while True:
    #read from the video/camera
    ret, img = cap.read()

    #convert the color of img to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detect faces in the picture usingg the haarcascade frontalface Classifier
    faces = face.detectMultiScale(gray, 1.3, 2)

    #cont faces in picture
    for (x,y,w,h) in faces:
        #detect the rectangle coordnition 
        roi_gray = gray[y:y+h , x:x+w]
        #print(x,y,w,h)
        #recognize 

        #predict the face in the rectangle
        faceID, conf = recognizer.predict(roi_gray)
        #print(conf)

        #if the conf is close to 0 that mean it's probley the face
        if conf >= 5 and conf <=90:
            #print(faceID)
            #print(labels[faceID].replace("-"," ").title())
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[faceID].replace("-"," ").title()
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            #print(str(name)+" : "+str(conf))

        #if the conf is away from 90 that mean it's not the person
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "Unknown"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            #print(str(name)+" : "+str(conf))
        #To Save The Face Just UnComment this Line and that i variable Up There in Line 12
        #roi_gray = img[y:y+h , x:x+w]
        #img_item = "img/number_"+str(i)+".png"
        #cv2.imwrite(img_item, roi_gray)

        #draw the rectangle
        cv2.rectangle(img, (x,y), (x+w , y+h) , (255,0,0) , 2) 
        #i = i + 1

        #cv2.rectangle(image, start_point , end_point , colorRBG , SolidSise)
        #roi_gray = gray[y:y+h , x:x+w]
        #roi_color = gray[y:y+h , x:x+w]
        #eyes = eye.detectMultiScale(roi_gray)
        #for(ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(roi_color, (ex,ey), (ex+ew , ey+eh) , (105,121,152) , 2) 
            
    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#Unknown : 96.97343204015465
