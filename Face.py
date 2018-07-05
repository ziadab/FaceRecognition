# coding: utf-8

import numpy as np
import cv2, pickle



recognizer = cv2.face.LBPHFaceRecognizer_create()
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eye = cv2.CascadeClassifier("haarcascade_eye.xml")
recognizer.read('trainner.yml')

labels = {}
with open("labels.pkl","rb") as f:
    labelsH = pickle.load(f)
    labels = {v:k for k,v in labelsH.items()}

cap = cv2.VideoCapture("OverMyHead.mp4")

#i = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 2)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h , x:x+w]
        #print(x,y,w,h)
        #recognize 
        faceID, conf = recognizer.predict(roi_gray)
        #print(conf)
        if conf >= 5 and conf <= 80:
            #print(faceID)
            #print(labels[faceID].replace("-"," ").title())
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[faceID].replace("-"," ").title()
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "Unknown"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        #To Save The Face Just UnComment this Line and that i variable Up There in Line 12
        #roi_gray = img[y:y+h , x:x+w]
        #img_item = "img/number_"+str(i)+".png"
        #cv2.imwrite(img_item, roi_gray)
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
