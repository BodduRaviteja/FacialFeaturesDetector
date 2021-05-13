import cv2
import numpy as np
# Load the cascade
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
# Read the input image
img = cv2.imread('group_pic.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)
#cv2.waitKey(0)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.005, 2)
#print(type(faces))
#print(faces.shape)
#print(faces)
# Draw rectangle around the faces
for (x,y,w,h) in faces:
     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
     roi_gray = gray[y:y + h, x:x + w]
     #cv2.imshow('roi_gray', roi_gray)
     roi_color = img[y:y + h, x:x + w]
     #cv2.imshow('roi_color', roi_color)
     eyes = eye_cascade.detectMultiScale(roi_gray, 2, 5)
     smile = smile_cascade.detectMultiScale(roi_gray, 2, 5)
     for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
     for (sx,sy,sw,sh) in smile:
         cv2.rectangle(roi_gray, (sx,sy), (sx+sw, sy+sh), (0,255,0,2))

# Display the output
cv2.imshow('img', img)
#print(x, y, w, h)
cv2.waitKey()
cv2.destroyAllWindows()

