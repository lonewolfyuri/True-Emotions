# %%
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.optimizers import SGD
import keras
import tensorflow as tf
import glob
# %%

def FaceRecognition(img,padding=10):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = gray
    # Find faces
    print(gray.shape)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(48, 48)
    )
    for (x, y, w, h) in faces:
        print(x)

        cv2.rectangle(img, (x-padding-1, y-1-padding), (x + w+1+padding, y + h+1+padding), (0, 255, 0), 2)
    try:
        if x>0 and y >0 and (x+w)<img.shape[1] and (y+h) <img.shape[0]:
            faceImg = img[(y-padding):(y+h+padding),(x-padding):(x+w+padding)]
        else:
            faceImg = 1.0*img
        faceImg = cv2.resize(faceImg,(48,48))
    except UnboundLocalError:
        faceImg = 1.0*img
        faceImg = cv2.resize(faceImg,(48,48))


    print(img.shape,faceImg.shape)
    return img, faceImg
#opt = SGD(lr=0.005, momentum=0.9)
# %%
model = tf.keras.models.load_model(r'test_model.h5')

# %%

def label_key(label_num):
    if label_num == 0:
        label_string = 'Angry'
    elif label_num == 1:
        label_string = 'Disgust'
    elif label_num == 2:
        label_string = 'Fear'
    elif label_num == 3:
        label_string = 'Happy'
    elif label_num == 4:
        label_string = 'Sad'

    elif label_num == 5:
        label_string = 'Surprised'
    elif label_num == 6:
        label_string = 'Neutral'
    else:
        print('Unknown label number')
        label_string ='Invalid'
    return label_string
# %%
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

keys = []
for ii in range(7):
    keys.append(label_key(ii))

print('here')
while(True):
    #Capture images
    ret, frame = cap.read()
    #Display
    image, face = FaceRecognition(frame)
    print(image.shape)
    print(face.shape)
    prediction = model.predict(np.array([face/255.0]).reshape((1,48,48,1)))
    text1 = ''
    text2 = ''
    for ii in range(4):
        text1 += '%s: %0.2f    '%(keys[ii],prediction[0][ii])

    for ii in range(3):
        text2 += '%s: %0.2f    '%(keys[ii+4],prediction[0][ii+4])
    resizeFace= cv2.resize(face,(640,480))
    #resizeFace = face
    combined = np.hstack((image,resizeFace))
    cv2.putText(combined,text1,(50,50),font,1.0,(0,0,0),2,0)
    cv2.putText(combined,text2,(50,75),font,1.0,(0,0,0),2,0)

    cv2.imshow('images', combined)
#    ax.cla()
#    ax.imshow(combined)
#    ax.set_title(emotion,size=20)
#    plt.pause(0.1)
    #cv2.title(emotion)
    #Click q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Release camera
cap.release()
cv2.destroyAllWindows()