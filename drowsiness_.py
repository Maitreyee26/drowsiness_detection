#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

import numpy as np


# In[4]:


Datadirectory = "C:\\Users\\USER\\Desktop\\project\\Train/"
Classes = ["Closed_Eyes", "Open_Eyes"]
for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            plt.imshow(img_array, cmap="gray")
            plt.show()
            break
    break


# In[5]:


#Resizing the Image

#Resize all the images into 224 x 224 for better feature extraction.

img_size = 224
new_array = cv2.resize(backtorgb, (img_size, img_size))
plt.imshow(new_array, cmap="gray")
plt.show()


# In[6]:


training_Data = []
def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category) # 0 1,
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
                new_array = cv2.resize(backtorgb, (img_size, img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_Data()


# In[8]:


#Random shuffling to avoid overfitting.

import random
random.shuffle(training_Data)


# In[9]:


#Creating arrays to store features and labels

X = []
y = []
for features,label in training_Data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)


# In[10]:


X.shape


# In[11]:


#normalization of x
X=X/255.0


# In[12]:


Y=np.array(y)


# In[13]:


import pickle
pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()


# In[14]:


pickle_in=open("X.pickle","rb")
X=pickle.load(pickle_in)

pickle_in=open("y.pickle","rb")
y=pickle.load(pickle_in)


# In[15]:


from tensorflow import keras
from tensorflow.keras import layers

from keras.models import load_model

# load the H5 model file

new_model = keras.models.load_model('drowsiness_model.h5')


# In[ ]:


model = tf.keras.applications.mobilenet.MobileNet()
base_input = model.layers[0].input ##input
base_output = model.layers[-4].output
Flat_layers = layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layers)
final_output = layers.Activation('sigmoid')(final_output)
new_model = keras.Model(inputs = base_input, outputs = final_output)
new_model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=["accuracy"])
new_model.fit(X,Y, epochs = 20, validation_split = 0.2) ##training
new_model.save('drowsiness_model.h5')


# In[16]:


new_model.summary()


# In[1]:


from platform import python_version
print(python_version())


# In[21]:


img_array = cv2.imread('s0001_02342_0_0_1_0_0_01.png', cv2.IMREAD_GRAYSCALE)
backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
new_array = cv2.resize(backtorgb, (img_size, img_size))


# In[22]:


X_input = np.array(new_array).reshape(1, img_size, img_size, 3)
#X_input = X_input/255.0 #normalizing data
prediction = new_model.predict(X_input)


# In[23]:


#show the image
X_input.shape
plt.imshow(new_array)


# In[24]:


prediction



# In[ ]:





# In[30]:


import winsound
frequency = 2000  # Set frequency to 2500
duration = 1500  # Set duration to 1500 ms == 1.5 sec
import numpy as np
import cv2
path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(1)
#check if webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
cap.set(cv2.CAP_PROP_FPS, 5)
counter = 0
while True:
    ret,frame = cap.read()
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    for x,y,w,h in eyes:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
        eyess = eye_cascade.detectMultiScale(roi_gray)
        if len(eyess) == 0:
            print("Eyes are not detected")
        else:
            for (ex, ey, ew, eh) in eyess:
                eyes_roi = roi_color[ey: ey+eh, ex: ex+ew]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(faceCascade.empty()==False):
            print("detected")
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    # # Draw a rectangle around eyes
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
    final_image = cv2.resize(eyes_roi, (224,224))
    final_image = np.expand_dims(final_image, axis=0)
    #final_image = final_image/255.0
    Predictions = new_model.predict(final_image)
    if (Predictions>0.9):
        status = "Open Eyes"
        cv2.putText(frame,
               status,
               (150,150),
               font, 3,
               (0, 255, 0),
               2,
               cv2.LINE_4)
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0,0,0), -1)
       #Add text
        cv2.putText(frame, 'Active', (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        counter = counter + 1
        status = "Closed Eyes"
        cv2.putText(frame,
                status,
                (150,150),
                font, 3,
                (0, 0, 255),
                2,
                cv2.LINE_4)
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame, (x1,y1), (x1 + w1, y1 + h1), (0,0,255), 2)
        if counter > 10:
            x1,y1,w1,h1 = 0,0,175,75
            #Draw black background rectangle
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0,0,0), -1)
            #Add text
            cv2.putText(frame, "Sleep Alert !!!", (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            winsound.Beep(frequency,duration)
            counter = 0
    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




