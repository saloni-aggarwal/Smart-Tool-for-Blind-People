import keras
import tensorflow as tf
import h5py
from keras.models import load_model
import cv2
import numpy as np
from gtts import gTTS
import os

Dict = {0: 'Airplane', 1: 'Automobile', 2: 'Bird',
        3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse',
        8: 'Ship', 9: 'Truck'}

model = load_model('model3.h5')
# print('Model Loaded')

dim = (32, 32)
img = cv2.imread('downloadc.jpg')
img = cv2.resize(img, dim)
Array = np.reshape(img, (1, 32, 32, 3))

Prediction = model.predict(Array)
# print(Prediction)
Prediction = Prediction.tolist()
# print(Prediction)

FinalList = []

for SubList in Prediction:
    for element in SubList:
        FinalList.append(element)

# print(FinalList)

index = FinalList.index(1.0)

tts = gTTS(text=('Its a ' + Dict[index] + ' you looking at'), lang='en')
tts.save("good.mp3")
os.system("good.mp3")
