from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import sys
import cv2

n_classes = 133
n_rows = 32
n_cols = 32
batch_size = 256
epochs = 20
learning_rate = 0.001
decay = 1e-6
momentum = .9

def create_model():
    # conv-conv-pool ==> conv-conv-pool ==> dense ==>dense
    print('Building Model..')
    model = Sequential()
    
    model.add(Conv2D(32,(3,3),input_shape=(n_rows,n_cols,1),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    BatchNormalization(axis=1)
    model.add(Conv2D(64,(3,3),activation='relu'))
    BatchNormalization(axis=1)
    model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    BatchNormalization(axis=1)
    model.add(Conv2D(64,(3,3),activation='relu')) 
    BatchNormalization(axis=1)

    model.add(MaxPooling2D((2,2),strides=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(2018, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(n_classes, activation='softmax'))
    
    sgd = SGD(lr = learning_rate, decay = decay, momentum = momentum,nesterov=True)
    model.compile(optimizer = sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    
    print('Model Generated')
    
    return model

def getClassLabel():
    classLabels = np.array(["'", "'2", '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6',
       '7', '8', '9', "item_''", 'item_,', 'item_dot', 'item_ques',
       '\xe0\xb4\x85', '\xe0\xb4\x86', '\xe0\xb4\x87', '\xe0\xb4\x89',
       '\xe0\xb4\x8b', '\xe0\xb4\x8e', '\xe0\xb4\x8f', '\xe0\xb4\x92',
       '\xe0\xb4\x95', '\xe0\xb4\x95\xe0\xb5\x8d\xe0\xb4\x95',
       '\xe0\xb4\x95\xe0\xb5\x8d\xe0\xb4\xa4',
       '\xe0\xb4\x95\xe0\xb5\x8d\xe0\xb4\xb7', '\xe0\xb4\x96',
       '\xe0\xb4\x97', '\xe0\xb4\x97\xe0\xb5\x8d\xe0\xb4\x97',
       '\xe0\xb4\x97\xe0\xb5\x8d\xe0\xb4\xa8',
       '\xe0\xb4\x97\xe0\xb5\x8d\xe0\xb4\xae',
       '\xe0\xb4\x97\xe0\xb5\x8d\xe0\xb4\xb2', '\xe0\xb4\x98',
       '\xe0\xb4\x99', '\xe0\xb4\x99\xe0\xb5\x8d\xe0\xb4\x95',
       '\xe0\xb4\x99\xe0\xb5\x8d\xe0\xb4\x99', '\xe0\xb4\x9a',
       '\xe0\xb4\x9a\xe0\xb5\x8d\xe0\xb4\x9a',
       '\xe0\xb4\x9a\xe0\xb5\x8d\xe0\xb4\x9b', '\xe0\xb4\x9b',
       '\xe0\xb4\x9c', '\xe0\xb4\x9c\xe0\xb5\x8d\xe0\xb4\x9c',
       '\xe0\xb4\x9c\xe0\xb5\x8d\xe0\xb4\x9e', '\xe0\xb4\x9d',
       '\xe0\xb4\x9e', '\xe0\xb4\x9e\xe0\xb5\x8d\xe0\xb4\x9a',
       '\xe0\xb4\x9e\xe0\xb5\x8d\xe0\xb4\x9e', '\xe0\xb4\x9f',
       '\xe0\xb4\x9f\xe0\xb5\x8d\xe0\xb4\x9f', '\xe0\xb4\xa0',
       '\xe0\xb4\xa1', '\xe0\xb4\xa2', '\xe0\xb4\xa3',
       '\xe0\xb4\xa3\xe0\xb5\x8d\xe0\xb4\x9f',
       '\xe0\xb4\xa3\xe0\xb5\x8d\xe0\xb4\xa1',
       '\xe0\xb4\xa3\xe0\xb5\x8d\xe0\xb4\xa3',
       '\xe0\xb4\xa3\xe0\xb5\x8d\xe0\xb4\xae', '\xe0\xb4\xa4',
       '\xe0\xb4\xa4\xe0\xb5\x8d\xe0\xb4\xa4',
       '\xe0\xb4\xa4\xe0\xb5\x8d\xe0\xb4\xa5',
       '\xe0\xb4\xa4\xe0\xb5\x8d\xe0\xb4\xa8',
       '\xe0\xb4\xa4\xe0\xb5\x8d\xe0\xb4\xad',
       '\xe0\xb4\xa4\xe0\xb5\x8d\xe0\xb4\xae',
       '\xe0\xb4\xa4\xe0\xb5\x8d\xe0\xb4\xb8', '\xe0\xb4\xa5',
       '\xe0\xb4\xa6', '\xe0\xb4\xa6\xe0\xb5\x8d\xe0\xb4\xa6',
       '\xe0\xb4\xa6\xe0\xb5\x8d\xe0\xb4\xa7', '\xe0\xb4\xa7',
       '\xe0\xb4\xa8', '\xe0\xb4\xa8\xe0\xb5\x8d\xe0\xb4\xa4',
       '\xe0\xb4\xa8\xe0\xb5\x8d\xe0\xb4\xa5',
       '\xe0\xb4\xa8\xe0\xb5\x8d\xe0\xb4\xa6',
       '\xe0\xb4\xa8\xe0\xb5\x8d\xe0\xb4\xa7',
       '\xe0\xb4\xa8\xe0\xb5\x8d\xe0\xb4\xa8',
       '\xe0\xb4\xa8\xe0\xb5\x8d\xe0\xb4\xae',
       '\xe0\xb4\xa8\xe0\xb5\x8d\xe0\xb4\xb1', '\xe0\xb4\xaa',
       '\xe0\xb4\xaa\xe0\xb5\x8d\xe0\xb4\xaa', '\xe0\xb4\xab',
       '\xe0\xb4\xac', '\xe0\xb4\xac\xe0\xb5\x8d\xe0\xb4\xac',
       '\xe0\xb4\xad', '\xe0\xb4\xae',
       '\xe0\xb4\xae\xe0\xb5\x8d\xe0\xb4\xaa',
       '\xe0\xb4\xae\xe0\xb5\x8d\xe0\xb4\xae',
       '\xe0\xb4\xae\xe0\xb5\x8d\xe0\xb4\xb2', '\xe0\xb4\xaf',
       '\xe0\xb4\xaf\xe0\xb5\x8d\xe0\xb4\xaf', '\xe0\xb4\xb0',
       '\xe0\xb4\xb1', '\xe0\xb4\xb1\xe0\xb5\x8d\xe0\xb4\xb1',
       '\xe0\xb4\xb1\xe0\xb5\x8d\xe0\xb4\xb12', '\xe0\xb4\xb2',
       '\xe0\xb4\xb2\xe0\xb5\x8d\xe0\xb4\xb2', '\xe0\xb4\xb3',
       '\xe0\xb4\xb3\xe0\xb5\x8d\xe0\xb4\xb3', '\xe0\xb4\xb4',
       '\xe0\xb4\xb5', '\xe0\xb4\xb5\xe0\xb5\x8d\xe0\xb4\xb5',
       '\xe0\xb4\xb6', '\xe0\xb4\xb6\xe0\xb5\x8d\xe0\xb4\x9a',
       '\xe0\xb4\xb7', '\xe0\xb4\xb8',
       '\xe0\xb4\xb8\xe0\xb5\x8d\xe0\xb4\xa5', '\xe0\xb4\xb9',
       '\xe0\xb4\xb9\xe0\xb5\x8d\xe0\xb4\xa8',
       '\xe0\xb4\xb9\xe0\xb5\x8d\xe0\xb4\xae', '\xe0\xb4\xbe',
       '\xe0\xb4\xbf', '\xe0\xb5\x80', '\xe0\xb5\x81', '\xe0\xb5\x82',
       '\xe0\xb5\x83', '\xe0\xb5\x86', '\xe0\xb5\x87', '\xe0\xb5\x8d',
       '\xe0\xb5\x8d\xe0\xb4\xaf', '\xe0\xb5\x8d\xe0\xb4\xb0',
       '\xe0\xb5\x8d\xe0\xb4\xb5', '\xe0\xb5\x97', '\xe0\xb5\xba',
       '\xe0\xb5\xbb', '\xe0\xb5\xbc', '\xe0\xb5\xbd', '\xe0\xb5\xbe',
       '\xe0\xb5\xbe2'],dtype='|S10')

    return classLabels

def predictY(model,X_test):
    X_test = X_test.reshape(X_test.shape[0],n_rows,n_cols,1)
    predictions = model.predict(X_test,batch_size = 50,verbose =1 )
    return np.array(predictions)

def preprocessImage(image):
    #Does Adaptive Gaussian Thresholding (See sudoku image in documentation for understanding)
    adaptiveGThreshold = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    retVal,thresholded_img = cv2.threshold(adaptiveGThreshold,127,255,cv2.THRESH_BINARY)
    img = cv2.resize(thresholded_img,(32,32),interpolation = cv2.INTER_CUBIC)
    img[img<127] = 0
    img[img>=127] = 255
    return img


def recognitionEngine():
    print("=======================================================================")
    print("                 Malayalam Character Recognition System")
    print("=======================================================================")
    print("Please wait while we load the model ...................................")
    model = create_model()
    model.load_weights("../Successful-Models/train3_bn_96.h5")
    print("Ready to test....")
    while(True):
      print("Input an Image source:"),
      image = raw_input()
      image_file = cv2.imread(image,0)
      image_file = cv2.resize(image_file,(32,32))
      # image_file = preprocessImage(image_file)
      # plt.imshow(image_file)
      # plt.show() 
      classLabels = getClassLabel()
      prediction = model.predict(image_file.reshape(1,n_rows,n_cols,1))
      print("=======================================================================")
      print "Predicted Output is : "+classLabels[np.argmax(prediction)]
      print("=======================================================================")
    
recognitionEngine()
