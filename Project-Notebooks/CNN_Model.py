
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
import numpy as np
import nbimporter
from Train_Dataset_Generation import *


# ### Hyperparameters

# In[2]:

n_classes = 133
n_rows = 32
n_cols = 32
batch_size = 16
epochs = 10
learning_rate = 0.1
decay = 1e-6
momentum = .9


# ### Creating The Model

# In[3]:

def create_model():
    # conv-conv-pool ==> conv-conv-pool ==> dense ==>dense
    print('Building Model..')
    model = Sequential()
    
    model.add(Conv2D(32,(3,3),input_shape=(n_rows,n_cols,1),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    
    model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,(3,3),activation='relu')) 
    
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


# In[4]:

def trainCNN(X_train,y_train):
    
    model = create_model()
    print('Training Started . Please Wait ......')
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1)
    print('Training Completed')
    
    return model


# In[5]:

def saveWeights(model,name):
    model.save('Models/'+name+'.h5')
    print('Model Saved to Models/'+name+'.h5')


# In[6]:

def loadSavedWeight(name):
    model = load_model('Models/'+name+'.h5')
    return model


# In[7]:

def MalayalamCharacterRecognition():
    
    X_train,X_test,y_train,y_test = mal_char_data()
    print('Loaded Dataset')
    
#     model = loadSavedWeight("train1")
    model = trainCNN(X_train,y_train)
    
    saveWeights(model,"train1")
    
    return model


# In[8]:

model = MalayalamCharacterRecognition()


# In[ ]:



