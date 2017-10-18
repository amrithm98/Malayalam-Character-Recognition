import cv2
import os,glob
def process(filename, key,folderName):

    image = cv2.imread(filename)

    #print image.shape

    r = 100.0 / image.shape[1]
    dim = (100, int(image.shape[0] *r))

    imageresized = cv2.resize(image,(32,32),dim,interpolation = cv2.INTER_AREA)
    imageresized = cv2.cvtColor(imageresized,cv2.COLOR_BGR2GRAY)
    path='/home/amrith/Machine-Learning/MalayalamOCR/datasetCreation/'+folderName	
    cv2.imwrite(os.path.join(path,'{}_imageresized_{}.jpg'.format(folderName,key)) ,imageresized)
    print 'imageresized_{}.jpg'.format(key)
