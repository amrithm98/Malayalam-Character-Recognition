# -*- coding: utf-8 -*-
import os,glob
import cv2
import unicodedata
import sys
print sys.getdefaultencoding()
s=u'കക'
b=s.encode('utf-8').decode('utf-8')
print(b)
from resize import process
dir_names=(os.listdir('/home/amrith/Machine-Learning/MalayalamOCR/lekha-OCR-database/train_images/'))
actual_list=[]
for i in dir_names:
	#s='u'+'''+'''+i
	#print i
	#b=i.encode('utf-8').decode('utf-8')
	actual_list+=[b]
for folder in dir_names:
	os.mkdir(folder)
	for (i,image_file) in enumerate(glob.iglob('/home/amrith/Machine-Learning/MalayalamOCR/lekha-OCR-database/train_images/'+folder+'/*.png')):
		process(image_file, i,folder)
