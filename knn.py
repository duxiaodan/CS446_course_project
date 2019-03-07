# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2
import glob
import torch
import sklearn.neighbors as skn

train64 = sorted(glob.glob('./train_images_64x64/*'))
train128 = sorted(glob.glob('./train_images_128x128/*'))
test64 = sorted(glob.glob('./test_images_64x64/*'))
num_patch = 16#64#64*64
patch_size = int(64/(num_patch)**0.5)
train64_np = np.zeros([len(train64),num_patch,patch_size**2*3], dtype='uint8')
train128_np = np.zeros([len(train128),num_patch,(patch_size*2)**2*3], dtype='uint8')
for i in range(len(train64)):
    train_l = cv2.imread(train64[i])
    train_h = cv2.imread(train128[i])
    temp_l =[]
    temp_h =[]
    for x in range(0,64//patch_size):
        for y in range(0,64//patch_size):
            temp_l.append(train_l[x*patch_size:x*patch_size+patch_size,y*patch_size:y*patch_size+patch_size])
            temp_h.append(train_h[x*patch_size*2:x*patch_size*2+patch_size*2,y*patch_size*2:y*patch_size*2+patch_size*2])
    train64_np[i] = np.asarray(temp_l).reshape(num_patch,patch_size**2*3)
    train128_np[i] = np.asarray(temp_h).reshape(num_patch,(patch_size*2)**2*3)
#    break

test64_np = np.zeros([len(test64),num_patch,patch_size**2*3], dtype='uint8')
for i in range(len(test64)):
    test_l = cv2.imread(test64[i])
    temp_l =[]
    for x in range(0,64//patch_size):
        for y in range(0,64//patch_size):
            temp_l.append(test_l[x*patch_size:x*patch_size+patch_size,y*patch_size:y*patch_size+patch_size])
    test64_np[i] = np.asarray(temp_l).reshape(num_patch,patch_size**2*3)
#    break
test128_np = np.zeros([len(test64),num_patch,patch_size*2*patch_size*2*3], dtype='uint8')
for i in range(num_patch):
    test128_np[:,i] = skn.KNeighborsClassifier(n_neighbors=1).fit(train64_np[:,i],train128_np[:,i]).predict(test64_np[:,i])
#    break
test128_np=test128_np.reshape(len(test128_np),num_patch,patch_size*2,patch_size*2,3).astype('uint8')


test128_output = np.zeros([len(test64),128,128,3], dtype='uint8')
#for i in range(len(test64)):
for j in range(num_patch):
    h = (j // int(num_patch**0.5))*patch_size*2
    w = (j % int(num_patch**0.5))*patch_size*2
    for y in range(patch_size*2):
        for x in range(patch_size*2):
            test128_output[:,h+y,w+x] = test128_np[:,j,y,x]
            

for i in range(len(test64)):
	cv2.imwrite('./output/test_' + '%05d'% (i+1) + '.png', test128_output[i])
