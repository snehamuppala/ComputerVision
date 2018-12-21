#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:01:21 2018

@author: snehamuppala
"""

#2a
print("UBID:snehamup | person: 50288710")



import numpy as np

import matplotlib.pyplot as plt1 
import cv2
img = cv2.imread('point.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('point.jpg')

edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype = np.float)


def Filpping(image):
    image_copy=image.copy()
    
    for i in  range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j]=image[image.shape[0]-i-1][image.shape[1]-j-1]
    
    return image_copy

def sobel(img,sobel):
    sobel=Filpping(sobel)
    img_y=img.shape[0]
    img_x=img.shape[1]
    array_y=sobel.shape[0]
    array_x=sobel.shape[1]
    X=array_x
    Y=array_y
#4.For the pixels on the border of image matrix, some elements of the kernel might stands out of the image matrix. 
#we can apply padding 0 to the input matrix
    image_c=np.zeros(img.shape)
    for i in range(Y, img_y-Y):
        for j in range(X, img_x-X):
            sum=0
            for k in range(X):
                for l in range(Y):
                    
                    sum=(sum+sobel[k][l]*img[i-Y-k][j-X-l])
                  
            image_c[i-4][j-4]=sum
           
                
            
    return image_c


pos=sobel(img,edge_kernel)
pos1=sobel(img,edge_kernel)
cv2.imwrite('point_res1.jpg', pos1)
img_y=pos.shape[0]
img_x=pos.shape[1]

for i in range( img_y):
    for j in range(img_x):
        if(abs(pos[i][j])>1000):
           
            print("The porosity co-ordinates are::",i,j)
            
            pos[i][j]=255
        else:
            pos[i][j]=0
            
        

cv2.imwrite('point_res2.jpg', pos)





plt1.imshow(img2)

label = (445, 249)
plt1.text(445, 249, label)

fig1 = plt1.gcf()
#plt.draw()

fig1.savefig('point_res3.jpg',dpi=100)




