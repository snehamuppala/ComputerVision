#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:26:17 2018

@author: snehamuppala
"""
print("UBID:snehamup | person: 50288710")
import cv2
import numpy as np


img = cv2.imread('noise.jpg',cv2.IMREAD_GRAYSCALE)


kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = np.float)
def Dilation(img,sobel):
   
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
            for k in range(array_y):
                for l in range(array_x):
                    
                    sum=(sum+sobel[k][l]*img[i-Y-k][j-X-l])
            image_c[i][j]=sum 
            
    return image_c
dilate=Dilation(img,kernel)
dilate[dilate > 0] = 255

def Erosion(input_array, structure=np.ones((3,3)).astype(np.bool)):
    
    rows, cols = input_array.shape
    
    pad_shape = (
        input_array.shape[0] + structure.shape[0] - 1, 
        input_array.shape[1] + structure.shape[1] - 1)
    input_pad_array = np.zeros(pad_shape).astype(np.bool)
    input_pad_array[1:rows+1,1:cols+1] = input_array
    binary_erosion = np.zeros(pad_shape).astype(np.bool)
     
   
    struc_mask = structure.astype(np.bool)
     
    
    for row in range(rows):
        for col in range(cols):
            
            binary_erosion[row+1,col+1] = np.min(
                input_pad_array[row:row+3, col:col+3][struc_mask])
    return binary_erosion[1:rows+1,1:cols+1]
b=Erosion(img, kernel).astype(np.int)
erosion=b*255


bou=img-erosion

bou1=img-dilate



#Closing and opening

#closing--dilation --->erosion
closing=Erosion(dilate, kernel).astype(np.int)
closing=(closing*255)
#cv2.imwrite('closing.jpg', closing)
#opening--->erosion--->dilation
 #(A!B)+B
 
opening=Erosion(closing,kernel).astype(np.int)
opening=opening*255
opening1=Dilation(opening,kernel)
opening1[opening1 > 0] = 255
cv2.imwrite('res_noise2.jpg', opening1)




opening_bou=Erosion(opening1,kernel).astype(np.float)

opening_bou=(opening_bou*255)

bou=opening1-opening_bou
cv2.imwrite('res_bound2.jpg', bou)


#opening and closing
#opening--->erosion--->dilation
opening=Erosion(img,kernel).astype(np.int)
opening=opening*255
opening1=Dilation(opening,kernel)
opening1[opening1 > 0] = 255


closing=Dilation(opening1,kernel)
closing[closing > 0] = 255
closing=Erosion(closing, kernel).astype(np.int)
closing=(closing*255)

cv2.imwrite('res_noise1.jpg', closing)


closing_bou=Erosion(closing,kernel).astype(np.float)

closing_bou=(closing_bou*255)

bou=closing-closing_bou
cv2.imwrite('res_bound1.jpg', bou)





