#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:50:50 2018

@author: snehamuppala
"""

import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('/Users/snehamuppala/Downloads/proj1_cse573/task1.png',0)
#initialized the sobel matrix for x and y axis.
sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)

#Flip the kernel both horizontally and vertically. 
#As our selected kernel is symetric, the flipped kernel is equal to the original
def Filpping(image):
    image_copy=image.copy()
    
    for i in  range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j]=image[image.shape[0]-i-1][image.shape[1]-j-1]
    
    return image_copy
#1.Put the first element of the kernel at every pixel of the image (element of the image matrix). 
#Then each element of the kernel will stand on top of an element of the image matrix.
#2.Multiply each element of the kernel with its corresponding element of the image matrix (the one which is overlapped with it)
#3.Sum up all product outputs and put the result at the same position in the output matrix as the center of kernel in image matrix.

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
            for k in range(array_y):
                for l in range(array_x):
                    sum=(sum+sobel[k][l]*img[i-Y-k][j-X-l])
            image_c[i][j]=sum 
            
    return image_c
gx=sobel(img,sobelx)
gy=sobel(img,sobely)
#normalisation by eliminating negative values. 
pos_edge_x = np.abs(gx) / np.max(np.abs(gx))
pos_edge_y = np.abs(gy) / np.max(np.abs(gy))

cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', pos_edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/task1_x.png', pos_edge_x)
cv2.namedWindow('edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/task1_y.png', pos_edge_y)
