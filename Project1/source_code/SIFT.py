#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:20:55 2018

@author: snehamuppala
"""

import numpy as np
import cv2
import math 


img = cv2.imread('/Users/snehamuppala/Desktop/computer_vision/project1/ques2/task.jpg',0)
#resizing the image for each octave.-->sampling 
pyr=[img]
g=img[::2,::2]
pyr.append(g)
g=g[::2,::2]
pyr.append(g)
g=g[::2,::2]
pyr.append(g)



#calulating gaussian_filter for each sigma value
#1. The scale space
def gaussian_filter(m,n,sig):
    g_filter=np.zeros((m,n))
    m=m//2
    n=n//2
    grid = np.array([[((i**2+j**2)/(2.0*sig**2)) for i in range(-m, m+1)] for j in range(-n, n+1)])
    g_filter = (np.exp(-grid)/(2*np.pi*sig**2))
    g_filter /= np.sum(g_filter)
    
    return g_filter
#flipping of image
def flipping(image):
    image_copy=image.copy()
    
    for i in  range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j]=image[image.shape[0]-i-1][image.shape[1]-j-1]
    return image_copy

#defineing funtion to perform convolution for image and kernel
def convolution(img,g):
    g=flipping(g)
    img_y=img.shape[0]
    img_x=img.shape[1]
    array_y=g.shape[0]
    array_x=g.shape[1]
    X=array_x
    Y=array_y

    image_c=np.zeros(img.shape)
    for i in range(Y, img_y-Y):
        for j in range(X, img_x-X):
            sum=0
            for k in range(array_y):
                for l in range(array_x):
                    sum=(sum+g[k][l]*img[i-Y-k][j-X-l])
            image_c[i][j]=sum 
    return image_c


    
   
def Final_gaussian(img,sig):
    g=gaussian_filter(7,7,sig)
    image=convolution(img,g)
    return image
      


    
 #For each octave, finding the gauusians and Dog   
#2. LoG approximations
def octave(img,sig):
    g1=Final_gaussian(img,sig[0])
    g2=Final_gaussian(img,sig[1])
    g3=Final_gaussian(img,sig[2])
    g4=Final_gaussian(img,sig[3])
    g5=Final_gaussian(img,sig[4])
     
    dog1=g2-g1
    dog2=g3-g2
    dog3=g4-g3
    dog4=g5-g4
    oct=[dog1,dog2,dog3,dog4]
    return oct
#initialinzing sigma values
sigma1=[(1/math.sqrt(2)),1,math.sqrt(2),2,2*(math.sqrt(2))]
sigma2=[math.sqrt(2),2,(2*math.sqrt(2)),4,(4*math.sqrt(2))]
sigma3=[(2*math.sqrt(2)),4,(4*math.sqrt(2)),8,(8*math.sqrt(2))]
sigma4=[(4*math.sqrt(2)),8,(8*math.sqrt(2)),16,(16*math.sqrt(2))]


#findimg the max and min values.
#3. Finding keypoints
def point(m1,m2,m3):
    row1,col1=m2.shape[0:2]
    key=np.zeros(m2.shape)
    for i in range(1,(row1-1)):
        for j in range(1,(col1-1)):
            if m2[i,j]==max(m2[i-1,j-1],m2[i-1,j],m2[i-1,j+1],m2[i,j],m2[i,j],m2[i,j+1],m2[i+1,j+1]):
                if m2[i,j]> max(m1[i-1,j-1],m1[i-1,j],m1[i-1,j+1],m1[i,j],m1[i,j],m1[i,j+1],m1[i+1,j+1]):
                    if m2[i,j]> max(m3[i-1,j-1],m3[i-1,j],m3[i-1,j+1],m3[i,j],m3[i,j],m3[i,j+1],m3[i+1,j+1]):
                        key[i][j]=255
                    else:
                        continue
            elif m2[i,j]==min(m2[i-1,j-1],m2[i-1,j],m2[i-1,j+1],m2[i,j],m2[i,j],m2[i,j+1],m2[i+1,j+1]):
                if m2[i,j]> min(m1[i-1,j-1],m1[i-1,j],m1[i-1,j+1],m1[i,j],m1[i,j],m1[i,j+1],m1[i+1,j+1]):
                    if m2[i,j]> min(m3[i-1,j-1],m3[i-1,j],m3[i-1,j+1],m3[i,j],m3[i,j],m3[i,j+1],m3[i+1,j+1]):
                        key[i][j]=255
                        
    return key
                
#Displaying the results and saving the images
points=[]        
oct1=octave(pyr[0],sigma1)

cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave1/taskdog1.jpg', oct1[0])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave1/taskdog2.jpg', oct1[1])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave1/taskdog3.jpg', oct1[2])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave1/taskdog4.jpg', oct1[3])




a=oct1[0]
b=oct1[1]
c=oct1[2]                       
d=oct1[3]                    
key1=point(a,b,c)  
key2=point(b,c,d) 
key3=key1+key2


cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave1/keypoints3.jpg', key3)
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave1/keypoints1.jpg', key1)
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave1/keypoinys2.jpg', key2)

cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', key1)
cv2.waitKey(0)
cv2.destroyAllWindows()  
oct2=octave(pyr[1],sigma2)
cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', key2)
cv2.waitKey(0)
cv2.destroyAllWindows() 




 
oct2=octave(pyr[1],sigma2)



cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave2/taskdog1.jpg', oct2[0])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave2/taskdog2.jpg', oct2[1])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave2/taskdog3.jpg', oct2[2])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave2/taskdog4.jpg', oct2[3])







a=oct2[0]
b=oct2[1]
c=oct2[2]                       
d=oct2[3]                    
key1=point(a,b,c)  
key2=point(b,c,d) 
key4=key1+key2

cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave2/keypoints3.jpg', key3)

cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave2/keypoints1.jpg', key1)
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave2/keypoinys2.jpg', key2)
cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', key1)
cv2.waitKey(0)
cv2.destroyAllWindows()  
oct2=octave(pyr[1],sigma2)
cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', key2)
cv2.waitKey(0)
cv2.destroyAllWindows() 

oct3=octave(pyr[2],sigma3)


  

cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave3/taskdog1.jpg', oct3[0])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave3/taskdog2.jpg', oct3[1])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave3/taskdog3.jpg', oct3[2])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave3/taskdog4.jpg', oct3[3])

a=oct3[0]
b=oct3[1]
c=oct3[2]                       
d=oct2[3]                    
key1=point(a,b,c)  
key2=point(b,c,d) 

key5=key1+key2

cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave3/keypoints3.jpg', key3)
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave3/keypoints1.jpg', key1)
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave3/keypoinys2.jpg', key2)
cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', key1)
cv2.waitKey(0)
cv2.destroyAllWindows()  
oct2=octave(pyr[1],sigma2)
cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', key2)
cv2.waitKey(0)
cv2.destroyAllWindows() 

 



oct4=octave(pyr[3],sigma4)
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave4/taskdog1.jpg', oct4[0])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave4/taskdog2.jpg', oct4[1])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave4/taskdog3.jpg', oct4[2])
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave4/taskdog4.jpg', oct4[3])

a=oct4[0]
b=oct4[1]
c=oct4[2]                       
d=oct4[3]                    
key1=point(a,b,c)  
key2=point(b,c,d) 

key6=key1+key2



cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave4/keypoints31.jpg', key6)
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave4/keypoints1.jpg', key1)
cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1/project_t2/octave4/keypoinys2.jpg', key2)
cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', key1)
cv2.waitKey(0)
cv2.destroyAllWindows()  
oct2=octave(pyr[1],sigma2)
cv2.namedWindow('edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('edge_x_dir', key2)
cv2.waitKey(0)
cv2.destroyAllWindows() 



    
    
