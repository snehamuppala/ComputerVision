#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 02:50:04 2018

@author: snehamuppala
"""

import numpy as np
import cv2
import os
import glob


def template(image):
     # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #loading template image in gray
  
    template = cv2.imread('/Users/snehamuppala/Desktop/computer_vision/project1/ques3/task3/template/te.png',0)
    # resize images
    template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
    #reisze the image in the range of 0.5-0.9, by varing almost all templates are matched
#Perform GaussianBlur for image
#and apply Laplacian on GaussianBlur
    gausBlur = cv2.GaussianBlur(image, (3,3),0)  #change the gaussian for more accurate match.
    imageGray = cv2.Laplacian(gausBlur,cv2.CV_32F)
#apply Laplacian on template image
    templateGray = cv2.Laplacian(template,cv2.CV_32F)

# Find template
#Algorithm used is cv2.TM_CCOEFF
 #  5.	This program takes a “sliding window” of our cursor query image and slides it across our original image from 
 #left to right and top to bottom, one pixel at a time. Then, for each of these locations, we compute the correlation
# coefficient to determine how “good” or “bad” the match is. Regions with sufficiently high correlation can be considered
# “matches” for our cursor template.
    Match = cv2.matchTemplate(imageGray,templateGray, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(Match)
    top_left = max_loc
    h,w = templateGray.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(image_rgb,top_left, bottom_right,(193,182,255),4)
 
# Show result
#cv2.imshow("Template", template)
    cv2.imshow("Match", image_rgb)
    cv2.imwrite('/Users/snehamuppala/Desktop/computer_vision/project1//ques3/output.jpg', image_rgb)
    cv2.moveWindow("Match", 150, 50);
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 #traversing through all points.
path = '/Users/snehamuppala/Desktop/computer_vision/project1/ques3/task3/bonus/t3/'
for infile in glob.glob( os.path.join(path, '*.jpg') ):
    print ("current file is: " + infile) 
    image_rgb=cv2.imread(infile)
    template(image_rgb)
    



