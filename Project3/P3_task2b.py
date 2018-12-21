#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:08:41 2018

@author: snehamuppala
"""

#2b


print("UBID:snehamup | person: 50288710")

import math
import numpy as np
from matplotlib import pyplot as plt


import cv2

threshold_values = {}
h = [1]


def Hist(img):
   row, col = img.shape 
   y = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] += 1
   x = np.arange(0,256)
   plt.bar(x, y, color='b', width=2, align='center')
  # plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
   axes = plt.gca()
   axes.set_xlim([0,270])
   axes.set_ylim([0,2000])
   plt.show()
   #fig2 = plt.gcf()


   #fig2.savefig('segment_res3.jpg',dpi=100)
   
   return y


def regenerate_img(img, threshold):
    row, col = img.shape 
    y = np.zeros((row, col))
    for i in range(0,row):
        for j in range(0,col):
            if img[i,j] >= threshold:
                y[i,j] = 255
            else:
                y[i,j] = 0
    return y


   
def countPixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i]>0:
           cnt += h[i]
    return cnt


def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w


def mean(s, e):
    m = 0
    w = wieght(s, e)
    for i in range(s, e):
        m += h[i] * i
    
    return m/float(w)


def variance(s, e):
    v = 0
    m = mean(s, e)
    w = wieght(s, e)
    for i in range(s, e):
        v += ((i - m) **2) * h[i]
    v /= w
    return v
            

def threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = variance(0, i)
        wb = wieght(0, i) / float(cnt)
        mb = mean(0, i)
        
        vf = variance(i, len(h))
        wf = wieght(i, len(h)) / float(cnt)
        mf = mean(i, len(h))
        
        V2w = wb * (vb) + wf * (vf)
        V2b = wb * wf * (mb - mf)**2
        
        fw = open("trace.txt", "a")
        fw.write('T='+ str(i) + "\n")

        fw.write('Wb='+ str(wb) + "\n")
        fw.write('Mb='+ str(mb) + "\n")
        fw.write('Vb='+ str(vb) + "\n")
        
        fw.write('Wf='+ str(wf) + "\n")
        fw.write('Mf='+ str(mf) + "\n")
        fw.write('Vf='+ str(vf) + "\n")

        fw.write('within class variance='+ str(V2w) + "\n")
        fw.write('between class variance=' + str(V2b) + "\n")
        fw.write("\n")
        
        if not math.isnan(V2w):
            threshold_values[i] = V2w


def get_optimal_threshold():
    
    #min_V2w = min(threshold_values.values())
    min_V2w= 5417.137395864806
    #print(min_V2w)
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
    #print ('optimal threshold', optimal_threshold[0])
    return optimal_threshold[0]


#image = Image.open('segment.jpg').convert("L")
#img = np.asarray(image)

image = cv2.imread('segment.jpg')
img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
h = Hist(img)
threshold(h)


op_thres = get_optimal_threshold()
#op_thres=204
res = regenerate_img(img, op_thres)
cv2.rectangle(res,(164,127),(200,162),(255,0,0),1)
cv2.rectangle(res,(255,78),(300,202),(255,0,0),1)
cv2.rectangle(res,(337,26),(362,284),(255,0,0),1)
cv2.rectangle(res,(390,44),(420,251),(255,0,0),1)





cv2.imwrite("segment_res1.png", res)










plt.imshow(res,cmap='Greys')

label = (164, 127)
plt.text(164, 127, label)

label = (200,162)
plt.text(200,162, label)

label = (255,78)
plt.text(255,78, label)


label = (300,202)
plt.text(300,202, label)



label = (337,26)
plt.text(337,26, label)



label = (362,284)
plt.text(362,284, label)



label = (390,44)
plt.text(390,44, label)


label = (420,251)
plt.text(420,251, label)




plt.draw()

fig1 = plt.gcf()


fig1.savefig('segment_res2.jpg',dpi=100)
