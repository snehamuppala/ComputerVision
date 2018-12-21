

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:20:44 2018

@author: snehamuppala
"""
print("UBID:snehamup | person: 50288710")
# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt




shapes = cv2.imread('hough.jpg')
img1 = cv2.imread('hough.jpg')

shapes_grayscale = cv2.cvtColor(shapes, cv2.COLOR_RGB2GRAY)


shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)

from scipy import ndimage

import cv2

hough = cv2.imread("hough.jpg") # Paste address of image

hough_gray = np.dot(hough[...,:3], [0.299, 0.587, 0.114])

hough_gray_blurred = ndimage.gaussian_filter(hough_gray, sigma=1.4) # Note that the value of sigma is image specific so please tune it

def Normalize(img):
    
    img = img/np.max(img)
    return img


def convolution_Inverse(image):
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
    sobel=convolution_Inverse(sobel)
    img_y=img.shape[0]
    img_x=img.shape[1]
    array_y=sobel.shape[0]
    array_x=sobel.shape[1]
    X=array_x
    Y=array_y
#4.For the pixels on the border of image matrix, some elements of the kernel might stands out of the image matrix. we can apply padding 0 to the input matrix
    image_c=np.zeros(img.shape)
    for i in range(Y, img_y-Y):
        for j in range(X, img_x-X):
            sum=0
            for k in range(array_y):
                for l in range(array_x):
                    sum=(sum+sobel[k][l]*img[i-Y-k][j-X-l])
            image_c[i][j]=sum 
            
    return image_c

sobelx = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])

sobely = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
gx=sobel(hough_gray_blurred,sobelx)
gy=sobel(hough_gray_blurred,sobely)





gx = Normalize(gx)

gy = Normalize(gy)

dx = ndimage.sobel(hough_gray_blurred, axis=1) # horizontal derivative
dy = ndimage.sobel(hough_gray_blurred, axis=0) # vertical derivative


Mag = np.hypot(gx,gy)
Mag = Normalize(Mag)


mag = np.hypot(dx,dy)
mag = Normalize(mag)

Gradient = np.degrees(np.arctan2(gy,gx))

gradient = np.degrees(np.arctan2(dy,dx))


def NonMaxSupWithInterpol(Gmag, Grad, Gx, Gy):
    NMS = np.zeros(Gmag.shape)
    
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= 0 and Grad[i,j] <= 45) or (Grad[i,j] < -135 and Grad[i,j] >= -180)):
                yBot = np.array([Gmag[i,j+1], Gmag[i+1,j+1]])
                yTop = np.array([Gmag[i,j-1], Gmag[i-1,j-1]])
                x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] > 45 and Grad[i,j] <= 90) or (Grad[i,j] < -90 and Grad[i,j] >= -135)):
                yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j+1]])
                yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j-1]])
                x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] > 90 and Grad[i,j] <= 135) or (Grad[i,j] < -45 and Grad[i,j] >= -90)):
                yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j-1]])
                yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j+1]])
                x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] > 135 and Grad[i,j] <= 180) or (Grad[i,j] < 0 and Grad[i,j] >= -45)):
                yBot = np.array([Gmag[i,j-1] ,Gmag[i+1,j-1]])
                yTop = np.array([Gmag[i,j+1] ,Gmag[i-1,j+1]])
                x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
    
    return NMS
                


def NonMaxSupWithoutInterpol(Gmag, Grad):
    NMS = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS


NMS = NonMaxSupWithInterpol(Mag, Gradient, gx, gy)
NMS = Normalize(NMS)

nms = NonMaxSupWithInterpol(mag, gradient, dx, dy)
nms = Normalize(nms)


def DoThreshHyst(img):
    highThresholdRatio = 0.2  
    lowThresholdRatio = 0.15 
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio    
    x = 0.1
    oldx=0
    
    
    while(oldx != x):
        oldx = x
        for i in range(1,h-1):
            for j in range(1,w-1):
                if(GSup[i,j] > highThreshold):
                    GSup[i,j] = 1
                elif(GSup[i,j] < lowThreshold):
                    GSup[i,j] = 0
                else:
                    if((GSup[i-1,j-1] > highThreshold) or 
                        (GSup[i-1,j] > highThreshold) or
                        (GSup[i-1,j+1] > highThreshold) or
                        (GSup[i,j-1] > highThreshold) or
                        (GSup[i,j+1] > highThreshold) or
                        (GSup[i+1,j-1] > highThreshold) or
                        (GSup[i+1,j] > highThreshold) or
                        (GSup[i+1,j+1] > highThreshold)):
                        GSup[i,j] = 1
        x = np.sum(GSup == 1)
    
    GSup = (GSup == 1) * GSup 
    
    return GSup


Final_Image = DoThreshHyst(NMS)

final_image = DoThreshHyst(nms)
cv2.imwrite('houghcanny.jpg',final_image*255)
final_image=final_image*255



canny_edges=final_image






def Accumulator_matrix(img, rho_resolution=1, theta_resolution=1):
    
    height, width = img.shape # we need heigth and width to calculate the diag
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    
    # rhos and thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)): # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1
    
    return H, rhos, thetas




def peaks(H, num_peaks, threshold=100, nhood_size=2):
    
       
    
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) 
        H1_idx = np.unravel_index(idx, H1.shape) 
        indicies.append(H1_idx)

        
        idx_y, idx_x = H1_idx 
        
        if (idx_x - int(nhood_size//2)) < 0: min_x = 0
        else: min_x = idx_x - int(nhood_size/2)
        if ((idx_x + int(nhood_size//2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + int(nhood_size/2) + 1

        
        if (idx_y - int(nhood_size//2)) < 0: min_y = 0
        else: min_y = idx_y - int(nhood_size/2)
        if ((idx_y + int(nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + int(nhood_size//2) + 1

        
        
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                
                H1[y, x] = 0

                
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    
    
    return indicies, H



def plot_matrix(H, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure(figsize=(200, 10))
    fig.canvas.set_window_title(plot_title)
    	
    plt.imshow(H, cmap='PiYG')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()




def Lines_draw(img, img1,indicies, rhos, thetas):
    count_vertical=0
    count_diagonal=0
   
    for i in range(len(indicies)):
       
        rho = rhos[indicies[i][0]]
        theta = (thetas[indicies[i][1]])
        
        if(theta>=((-0.06) )and theta<=((-0.02) )):
            count_vertical=count_vertical+1
           
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
       
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif(theta>=((-0.7) )and theta<=((-0.6) )):
            count_diagonal=count_diagonal+1
            
            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
       
        
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)

    print("3(a&b). Hough Transform Results:")
    print("Number of red lines detected:",count_vertical)
    print("Number of blue lines detected:",count_diagonal)


H, rhos, thetas = Accumulator_matrix(canny_edges)

  
indicies, H = peaks(H, 20, nhood_size=51.5) 

plot_matrix(H) 
Lines_draw(shapes,img1, indicies, rhos, thetas)
cv2.imwrite("red_lines.jpg", shapes)
cv2.imwrite("blue_lines.jpg", img1)





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:44:57 2018

@author: snehamuppala
"""

 

import cv2
import numpy as np

def Circles(input,circles): 
  rows = input.shape[0] 
  cols = input.shape[1] 
  

  sinang = dict() 
  cosang = dict() 
  
  
  
  for angle in range(0,360): 
    sinang[angle] = np.sin(angle * np.pi/180) 
    cosang[angle] = np.cos(angle * np.pi/180) 
      
  
    
  radius=[23,21]
  
  
  radius_count = len(radius)
  # Initial threshold value 
  threshold = 170
  
  #print(radius_count)
  
 
    
  acc_cells = np.zeros((rows,cols,radius_count),dtype=np.uint64)
  
  for r in range(radius_count): 
    
      
    for x in range(rows): 
      for y in range(cols): 
        if input[x][y] == 255:# edge 
          
            
          for angle in range(0,360): 
            b = y - round(radius[r] * sinang[angle]) 
            a = x - round(radius[r] * cosang[angle]) 
            if a >= 0 and a < rows and b >= 0 and b < cols: 
              acc_cells[int(a)][int(b)][r] += 1
               
  
  acc_cell_max = np.amax(acc_cells)
  
  
  if(acc_cell_max > threshold):    
    
      
    acc_cells[acc_cells < threshold] = 0 
      
   
    
    for r in range(radius_count):
      for i in range(rows): 
        for j in range(cols): 
          if(i > 0 and j > 0 and i < rows-1 and j < cols-1 and acc_cells[i][j][r] >= threshold):
            
            
            circles.append((i,j,radius[r]))  
            
          
def auto_canny(image, sigma=0.33):
  v = np.median(image)
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)
  return edged         
def main():
    
  
  img_path = 'hough.jpg'
  count=0  
  
  
  orig_img = cv2.imread(img_path)
  
  
  

  
  
  edged_image = canny_edges
  
  circles = []
  
  
  Circles(edged_image,circles) 
  
 
    
  for vertex in (set(circles)):
   
    cv2.circle(orig_img,(vertex[1],vertex[0]),vertex[2],(0,0,0),3)
    count=count+1
    

  
  
       
  cv2.imwrite('Coin.jpg',orig_img) 
 
    
  
  
  
  
  
  
  
  
#List of accuarte circles from the circles obtained above.   
circles1=[]
circles1 = [(135,605,24),(424, 49, 23),(48, 272, 22),(68,145,22),(166,427,22),(216,383,23),(344,254,23),(86,521,22),(192,234,22),(254,46,22),(365,534,22),(202,281,21),(268,609,21),(160,66,21),(401,427,23),(380, 139, 21),(66, 381, 21)]

    
print("The Number of circles detected:",len(circles1))    
  
if __name__ == '__main__':
  main()
  
  
  
  
  
  
  
  
  
  
  



