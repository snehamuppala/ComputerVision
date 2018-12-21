
import cv2
import numpy as np


UBIT = 'snehamup'
np.random.seed(sum([ord(c) for c in UBIT]))
#load images
#task1
image_mountain1=cv2.imread("mountain1.jpg", cv2.IMREAD_GRAYSCALE)
image_mountain2=cv2.imread("mountain2.jpg", cv2.IMREAD_GRAYSCALE)
image_mountain1_color=cv2.imread("mountain1.jpg")
image_mountain2_color=cv2.imread("mountain2.jpg")
sift=cv2.xfeatures2d.SIFT_create()
keypoints,descriptors=sift.detectAndCompute(image_mountain1,None)
keypoints1,descriptors1=sift.detectAndCompute(image_mountain2,None)
image_mountain1=cv2.drawKeypoints(image_mountain1_color,keypoints,None)
image_mountain2=cv2.drawKeypoints(image_mountain2_color,keypoints1,None)
cv2.imwrite("task1_sift1.jpg",image_mountain1)
cv2.imwrite("task1_sift2.jpg",image_mountain2)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=2)


flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptors, descriptors1, k=2)

matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        matchesMask[i] = [1, 0]
        


draw_params = dict(matchColor=(0, 200, 0),
                   singlePointColor=(000, 0, 0),
                   matchesMask=matchesMask,
                   flags=2)


img3 = cv2.drawMatchesKnn(image_mountain1_color, keypoints, image_mountain2_color, keypoints1, matches, None, **draw_params)
cv2.imwrite("task1_matches_knn.jpg",img3)

def panoTwoImages(img1,img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result



FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=2)


flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptors, descriptors1, k=2)
MIN_MATCH_COUNT = 10
good = []
for m,n in matches:
   if m.distance < 0.75*n.distance:
        good.append(m)
        
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keypoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)




    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    print("Homography:",H)
    
    il=np.where(np.asarray(mask.ravel())==1)[0]
    rin=np.random.choice(il,10)
    
    good=[good[i] for i in rin ]
    matchesMask=[matchesMask[i] for i in rin ]
    good=np.asarray(good)

    h,w = image_mountain1.shape[:-1]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)
    result1 = panoTwoImages(image_mountain2_color, image_mountain1_color, H)


else:
    print ("No good match - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor = (0,200,0), 
                   singlePointColor =(000, 0, 0),
                   matchesMask = matchesMask, 
                   flags = 2)
img3 = cv2.drawMatches(image_mountain1_color,keypoints,image_mountain2_color,keypoints1,good,None,**draw_params)
cv2.imwrite("task1_matches.jpg",img3)
cv2.imwrite("task1_pano.jpg",result1)

#task2
tsucuba_left=cv2.imread("tsucuba_left.png", cv2.IMREAD_GRAYSCALE)
tsucuba_right=cv2.imread("tsucuba_right.png", cv2.IMREAD_GRAYSCALE)
tsucuba_left_color=cv2.imread("tsucuba_left.png")
tsucuba_right_color=cv2.imread("tsucuba_right.png")
sift=cv2.xfeatures2d.SIFT_create()
keypoints,descriptors=sift.detectAndCompute(tsucuba_left,None)
keypoints1,descriptors1=sift.detectAndCompute(tsucuba_right,None)
tsucuba_left=cv2.drawKeypoints(tsucuba_left_color,keypoints,None)
tsucuba_right=cv2.drawKeypoints(tsucuba_right_color,keypoints1,None)
cv2.imwrite("task2_sift1.jpg",tsucuba_left)
cv2.imwrite("task2_sift2.jpg",tsucuba_right)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=2)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(descriptors, descriptors1, k=2)

matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        matchesMask[i] = [1, 0]
        


draw_params = dict(matchColor=(0, 200, 0),
                   singlePointColor=(000, 0, 0),
                   matchesMask=matchesMask,
                   flags=2)


img3 = cv2.drawMatchesKnn(tsucuba_left_color, keypoints, tsucuba_right_color, keypoints1, matches, None, **draw_params)
cv2.imwrite("task2_matches_knn.jpg",img3)
img1=cv2.imread("tsucuba_left.png", cv2.IMREAD_GRAYSCALE)
img2=cv2.imread("tsucuba_right.png", cv2.IMREAD_GRAYSCALE)
img1_color=cv2.imread("tsucuba_left.png")
img2_color=cv2.imread("tsucuba_right.png")


sift=cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=10)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
     
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)


F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC,5.0)
print("Fundamental",F)
# We select only inlier points
pts11 = pts1[mask.ravel()==1]
pts21 = pts2[mask.ravel()==1]
il=np.where(np.asarray(mask.ravel())==1)[0]
rin=np.random.choice(il,10)

pts1=[pts1[i] for i in rin ]
pts2=[pts2[i] for i in rin ]
pts1=np.asarray(pts1)
pts2=np.asarray(pts2)
def drawlines(img1,img2,lines,pts1,pts2):
    
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    color11=[(173,255,47),(0,100,0),(0,255,0),(255,0,0),(0,244,234),(222,0,123),(220,234,123),(244,0,244),(100,100,100),(222,22,222)]
    #for i in color11:
    i = 0
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        
            #color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color11[i],1)
        img1 = cv2.circle(img1,tuple(pt1),5,color11[i],-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color11[i],-1)
        i=i+1
            
            
        
    return img1,img2
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
#


 
# create an index counter to avoid problems with identical values

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)

img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
cv2.imwrite("task2_epi_left.jpg",img5)
cv2.imwrite("task2_epi_right.jpg",img3)


imgL = cv2.imread('tsucuba_left.png')  
imgR = cv2.imread('tsucuba_right.png')

# SGBM Parameters -----------------
window_size = 5                  
 
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*3,            
    blockSize=5,
    P1=8 * 5 * window_size ** 2,    
    P2=32 * 5 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
filter1 = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
filter1.setLambda(lmbda)
filter1.setSigmaColor(sigma)

displ = left_matcher.compute(imgL, imgR)
dispr = right_matcher.compute(imgR, imgL)
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = filter1.filter(displ, imgL, None, dispr)  
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
cv2.imwrite('task2_disparity.jpg', filteredImg)

from copy import deepcopy
import numpy as np

from matplotlib import pyplot as plt



k = 3
# Getting the values and plotting it
f1=[5.9,4.6,6.2,4.7,5.5,5.0,4.9,6.7,5.1,6.0]
f2=[3.2,2.9,2.8,3.2,4.2,3.0,3.1,3.1,3.8,3.0]
C_x=[6.2,6.6,6.5]
C_y=[3.2,3.7,3.0]



X = np.array(list(zip(f1, f2)))
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)




C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))

error = dist(C, C_old, None)
print(error)
colors = ['r', 'g', 'b', 'y', 'c', 'm']

def plot(clusters,count):
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        
        plt.scatter(points[:, 0], points[:, 1], s=100, c=colors[i],marker='^')
    for  x, y in zip(f1,f2):
        label = '(%s, %s)' % (x, y)
        plt.text(x, y, label)
    #plt.show()
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    print("Iteration::",count)
    fig1.savefig('task3_iter'+str(count)+'_a.jpg', dpi=100)
   
def plot_M(C,count1):
    c1=C[:, 0]
    c2=C[:, 1]

    for i in range(k):
        plt.plot(c1[i], c2[i], color=colors[i], marker='o',ls='None')
    for  x, y in zip(c1,c2):
        label = '(%s, %s)' % (x, y)
        plt.text(x, y, label)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    
    fig1.savefig('task3_iter'+str(count1)+'_b.jpg', dpi=100)
   
count=0
count1=0

while error != 0:
    
    
    for i in range(len(X)):
        distances = dist(X[i], C)
        
        
        cluster = np.argmin(distances)
       
        clusters[i] = cluster
    count=count+1
    print("Classification vector:")
    print(clusters)
    plot(clusters,count)
   
    
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        
        
        C[i] = np.mean(points, axis=0)
        
    count1=count1+1
    print("Updated Centroid:",C)
        #print(C)
    plot_M(C,count1)
    error = dist(C, C_old, None)
    

import numpy as np

import cv2
import matplotlib.pyplot as plt
import numpy.matlib
image=cv2.imread('baboon.jpg')

image = image[:, :, ::-1]


rows = image.shape[0]
cols = image.shape[1]
image = image/255
X = image.reshape(image.shape[0]*image.shape[1],3)

K = [3,5,10,20] 
iters = 20 
import random
def Mu(X,K):
    c = random.sample(list(X),K)
    return c



def Closest_Mu(X,c):
    K = np.size(c,0)
    idx = np.zeros((np.size(X,0),1))
    arr = np.empty((np.size(X,0),1))
    for i in range(0,K):
        y = c[i]
        temp = np.ones((np.size(X,0),1))*y
        b = np.power(np.subtract(X,temp),2)
        a = np.sum(b,axis = 1)
        a = np.asarray(a)
        a.resize((np.size(X,0),1))
        
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr,0,axis=1)
    idx = np.argmin(arr, axis=1)
    return idx

def calculate_Mu(X,idx,K):
    n = np.size(X,1)
    centroids = np.zeros((K,n))
    for i in range(0,K):
        ci = idx==i
        ci = ci.astype(int)
        total_number = sum(ci);
        ci.resize((np.size(X,0),1))
        total_matrix = np.matlib.repmat(ci,1,n)
        ci = np.transpose(ci)
        total = np.multiply(X,total_matrix)
        centroids[i] = (1/total_number)*np.sum(total,axis=0)
    return centroids

def KMean_Calculation(X,initial_centroids,iters):
    m = np.size(X,0)
    n = np.size(X,1)
    K = np.size(initial_centroids,0)
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m,1))
    for i in range(1,iters):
        idx = Closest_Mu(X,centroids)
        centroids = calculate_Mu(X,idx,K)
    return centroids,idx

for i in K:
    
    initial_centroids = Mu(X,i)
    centroids,idx = KMean_Calculation(X,initial_centroids,iters)
    idx.resize((np.size(X,0),1))
   
    idx = Closest_Mu(X,centroids)
    X_recovered = centroids[idx]
   
    X_recovered = np.reshape(X_recovered, (rows, cols, 3))
    
    pixels1 = np.array(X_recovered)
    plt.imsave('task3_baboon_'+str(i)+'.jpg', X_recovered)
    


        
        
    
    
    













