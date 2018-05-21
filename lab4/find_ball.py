#!/usr/bin/env python3

import cv2
import sys
import copy

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit('install Pillow to run this code')


def find_ball(opencv_image, debug=False):
    """Find the ball in an image.
        
        Arguments:
        opencv_image -- the image
        debug -- an optional argument which can be used to control whether
                debugging information is displayed.
        
        Returns [x, y, radius] of the ball, and [0,0,0] or None if no ball is found.
    """

    ball = None
    
    ## TODO: INSERT YOUR SOLUTION HERE
    ball = [152, 94, 42]
    
    return ball


def display_circles(opencv_image, circles, best=None):
    """Display a copy of the image with superimposed circles.
        
       Provided for debugging purposes, feel free to edit as needed.
       
       Arguments:
        opencv_image -- the image
        circles -- list of circles, each specified as [x,y,radius]
        best -- an optional argument which may specify a single circle that will
                be drawn in a different color.  Meant to be used to help show which
                circle is ranked as best if there are multiple candidates.
        
    """
    #make a copy of the image to draw on
    circle_image = copy.deepcopy(opencv_image)
    circle_image = cv2.cvtColor(circle_image, cv2.COLOR_GRAY2RGB, circle_image)
    
    for c in circles:
        # draw the outer circle
        cv2.circle(circle_image,(c[0],c[1]),c[2],(255,255,0),2)
        # draw the center of the circle
        cv2.circle(circle_image,(c[0],c[1]),2,(0,255,255),3) 
        # write coords
        cv2.putText(circle_image,str(c),(c[0],c[1]),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2,cv2.LINE_AA)            
    
    #highlight the best circle in a different color
    if best is not None:
        # draw the outer circle
        cv2.circle(circle_image,(best[0],best[1]),best[2],(0,0,255),2)
        # draw the center of the circle
        cv2.circle(circle_image,(best[0],best[1]),2,(0,0,255),3) 
        # write coords
        cv2.putText(circle_image,str(best),(best[0],best[1]),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2,cv2.LINE_AA)            
        
    display(circle_image)

def median_blur(inputImg, ksize=5):
    img = copy.deepcopy(inputImg)
    img = cv2.medianBlur(img, ksize, img)
    return img

def gaussian_blur(inputImg, ksize=5):
    img = copy.deepcopy(inputImg)
    # Remove noise by blurring with a Gaussian filter
    img = cv2.GaussianBlur(img, (ksize, ksize), 0, img)
    return img

def sobel_blur(inputImg, ksize=5):
    img = copy.deepcopy(inputImg)
    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U. Twice.
    x64fImage = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=ksize)
    # ...take its absolute...
    absx64fImage = np.absolute(x64fImage)
    # ...convert to cv2.CV_8U.
    xu8Image = np.uint8(absx64fImage)
    # And again:
    y64fImage = cv2.Sobel(xu8Image,cv2.CV_64F,0,1,ksize=ksize)
    # ...take its absolute...
    absy64fImage = np.absolute(y64fImage)
    # ...convert to cv2.CV_8U.
    img = np.uint8(absy64fImage)
    return img

def display(img):
    pil_image = Image.fromarray(img)
    pil_image.show()
    return img

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def find_circles(greyimg):
    try:
        circles = cv2.HoughCircles(greyimg,cv2.HOUGH_GRADIENT,1,greyimg.shape[0]/2,param1=150,param2=100)
        circles = np.uint16(np.around(circles))
        rt = circles[0,:]
    except Exception as exc:
        print(exc)
        rt = []
    return rt

def draw_circles(greyimg, circles):
    rgbImage = copy.deepcopy(greyimg)
    rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_GRAY2BGR, rgbImage)
    for i in circles:
        # draw the outer circle
        cv2.circle(rgbImage,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(rgbImage,(i[0],i[1]),2,(0,0,255),3)
    return rgbImage

def thresh(img):
    bigmask = cv2.compare(img,np.uint8([127]),cv2.CMP_GE)
    smallmask = cv2.bitwise_not(bigmask)
    x = np.uint8([90])
    big = cv2.add(img,x,mask = bigmask)
    small = cv2.subtract(img,x,mask = smallmask)
    res = cv2.add(big,small)
    return res

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    return clahe.apply(opencv_image)

if __name__ == "__main__":
    # testImagePath = "./imgs/tdest01.bmp"
    testImagePath = "./devimgs/Circle.jpg"
    # testImagePath = "/Users/michael/Downloads/boston8lg2013.bmp"

    opencv_image = cv2.imread(testImagePath, cv2.IMREAD_GRAYSCALE)
    processedImg = copy.deepcopy(opencv_image)

    # processedImg = cv2.equalizeHist(processedImg)
    # processedImg = median_blur(processedImg, 3)
    # processedImg = gaussian_blur(processedImg, 3)
    processedImg = sobel_blur(processedImg, 5)                                                                                                                                                                                                                                                                                                                                                                

    display(processedImg)

    cs = find_circles(processedImg)
    display(draw_circles(opencv_image, cs))

# best for test01: equalize, median_blur(3), sobel_blur(3)
# best for boston8lg2013: none
# best for 
