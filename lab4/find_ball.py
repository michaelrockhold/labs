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

        Uses the a handpicked algorithm to blur the image, and some variant of Hough_Gradient
        to detect the circles
    """

    ball = None

    bf = ball_finder(compose(identity(),gaussian(ksize=7)), hough(200, 100))
    cs = bf(opencv_image)
    if cs is not None and len(cs) > 0:
        ball = cs[0]
    
    return ball


def display_circles(opencv_image, circles):
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
    
    def draw_circle(c,colour):
        # draw the outer circle
        cv2.circle(circle_image,(c[0],c[1]),c[2],colour,2)
        # draw the center of the circle
        cv2.circle(circle_image,(c[0],c[1]),2,(0,255,255),3) 
        # write coords
        cv2.putText(circle_image,str(c),(c[0],c[1]),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2,cv2.LINE_AA)            

    for c in circles[3:]:
        draw_circle(c, (255,255,0))
    
    #highlight the best circles in a different color
    for c in circles[:3]:
        draw_circle(c, (0,0,255))
    return circle_image

# detectors

def hough(p1, p2):
    def fn(greyimg):
        return cv2.HoughCircles(greyimg,cv2.HOUGH_GRADIENT,1,greyimg.shape[0]/8,param1=p1,param2=p2)
    return fn

# blur processors

def compose(a, b):
    def fn(img):
        return a(b(img))
    return fn

def identity():
    def fn(image):
        return image
    return fn

def inverter():
    def fn(image):
        return cv2.bitwise_not(image)
    return fn

def equalizer():
    def fn(img):
        return cv2.equalizeHist(img)
    return fn

def median(ksize=5):
    def fn(img):
        return cv2.medianBlur(img, ksize, img)
    return fn

def gaussian(ksize=5):
    def fn(img):
        # Remove noise by blurring with a Gaussian filter
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    return fn

def sobel(ksize=5):
    def fn(img):
        ddepth = cv2.CV_16S
        # Gradient-X
        grad_x = cv2.Sobel(img,ddepth,1,0,ksize = ksize, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8

        # Gradient-Y
        grad_y = cv2.Sobel(img,ddepth,0,1,ksize = ksize, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        return cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    return fn

def scharr(ksize=3):
    def fn(img):
        ddepth = cv2.CV_16S
        # Gradient-X
        grad_x = cv2.Scharr(img,ddepth,1,0)
        abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8

        # Gradient-Y
        grad_y = cv2.Scharr(img,ddepth,0,1)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        return cv2.add(abs_grad_x,abs_grad_y)
    return fn

def applyToCopy(fn, image):
    return fn(copy.deepcopy(image))

# utilities
def display(img):
    pil_image = Image.fromarray(img)
    pil_image.show()
    return img

def find_circles(detector, greyimg):
    try:
        circles = detector(greyimg)
        circles = np.uint16(np.around(circles))
        rt = circles[0,:]
    except Exception as exc:
        print(exc)
        rt = []
    return rt

def display_found_circles(img, processor, detector, debug):
    bf = ball_finder(processor, detector, debug=True)
    display(display_circles(img, bf(img)))

def ball_finder(processor, detector, debug=False):
    def fn(img):
        processedImg = applyToCopy(processor, img)
        if debug:
            display(processedImg)
        return find_circles(detector, processedImg)
    return fn


if __name__ == "__main__":
    test01 = "./imgs/test01.bmp"
    bike = "./devimgs/boston8lg2013.bmp"
    fourCircles = "./devimgs/Circle.jpg"
    inv = inverter()

    # best for test01: scharr(3)
    # best for fourCircles: gaussian(3) or (5)
    # very good for boston8lg2013: gaussian(ksize=5), hough(200, 100), ignore all but the first three results

    opencv_image = cv2.imread(test01, cv2.IMREAD_GRAYSCALE)
    
    display_found_circles(opencv_image, compose(inverter(),sobel(ksize=5)), hough(200, 100), True)
