#!/usr/bin/env python3

import cv2
import cv2 as cv
import sys
import copy

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit('install Pillow to run this code')


def find_ball(opencv_image, debug=False):
    return ballFinders[5](opencv_image)

def median(k):
    def m(img):
        cimg = copy.deepcopy(img)
        return cv2.medianBlur(cimg,k)
    return m

def gaussian(ksize):
    def g(img):
        cimg = copy.deepcopy(img)
        return  cv2.GaussianBlur(cimg, (ksize, ksize), 0)
    return g

def hough(p1,p2):
    def h(img):
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,img.shape[0]/2, param1=p1,param2=p2, minRadius=0,maxRadius=0)
        return np.uint16(np.around(circles))[0,:]
    return h

def find_ball_with_params(opencv_image, p1, p2, blurrer, ksize):
    if blurrer == 'median':
        blurfn = median(ksize)
    else:
        blurfn = gaussian(ksize)
    return __find_ball(opencv_image, False, blurfn, hough(p1,p2))

def __find_ball(opencv_image, debug, blurfn, circlefn):
    def drawCircleOn(image,x,y,r):
        # draw the outer circle
        cv2.circle(image,(x,y),r,(0,255,0),2)
        # draw the center of the circle
        cv2.circle(image,(x,y),2,(0,0,255),3)

    try:
        img = blurfn(opencv_image)
        circles = circlefn(img)
        (x,y,r) = circles[0]
        if x==0 or y==0 or r==0:
            raise Exception("malformed circle")
        if debug:
            print("%d circles" % len(circles))
            drawCircleOn(img,x,y,r)
            display(img, "detected")
        return [x,y,r]
    except Exception as exc:
        if debug:
            print("no circles: ", exc)
        return None


def display(img, t):
    pil_image = Image.fromarray(img)
    pil_image.show(title=t)
    return img


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

def ballFinder(p1, p2, blurrer, ksize):
    def b(img):
        return find_ball_with_params(img, p1, p2, blurrer, ksize)
    return b

# autogrinder discovered 7 decent ball finders:
ballFinders = [
    ballFinder(160, 25, "median", 9),
    ballFinder(170, 25, "median", 9),
    ballFinder(180, 25, "median", 9),
    ballFinder(190, 25, "median", 9),
    ballFinder(170, 25, "gaussian", 3),
    ballFinder(180, 25, "gaussian", 3),
    ballFinder(30, 35,  "gaussian", 9)
]

if __name__ == "__main__":
    for f in ['test87.bmp']:
        opencv_image = cv2.imread("./imgs/%s" % f, cv2.COLOR_GRAY2RGB)
        ball = __find_ball(opencv_image, True, gaussian(3), hough(180,25))
        print(ball)
