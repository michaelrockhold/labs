#!/usr/bin/env python3

import cv2
import os
import math
import find_ball
import numpy as np

# load ground truth
with open('./imgs/ground_truth.txt') as f:
	grid_data = [i.split() for i in f.readlines()]

# thresh hold to accept circle and give credit per circle
center_err_thresh = 20.0
radius_err_thresh = 10.0

def grind(p1,p2,blurrer, ksize):
    score = 0;
    fails = []

    # check each image
    for filedata in grid_data:
        file = filedata[0]

        #read in image as grayscale
        opencv_image = cv2.imread("./imgs/" + file, cv2.COLOR_GRAY2RGB)

        #try to find the ball in the image
        ball = find_ball.find_ball_with_params(opencv_image, p1, p2, blurrer, ksize)
        if ball is None:
            ball = np.array([0, 0, 0])

        # get center err
        center_err = math.sqrt((ball[0] - float(filedata[1]))**2 + (
            ball[1] - float(filedata[2]))**2)

        # get radius err
        r_err = math.fabs(ball[2] - float(filedata[3]))

        if center_err <= center_err_thresh and r_err <= radius_err_thresh:
            score += 1;
        else:
            fails.append(file)
    if score == 50:
        result = 'OK'
    else:
        result = 'NOPE'
    print(("%s:  p1 %d p2 %d blurrer %s(%d) score = %d " % (result, p1, p2, blurrer, ksize, score)), " fails: ", fails)

for blurrer in ['median', 'gaussian']:
    for ksize in [3, 5, 7, 9]:
        for p1 in range(10,200,10):
            for p2 in range(10,60,5):
                grind(p1, p2, blurrer, ksize)
