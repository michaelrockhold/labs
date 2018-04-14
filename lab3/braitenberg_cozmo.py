#!/usr/bin/env python3

'''Make Cozmo behave like a Braitenberg machine with virtual light sensors and wheels as actuators.
'''

import asyncio
import time
import cozmo
import cv2 as cv2
import numpy as np
import sys
import math

max_speed = 50.0

def sense_brightnesses(image):
	'''Return the left,right pair of sensor values,
	which are the relative proportions of pixels in each "sensor"
	that exceed the threshhold value. We also enhance the contrast
	by reducing whichever one is lower by ten percent, while increasing
	the reported value for the other by the same amount.
	'''

	def sense_brightness(image, leftmost_x, sensor_width, normaliser):
		'''Return a brightness value for the region of the image defined in sensor_region
		as a ratio of the region's average brightness to the maximum brightness possible.
		We normalise the pixel's value to values between 0 and 1, based on the darkest
		and lightest pixels found in the image.
		'''
		h = image.shape[0]

		average_brightness = 0.0
		for y in range(0, h):
			for x in range(leftmost_x, leftmost_x+sensor_width):
				average_brightness += normaliser(image[y,x])
		average_brightness /= (sensor_width * h)
		return average_brightness

	min_value = image.flatten().min()
	max_value = image.flatten().max()
	rng = max_value - min_value
	if rng == 0:
		return min_value, min_value

	def normal(pixel_value):
		return 1.0 * (pixel_value - min_value) / rng

	## here we treat the camera as two halves: left sensor, right sensor
	sensor_width = int(image.shape[1]/2)
	sensor_left = sense_brightness(image, 0, sensor_width, normal)
	sensor_right = sense_brightness(image, sensor_width, sensor_width, normal)
	return sensor_left, sensor_right



def sense_brightnesses2(image):
	def sense_brightness(image, leftmost_x, sensor_width):
		h = image.shape[0]

		aggregate_brightness = 0.0
		for y in range(0, h):
			for x in range(leftmost_x, leftmost_x+sensor_width):
				aggregate_brightness += pow(image[y,x], 1)
		return aggregate_brightness

	def apply_sensor_threshhold(value):
		return 0 if value < 0.05 else value

	## here we treat the camera as two halves: left sensor, right sensor
	maximum = pow(255.0, 1) * image.shape[0] * image.shape[1] / 2
	sensor_width = int(image.shape[1]/2)
	sensor_left = sense_brightness(image, 0, sensor_width)
	sensor_right = sense_brightness(image, sensor_width, sensor_width)
	total = sensor_left + sensor_right
	return apply_sensor_threshhold(sensor_left/maximum), apply_sensor_threshhold(sensor_right/maximum)




def ipsilateral(f, left_sensor_value, right_sensor_value):
	'''Maps a sensor reading to a wheel motor command'''
	return f(left_sensor_value), f(right_sensor_value)

def contralateral(f, left_sensor_value, right_sensor_value):
	'''Maps a sensor reading to a wheel motor command'''
	return f(right_sensor_value), f(left_sensor_value)



def positive(s):
	return max_speed * s + 32

def inhibitory(s):
	return max_speed * (1.0 - s) - 32

def fear(left_sensor_value, right_sensor_value):
	return ipsilateral(positive, left_sensor_value, right_sensor_value)

def aggression(left_sensor_value, right_sensor_value):
	return contralateral(positive, left_sensor_value, right_sensor_value)

def liking(left_sensor_value, right_sensor_value):
	return ipsilateral(inhibitory, left_sensor_value, right_sensor_value)

def love(left_sensor_value, right_sensor_value):
	return contralateral(inhibitory, left_sensor_value, right_sensor_value)



def log(left_sensor, right_sensor, left_motor, right_motor):
	light_threshhold = 0
	if left_sensor > right_sensor + light_threshhold:
		light_dir = "ON THE LEFT"
	elif left_sensor < right_sensor - light_threshhold:
		light_dir = "ON THE RIGHT"
	else:
		light_dir = "AHEAD"

	motor_threshhold = 0
	if left_motor > right_motor + motor_threshhold:
		go_dir = "TURN RIGHT"
	elif left_motor < right_motor - motor_threshhold:
		go_dir = "TURN LEFT"
	else:
		go_dir = "GO STRAIGHT"

	print("-")
	print(f"  {left_sensor},{right_sensor}: LIGHT {light_dir}")
	print(f"  {left_motor},{right_motor}:  {go_dir}")



def makeDriver(strategy):
	async def braitenberger(robot: cozmo.robot.Robot):
		'''The core of the braitenberg machine program'''

		# Move lift down and tilt the head up
		robot.move_lift(-3)
		robot.set_head_angle(cozmo.robot.util.degrees(0)).wait_for_completed()
		print("Press CTRL-C to quit")

		print("sensor_right,sensor_left,  motor_right,motor_left")

		while True:
			
			#get camera image
			event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

			#convert camera image to opencv format
			opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)

			# Sense the current brightness values on the right and left of the image.
			(sensor_left, sensor_right) = sense_brightnesses2(opencv_image)

			# Map the sensors to actuators using the supplied strategy
			(motor_left, motor_right) = strategy(sensor_left, sensor_right)

			log(sensor_left, sensor_right, motor_left, motor_right)

			# Send commands to the robot
			await robot.drive_wheels(motor_left, motor_right)

			time.sleep(.1)
	return braitenberger

def go(behaviour):
    cozmo.camera.Camera.enable_auto_exposure = False
    cozmo.run_program(makeDriver(behaviour), use_viewer=True, force_viewer_on_top=True)

go(love)
