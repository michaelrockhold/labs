#!/usr/bin/env python3

'''Make Cozmo behave like a Braitenberg machine with virtual light sensors and wheels as actuators.

The following is the starter code for lab.
'''

import asyncio
import time
import cozmo
import cv2
import numpy as np
import sys
import math

max_speed = 70.0

def sense_brightnesses(image):
	
	def sense_brightness(image, sensor_region):
		'''Maps a sensor reading to a wheel motor command
		Return the brightness of sensor_region as 0..1
		'''
		h = image.shape[0]
		ws = sensor_region.shape[0]
		avg_brightness = 0

		for y in range(0, h):
			for x in sensor_region:
				avg_brightness += image[y,x]

		avg_brightness /= (h * ws)
		avg_brightness /= 256.0
		print(f"brightness a {avg_brightness}")
		# if avg_brightness > .7:
		# 	avg_brightness = .7
		# if avg_brightness < .3:
		# 	avg_brightness = .3
		# print(f"brightness b {avg_brightness}")
		return avg_brightness

	image_width = image.shape[1]

	## Decide on the number of columns to use; here we treat the camera as two halves: left sensor, right sensor
	sensor_width = int(image_width/2)
	sensor_right = sense_brightness(image, sensor_region=np.arange(image_width-sensor_width, image_width))
	sensor_left = sense_brightness(image, sensor_region=np.arange(0, sensor_width))

	return sensor_left, sensor_right


def ipsilateral(f, left_sensor_value, right_sensor_value):
	'''Maps a sensor reading to a wheel motor command'''
	return f(left_sensor_value), f(right_sensor_value)

def contralateral(f, left_sensor_value, right_sensor_value):
	'''Maps a sensor reading to a wheel motor command'''
	return f(right_sensor_value), f(left_sensor_value)

def positive(s):
	t = max_speed * s
	print(f"positive s {s}; {t}")
	# s is the relative brightness, ranging from 0 .. 1
	return max_speed * s

def inhibitory(s):
	t = max_speed * (1.0 - s)
	print(f"inhibitory s {s}; {t}")
	return max_speed * (1.0 - s)

def fear(left_sensor_value, right_sensor_value):
	return ipsilateral(positive, left_sensor_value, right_sensor_value)

def aggression(left_sensor_value, right_sensor_value):
	return contralateral(positive, left_sensor_value, right_sensor_value)

def liking(left_sensor_value, right_sensor_value):
	return ipsilateral(inhibitory, left_sensor_value, right_sensor_value)

def love(left_sensor_value, right_sensor_value):
	return contralateral(inhibitory, left_sensor_value, right_sensor_value)

async def braitenberg_machine(robot: cozmo.robot.Robot):
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
		(sensor_left, sensor_right) = sense_brightnesses(opencv_image)

		# Map the sensors to actuators
		# (motor_left, motor_right) = fear(sensor_left, sensor_right)
		# (motor_left, motor_right) = aggression(sensor_left, sensor_right)
		# (motor_left, motor_right) = liking(sensor_left, sensor_right)
		(motor_left, motor_right) = love(sensor_left, sensor_right)

		print(f"{sensor_right},{sensor_left},    {motor_right},{motor_left}")

		# Send commands to the robot
		await robot.drive_wheels(motor_left, motor_right)

		time.sleep(.1)


cozmo.run_program(braitenberg_machine, use_viewer=True, force_viewer_on_top=True)
