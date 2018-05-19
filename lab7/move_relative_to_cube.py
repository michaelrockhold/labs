#!/usr/bin/env python3

'''
This is starter code for Lab 7.

'''

import cozmo
from cozmo.util import degrees, Angle, Pose, distance_mm, speed_mmps
import math
import time
import sys

from odometry import cozmo_go_to_pose

sys.path.insert(0, '../lab6')
import pose_transform

def move_relative_to_cube(robot: cozmo.robot.Robot):
	'''Looks for a cube while sitting still, when a cube is detected it 
	moves the robot to a given pose relative to the detected cube pose.'''

	robot.move_lift(-3)
	robot.set_head_angle(degrees(0)).wait_for_completed()
	cube_w = None

	while cube_w is None:
		try:
			cube_w = robot.world.wait_for_observed_light_cube(timeout=30)
			if cube_w:
				print("Found a cube, pose in the robot coordinate frame: %s" % pose_transform.get_relative_pose(cube_w.pose, robot.pose))
				break
		except asyncio.TimeoutError:
			print("Didn't find a cube")

	desired_pose_relative_to_cube = Pose(0, 100, 0, angle_z=degrees(90))

	# ####
	# TODO: Make the robot move to the given desired_pose_relative_to_cube.
	# Use the get_relative_pose function you implemented to determine the
	# desired robot pose relative to the robot's current pose and then use
	# one of the go_to_pose functions you implemented in Lab 6.
	# ####

	pose_relative_to_world = pose_transform.getWorldPoseOfRelativeObject(desired_pose_relative_to_cube, robot.pose)
	print("destination pose in the world coordinate frame: %s" % pose_transform.format_pose(pose_relative_to_world))

if __name__ == '__main__':

	cozmo.run_program(move_relative_to_cube)
