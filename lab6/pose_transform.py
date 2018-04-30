#!/usr/bin/env python3

'''
This is starter code for Lab 6 on Coordinate Frame transforms.

'''

import asyncio
import cozmo
import numpy
from cozmo.util import degrees

def get_relative_pose(object_pose, reference_frame_pose):
	(refx, refy, refz) = reference_frame_pose.position.x_y_z
	(r, Θ) = cartesianToPolar(refx,refy)
	refMatrix = numpy.ndarray((3,3), dtype=float, buffer=numpy.array([
		numpy.cos(Θ), 	-numpy.sin(Θ),	 refx,
		numpy.sin(Θ), 	numpy.cos(Θ), 	 refy,
		0.0, 			0.0, 			 1
	]))
	
	(x, y, z) = object_pose.position.x_y_z
	objectMatrix = numpy.ndarray((3,1), dtype=float, buffer=numpy.array([
		x,
		y,
		1
	]))

	result = numpy.dot(refMatrix, objectMatrix)
	x = result.item((0, 0))
	y = result.item((1, 0))
	theta = object_pose.rotation.angle_z - reference_frame_pose.rotation.angle_z
	return cozmo.util.pose_z_angle(x, y, z, theta, origin_id=object_pose.origin_id)

def format_pose(pose):
	args = pose.position.x_y_z + (pose.rotation.angle_z.radians,)
	s1 = "x = %4.3f, y = %4.3f, z = %4.3f, az = %3.3f" % args
	spaces = ' ' * (70 - len(s1))
	return s1 + spaces + ("Polar: r = %4.3f, Θ =  %3.3f" % cartesianToPolar(args[0],args[1]))

def cartesianToPolar(x, y):
	return numpy.sqrt(x**2+y**2), numpy.arctan2(y,x)

def find_relative_cube_pose(robot: cozmo.robot.Robot):
	'''Looks for a cube while sitting still, prints the pose of the detected cube
	in world coordinate frame and relative to the robot coordinate frame.'''

	robot.move_lift(-3)
	robot.set_head_angle(degrees(0)).wait_for_completed()
	cube = None

	while True:
		try:
			cube = robot.world.wait_for_observed_light_cube(timeout=30)
			if cube:
				print("Robot pose:                              %s" % format_pose(robot.pose))
				print("Cube pose:                               %s" % format_pose(cube.pose))
				print("Cube pose in the robot coordinate frame: %s" % format_pose(get_relative_pose(cube.pose, robot.pose)))
		except asyncio.TimeoutError:
			print("Didn't find a cube")


if __name__ == '__main__':
	cozmo.run_program(find_relative_cube_pose)
