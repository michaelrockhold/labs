#!/usr/bin/env python3

import asyncio
import cozmo
import numpy
from numpy import dot
import math
from cozmo.util import degrees

𝚹_north = cozmo.util.radians(0)
𝚹_south = cozmo.util.radians(math.pi)
𝚹_west = cozmo.util.radians(math.pi/2)	
𝚹_east = cozmo.util.radians(-math.pi/2)

def get_relative_pose(cubePose, reference_frame_pose):
	'''Create a transformation matrix based on the reference frame, and 
	   apply it to the object. The result should be the pose of the object
	   in terms of the reference frame.'''
	(refx, refy, refz) = reference_frame_pose.position.x_y_z
	(x, y, z) = cubePose.position.x_y_z
	Θ = -reference_frame_pose.rotation.angle_z.radians	
	objectMatrix = numpy.array([
		float(x),
		float(y),
		1.0
	])

	xlate_home = numpy.array([
		[1.0,   0.0,   -float(refx)],
		[0.0,   1.0,   -float(refy)],
		[0.0,   0.0,            1.0]
		])
	rotate = numpy.array([
		[numpy.cos(Θ),  -numpy.sin(Θ),    0.0],
		[numpy.sin(Θ),   numpy.cos(Θ),    0.0],
		[         0.0,            0.0,    1.0]
		])
	tr = dot(rotate, xlate_home)
	result = dot(tr, objectMatrix)
	x = result[0]
	y = result[1]
	theta = cubePose.rotation.angle_z - reference_frame_pose.rotation.angle_z
	return cozmo.util.pose_z_angle(x, y, z, theta, origin_id=cubePose.origin_id)

def getWorldPoseOfRelativeObject(cubePose, robotPose):
	x, y, z = (0,0,0)
	(refx, refy, refz) = robotPose.position.x_y_z
	(x, y, z) = cubePose.position.x_y_z
	Θ = robotPose.rotation.angle_z.radians	
	objectMatrix = numpy.array([
		float(x),
		float(y),
		1.0
	])

	xlate_home = numpy.array([
		[1.0,   0.0,   float(refx)],
		[0.0,   1.0,   float(refy)],
		[0.0,   0.0,            1.0]
		])
	rotate = numpy.array([
		[numpy.cos(Θ),  -numpy.sin(Θ),    0.0],
		[numpy.sin(Θ),   numpy.cos(Θ),    0.0],
		[         0.0,            0.0,    1.0]
		])
	tr = dot(xlate_home,rotate)
	result = dot(tr, objectMatrix)
	x = result[0]
	y = result[1]
	theta = cubePose.rotation.angle_z + robotPose.rotation.angle_z
	return cozmo.util.pose_z_angle(x, y, z, theta, origin_id=cubePose.origin_id)

def format_pose(pose):
	args = pose.position.x_y_z + (pose.rotation.angle_z.radians,)
	s1 = "x = %4.3f, y = %4.3f, z = %4.3f, az = %3.3f" % args
	spaces = ' ' * (70 - len(s1))
	return s1 + spaces + ("Polar: r = %4.3f, Θ =  %3.3f" % cartesianToPolar(args[0],args[1]))

def format_pose2(pose):
	(x,y,z) = pose.position.x_y_z
	𝚹 = pose.rotation.angle_z.radians
	return "(% 8.2f, % 8.2f, % 8.2f), 𝚹: % 4.3f" % (x, y, z, 𝚹)

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

def pose_equal(p1, p2):
	(x1,y1,z1) = p1.position.x_y_z
	(x2,y2,z2) = p2.position.x_y_z
	a1 = numpy.array([x1, y1, z1, p1.rotation.angle_z.radians])
	a2 = numpy.array([x2, y2, z2, p2.rotation.angle_z.radians])
	try:
		numpy.testing.assert_array_almost_equal(a1, a2)
		return True
	except Exception as e:
		return False

def expect_equal(p1, p2):
	if pose_equal(p1, p2):
		return True
	else:
		print("\n    expected %s" % (format_pose2(p1)), end='')
		print("\n    to equal %s" % (format_pose2(p2)))
		return False

def createPoses(x1, y1, 𝚹1, x2, y2, 𝚹2):
	p1 = cozmo.util.pose_z_angle(x1,y1,0, 𝚹1, origin_id=1)
	p2 = cozmo.util.pose_z_angle(x2,y2,0, 𝚹2, origin_id=1)
	return (p1, p2)

def createRelativePoseTest(x1, y1, 𝚹1, x2, y2, 𝚹2, x3, y3, 𝚹3):
	def test():
		(robot_w, cube_w) = createPoses(x1, y1, 𝚹1, x2, y2, 𝚹2)
		return expect_equal(get_relative_pose(cube_w, robot_w), cozmo.util.pose_z_angle(x3, y3, 0, 𝚹3, origin_id=1))
	test.__name__ = ("rel_%d_%d_%4.3f_%d_%d_%4.3f" % (x1, y1, 𝚹1.radians, x2, y2, 𝚹2.radians))
	return test

def createWorldPoseTest(x1, y1, 𝚹1, x2, y2, 𝚹2, x3, y3, 𝚹3):
	def test():
		(robot_w, cube_w) = createPoses(x1, y1, 𝚹1, x2, y2, 𝚹2)
		return expect_equal(getWorldPoseOfRelativeObject(cube_w, robot_w), cozmo.util.pose_z_angle(x3, y3, 0, 𝚹3, origin_id=1))
	test.__name__ = ("world_%d_%d_%4.3f_%d_%d_%4.3f" % (x1, y1, 𝚹1.radians, x2, y2, 𝚹2.radians))
	return test

allTests = [
	## get_relative_pose tests
	createRelativePoseTest(0,0, 𝚹_north, 0,0, 𝚹_north, 0,0,𝚹_north),
	createRelativePoseTest(0,0,𝚹_north, 10,10,𝚹_north, 10, 10, 𝚹_north),
	createRelativePoseTest(0,0, 𝚹_west, 16,0, 𝚹_south, 0, -16, 𝚹_west),

	createRelativePoseTest(240,600, 𝚹_east, -240,-600, 𝚹_east, 1200, -480, 𝚹_north),
	createRelativePoseTest(240,600, 𝚹_east, -240,-600, 𝚹_west, 1200, -480, 𝚹_south), 
	createRelativePoseTest(10,20,𝚹_north, 20,30,𝚹_north, 10,10,𝚹_north),

	createRelativePoseTest(10,20,𝚹_east, 20,30,𝚹_north, -10,10,𝚹_west),
	createRelativePoseTest(0,0,𝚹_east, 20,30,𝚹_north, -30,20,𝚹_west),

	# getWorldPoseOfRelativeObject tests
	createWorldPoseTest(0,0,𝚹_north, 0,0,𝚹_north, 0,0,𝚹_north),

	createWorldPoseTest(0,0,𝚹_east, -20,30,𝚹_west, 30,20,𝚹_north),
	createWorldPoseTest(30,20,𝚹_north, 20,30,𝚹_north, 50,50,𝚹_north),
	createWorldPoseTest(30,20,𝚹_east, -30,20,𝚹_west, 50,50,𝚹_north),

	createWorldPoseTest(30,20,𝚹_south, 30,20,𝚹_south, 0,0,𝚹_north),
	createWorldPoseTest(30,30,𝚹_south, 50,50,𝚹_south, -20,-20,𝚹_north),
	createWorldPoseTest(-30,20,𝚹_west, 0,0,𝚹_north, -30,20,𝚹_west)
	]
def tests():
	successes = 0
	failures = 0
	total = len(allTests)
	i = 0
	for test in allTests:
		i += 1
		try:
			print("Test %d, %s" % (i, test.__name__), end='')
			r = test()
			if r:
				print(" passed")
				successes += 1
			else:
				print("    Failed")
				failures += 1
		except Exception as e:
			failures += 1
			print(": Exception: %s" % (e))

	print("%d passed, %d failed in %d total tests" % (successes, failures, total))

if __name__ == '__main__':
	#tests()
	cozmo.run_program(find_relative_cube_pose)
