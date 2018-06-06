#!/usr/bin/env python3

import asyncio
import sys
import time

import cv2
import numpy as np

import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
from random import randint
import math

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')

sys.path.insert(0, '../lab4')
import find_ball

max_speed = 50.0


# Define a decorator as a subclass of Annotator; displays battery voltage
class BatteryAnnotator(cozmo.annotate.Annotator):
    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)
        batt = self.world.robot.battery_voltage
        text = cozmo.annotate.ImageText('BATT %.1fv' % batt, color='green')
        text.render(d, bounds)

# Define a decorator as a subclass of Annotator; displays the ball
class BallAnnotator(cozmo.annotate.Annotator):

    ball = None

    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)

        if BallAnnotator.ball is not None:

            #double size of bounding box to match size of rendered image
            BallAnnotator.ball = np.multiply(BallAnnotator.ball,2)

            #define and display bounding box with params:
            #msg.img_topLeft_x, msg.img_topLeft_y, msg.img_width, msg.img_height
            box = cozmo.util.ImageBox(BallAnnotator.ball[0]-BallAnnotator.ball[2],
                                      BallAnnotator.ball[1]-BallAnnotator.ball[2],
                                      BallAnnotator.ball[2]*2, BallAnnotator.ball[2]*2)
            cozmo.annotate.add_img_box_to_image(image, box, "green", text=None)

            BallAnnotator.ball = None

# Utilities

def did_occur_recently(event_time, max_elapsed_time):
    '''Did event_time occur and was it within the last max_elapsed_time seconds?'''
    if event_time is None:
        return False
    elapsed_time = time.time() - event_time
    return elapsed_time < max_elapsed_time


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

# Behavior components

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

# Behavior functions

def fear(left_sensor_value, right_sensor_value):
	return ipsilateral(positive, left_sensor_value, right_sensor_value)

def aggression(left_sensor_value, right_sensor_value):
	return contralateral(positive, left_sensor_value, right_sensor_value)

def liking(left_sensor_value, right_sensor_value):
	return ipsilateral(inhibitory, left_sensor_value, right_sensor_value)

def love(left_sensor_value, right_sensor_value):
	return contralateral(inhibitory, left_sensor_value, right_sensor_value)


class Ball:

    def __init__(self, coords, image):
        self.x = coords[0]
        self.y = coords[1]
        self.radius = coords[2]
        self.imageWidth = image.shape[1]
        self.imageHeight = image.shape[0]

    def brightnesses(self):
        max_radius = self.imageHeight
        max_signal = max_radius * self.imageHeight / 2
        left = 0
        right = 0
        if self.x > self.imageWidth / 2:
            right = self.x
        else:
            left = self.imageWidth - self.x

        return left/max_signal, right/max_signal


class BallSearcher:

    def __init__(self, robot, strategy):
        self.robot = robot
        self.initial_pose_angle = 0
        self.patrol_offset = 0  # middle
        # Time to wait between each turn and patrol, in seconds
        self.time_between_turns = 2.5
        self.time_between_patrols = 20
        self.time_for_next_turn = 0
        self.time_for_next_patrol = 0
        self.found_ball = None
        self.ballFinder = find_ball.ballFinders[2]
        self.strategy = strategy

        # Turn on image receiving by the camera
        self.robot.camera.image_stream_enabled = True

        #add annotators for battery level and ball bounding box
        self.robot.world.image_annotator.add_annotator('battery', BatteryAnnotator)
        self.robot.world.image_annotator.add_annotator('ball', BallAnnotator)


    async def imposition(self):
        # Make sure Cozmo is clear of the charger
        if self.robot.is_on_charger:
            # Drive fully clear of charger (not just off the contacts)
            await self.robot.drive_off_charger_contacts().wait_for_completed()
            await self.robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()
        await self.robot.set_head_angle(cozmo.util.radians(0)).wait_for_completed()


    async def main(self):

        await self.imposition()

        self.initial_pose_angle = self.robot.pose_angle

        self.time_for_next_turn = time.time() + self.time_between_turns
        self.time_for_next_patrol = time.time() + self.time_between_patrols

        # Main Loop
        while True:

            # Turn head every few seconds to cover a wider field of view
            # Only do this if not currently examining a possible ball
            #             if (time.time() > self.time_for_next_turn) and self.found_ball == None:
            if self.found_ball == None:
                await self.swivelHead()

            # Every now and again patrol left and right between 3 patrol points
            #if (time.time() > self.time_for_next_patrol) and self.found_ball == None:
            if self.found_ball == None:
                await self.patrol()

            await self.lookForBall()

            await self.approach_ball()

            # Sleep to allow other things to run
            await asyncio.sleep(0.05)


    async def swivelHead(self):

        max_pose_angle = 45  # offset from initial pose_angle (up to +45 or -45 from this)

        # pick a random amount to turn
        angle_to_turn = randint(10,40)

        # 50% chance of turning in either direction
        if randint(0,1) > 0:
            angle_to_turn = -angle_to_turn

        # Clamp the amount to turn
        face_angle = (self.robot.pose_angle - self.initial_pose_angle).degrees

        face_angle += angle_to_turn
        if face_angle > max_pose_angle:
            angle_to_turn -= (face_angle - max_pose_angle)
        elif face_angle < -max_pose_angle:
            angle_to_turn -= (face_angle + max_pose_angle)

        # Turn left/right
        await self.robot.turn_in_place(degrees(angle_to_turn)).wait_for_completed()

        # Queue up the next time to look around
        self.time_for_next_turn = time.time() + self.time_between_turns


    async def patrol(self):
        # Check which way robot is facing vs initial pose, pick a new patrol point

        face_angle = (self.robot.pose_angle - self.initial_pose_angle).degrees
        drive_right = (self.patrol_offset < 0) or ((self.patrol_offset == 0) and (face_angle > 0))

        # Turn to face the new patrol point
        if drive_right:
            await self.robot.turn_in_place(degrees(90 - face_angle)).wait_for_completed()
            self.patrol_offset += 1
        else:
            await self.robot.turn_in_place(degrees(-90 - face_angle)).wait_for_completed()
            self.patrol_offset -= 1

        # Drive to the patrol point, playing animations along the way
        await self.robot.drive_wheels(20, 20)
        for i in range(1,4):
            await self.robot.play_anim("anim_hiking_driving_loop_0" + str(i)).wait_for_completed()

        # Stop driving
        self.robot.stop_all_motors()

        # Turn to face forwards again
        face_angle = (self.robot.pose_angle - self.initial_pose_angle).degrees
        if face_angle > 0:
            await self.robot.turn_in_place(degrees(-90)).wait_for_completed()
        else:
            await self.robot.turn_in_place(degrees(90)).wait_for_completed()

        # Queue up the next time to patrol
        self.time_for_next_patrol = time.time() + self.time_between_patrols


    async def lookForBall(self):

        MAX_LOOP = 25
        self.found_ball = None
        try:
            loopCount = 0
            while loopCount < MAX_LOOP:
                #get camera image
                event = await self.robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

                #convert camera image to opencv format
                opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)

                #find the ball
                ball = self.ballFinder(opencv_image)
                print("BALL: ", ball)

                if ball == None:
                    loopCount += 1
                    BallAnnotator.ball = None
                    self.lights_off()
                else:
                    loopCount = MAX_LOOP
                    BallAnnotator.ball = ball
                    self.found_ball = Ball(ball, opencv_image)
                    await self.robot.play_anim_trigger(cozmo.anim.Triggers.CubePouncePounceNormal).wait_for_completed()
                    self.lights_blue()
                    break

        except cozmo.RobotBusy as e:
            print(e)


    async def approach_ball(self):
        # Move lift down and tilt the head up
        # self.robot.move_lift(-3)

        if self.found_ball == None:
            self.lights_off()
            self.robot.stop_all_motors()

        else:
            self.lights_red()
            # Sense the current brightness values on the right and left of the image.
            (sensor_left, sensor_right) = self.found_ball.brightnesses()

            # Map the sensors to actuators using the supplied strategy
            (motor_left, motor_right) = self.strategy(sensor_left, sensor_right)

            log(sensor_left, sensor_right, motor_left, motor_right)

            # Send commands to the robot
            await self.robot.drive_wheels(motor_left, motor_right)

            time.sleep(.1)
            self.lights_blue


    def lights_off(self):
        self.robot.set_backpack_lights_off()

    def lights_red(self):
        self.robot.set_all_backpack_lights(cozmo.lights.red_light)

    def lights_green(self):
        self.robot.set_all_backpack_lights(cozmo.lights.green_light)

    def lights_blue(self):
        self.robot.set_all_backpack_lights(cozmo.lights.blue_light)

### startup

async def run(sdk_conn):
    '''The run method runs once the Cozmo SDK is connected.'''
    robot = await sdk_conn.wait_for_robot()

    # Create the BallSearcher
    ballSearcher = BallSearcher(robot, aggression)

    try:
        await ballSearcher.main()

    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")


if __name__ == '__main__':
    cozmo.setup_basic_logging()
    cozmo.camera.Camera.enable_auto_exposure = False
    cozmo.robot.Robot.drive_off_charger_on_connect = False  # Stay on charger until init
    try:
        cozmo.connect_with_tkviewer(run, force_on_top=True)
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)

