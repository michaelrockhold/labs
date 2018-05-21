#!/usr/bin/env python3

import sys
import time
import asyncio
import numpy as np
import imgclassification
from skimage import io, color

try:
    from PIL import Image
except ImportError:
    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")

import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

class Bot:
    def __init__(self, robot):
        self.robot = robot

    def say_label(self, label):
        self.robot.say_text(label).wait_for_completed()

    def wave_hi(self):
        self.robot.set_all_backpack_lights(cozmo.lights.green_light)
        time.sleep(1)
        self.robot.set_all_backpack_lights(cozmo.lights.blue_light)
        time.sleep(1)
        # turn off Cozmo's backpack lights
        self.robot.set_all_backpack_lights(cozmo.lights.off_light)

async def run(robot: cozmo.robot.Robot):

    classifier = imgclassification.ImageClassifier()
    (data, labels) = classifier.prep('./train/')
    bot = Bot(robot)

    try:

        while True:
            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)
            feature_data = classifier.features_from_image(event.image)

            #find a card
            labels = classifier.predict_labels([image])
            if len(labels) > 0:
                bot.say_label(labels[0])
                bot.wave_hi()
                break
            time.sleep(2)

    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)


cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo just stays on the charger
cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)
