#!/usr/bin/env python3

import sys
import time
import asyncio
import numpy as np
import imgclassification
from skimage import color
import cv2 as cv2

try:
    from PIL import Image
except ImportError:
    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")

import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

class Bot:
    def __init__(self, robot):
        self.robot = robot

    async def say_label(self, label):
        await self.robot.say_text(label).wait_for_completed()

    def wave_hi(self):
        self.robot.set_all_backpack_lights(cozmo.lights.green_light)
        time.sleep(1)
        self.robot.set_all_backpack_lights(cozmo.lights.blue_light)
        time.sleep(1)
        # turn off Cozmo's backpack lights
        self.robot.set_all_backpack_lights(cozmo.lights.off_light)
        time.sleep(1)

    async def celebrate(self, label):
        await self.say_label(label)
        self.wave_hi()



classifier = imgclassification.ImageClassifier()
(data, labels) = classifier.prep('./train/')

async def run(robot: cozmo.robot.Robot):
    robot.drive_off_charger_on_connect = False  # Cozmo just stays on the charger
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True
    bot = Bot(robot)

    await bot.say_label("Hello, world!")
    try:
        while True:
            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)
            greyImage = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)
            feature_data = classifier.features_from_image(greyImage)

            #find a card
            labels = classifier.predict_labels([feature_data])
            print("labels", labels)
            if len(labels) > 0:
                await bot.celebrate(labels[0])
                break
            time.sleep(2)

    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)


cozmo.run_program(run)
