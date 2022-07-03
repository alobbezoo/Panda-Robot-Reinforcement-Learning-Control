# OpenAI Imports:

from PIL import Image
from torchvision import transforms

import gym
import gym.spaces
from panda_gym.envs.tasks.reach import Reach
import os
import register

import matplotlib.pyplot as plt
import numpy as np
#import tf.transformations

# ROS/Gazebo/Moveit Imports:
import sys
import geometry_msgs.msg
import time
from wrappers.pandaWrapperVect_BW_D_0 import PandaWrapper
from helpers.ros_bridge_send import publish_joint
# from motionFunctions import *

env = gym.make("PandaGraspDepthDense-v1", render=True)
env = PandaWrapper(env)
env.reset()
env.render()
publish_joint(env)
for i in range(20):
    time.sleep(3)
    obs, _, _, _ = env.step(action=env.action_space.sample())
    publish_joint(env)

# print("obs: ", obs)
# # controller = MoveGroupPythonInterface()

# action = env.action_space.sample()
# print("action: ", action)
# obs, _, _, _ = env.step(10*action)
# publish_joint(env)

# for i in range(5):

#     action = env.action_space.sample()
#     print("action: ", action)
#     obs, _, _, _ = env.step(10*action)
#     print("obs: ", obs)

#     quat = env.sim.get_link_orientation("panda", 11)
#     pos = env.sim.get_link_position("panda", 11)
#     tgt = env.sim.get_base_position("object")
#     print("target position: ", tgt)

#     print("get_ee_position: ", env.robot.get_ee_position())

#     publish_joint(env)

#     img = env.render()
#     time.sleep(2)



# print("obs: ", obs)
# print("get_ee_position: ", env.robot.get_ee_position())
# print("env.robot.get_fingers_width(): ", env.robot.get_fingers_width())
# print("target_base_position is: ", env.task.get_goal())
# print("body name: ", body_name)
# print("joint angles:", joint_angles)

# img, depth = env.render(mode = 'rgb_array',width = 100, height= 100, distance = 1, 
#     target_position = [0.3, 0, 0.15], yaw = 75, pitch = -20) 

