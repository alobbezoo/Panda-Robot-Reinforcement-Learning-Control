# OpenAI Imports:

from PIL import Image
from torchvision import transforms

from wrappers.pandaWrapperReach import PandaWrapper

import gym
import gym.spaces
import os
import register

import matplotlib.pyplot as plt
import numpy as np

# ROS/Gazebo/Moveit Imports:
import sys

from motionFunctions import *

env = gym.make("PandaReachDepth-v1", render=True)
env = PandaWrapper(env)
obs = env.reset()
controller = MoveGroupPythonInterface()

for i in range(5):

    action = env.action_space.sample()
    obs, _, _, _ = env.step(10*action)
    body_name = env.robot.body_name
    joint_angles = np.array([env.sim.get_joint_angle(joint=i, body=body_name) for i in range(7)])
    
    controller.go_to_joint_state(joint_goal0 = joint_angles[0], joint_goal1 = joint_angles[1], 
    joint_goal2 = joint_angles[2], joint_goal3 = joint_angles[3], joint_goal4 = joint_angles[4], 
    joint_goal5 = joint_angles[5], joint_goal6 = joint_angles[5])

    img = env.render()



# print("obs: ", obs)
# print("get_ee_position: ", env.robot.get_ee_position())
# print("env.robot.get_fingers_width(): ", env.robot.get_fingers_width())
# print("target_base_position is: ", env.task.get_goal())
# print("body name: ", body_name)
# print("joint angles:", joint_angles)

# img, depth = env.render(mode = 'rgb_array',width = 100, height= 100, distance = 1, 
#     target_position = [0.3, 0, 0.15], yaw = 75, pitch = -20) 

