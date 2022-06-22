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
import tf.transformations

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

    quat = env.sim.get_link_orientation("panda", 11)
    pos = env.sim.get_link_position("panda", 11)

    # print("get_ee_position: ", env.robot.get_ee_position())
    print("get_joint_angle: ",env.sim.get_link_orientation("panda", 11)) 
    print("get_link_position: ",env.sim.get_link_position("panda", 11)) 


    # controller.go_to_pose_goal(pose_goal_orientation_x = quat[0], pose_goal_orientation_y = quat[1], 
    #     pose_goal_orientation_z = quat[2], pose_goal_orientation_w = quat[3],  pose_goal_position_x = pos[0], 
    #     pose_goal_position_y = pos[1], pose_goal_position_z = pos[2])

    # pose_goal.orientation.x= quat[0]
    # pose_goal.orientation.y = quat[1]
    # pose_goal.orientation.z = quat[2]
    # pose_goal.orientation.w = quat[3]

    # pose_goal.position.x = pos[0]
    # pose_goal.position.y = pos[1]
    # pose_goal.position.z = pos[2]

    # controller.go_to_pose_goal(pose_goal)

    # body_name = env.robot.body_name
    # joint_angles = np.array([env.sim.get_joint_angle(joint=i, body=body_name) for i in range(7)])
    # controller.go_to_joint_state(joint_goal0 = joint_angles[0], joint_goal1 = joint_angles[1], 
    # joint_goal2 = joint_angles[2], joint_goal3 = joint_angles[3], joint_goal4 = joint_angles[4], 
    # joint_goal5 = joint_angles[5], joint_goal6 = joint_angles[5])

    # img = env.render()



# print("obs: ", obs)
# print("get_ee_position: ", env.robot.get_ee_position())
# print("env.robot.get_fingers_width(): ", env.robot.get_fingers_width())
# print("target_base_position is: ", env.task.get_goal())
# print("body name: ", body_name)
# print("joint angles:", joint_angles)

# img, depth = env.render(mode = 'rgb_array',width = 100, height= 100, distance = 1, 
#     target_position = [0.3, 0, 0.15], yaw = 75, pitch = -20) 

