# OpenAI Imports:
from PIL import Image
from torchvision import transforms

from wrappers.pandaWrapperBW_D import PandaWrapper

import gym
import gym.spaces
import os
import register

import matplotlib.pyplot as plt
import numpy as np


# ROS/Gazebo/Moveit Imports:
import sys

import moveit_commander
import moveit_msgs.msg
import rospy

#missing import
import geometry_msgs.msg

try:
    from math import pi, tau
except ImportError:  # For Python 2 compatibility
    from math import pi, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from datetime import datetime

date = datetime.now().date() 

# SET DIRECTORIES: 
import os
from datetime import datetime
date = datetime.now().date() 

homeDir = str(os.getcwd() + "/")

pathDir = homeDir + str(date) + "/"
if not os.path.exists(pathDir):
    os.mkdir(pathDir)

modelHomeDir = pathDir + "PPO/"
if not os.path.exists(modelHomeDir):
    os.mkdir(modelHomeDir)

imgDir = modelHomeDir + "images/"
if not os.path.exists(imgDir):
    os.mkdir(imgDir)


# Create The Controller:
# Initiate MoveIt commander and node
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("move_group_python_interface_testing", anonymous=True)

# Create commanders
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

# Load arm and hand groups
move_group = moveit_commander.MoveGroupCommander("panda_arm")
hand_move_group = moveit_commander.MoveGroupCommander("hand")

display_trajectory_publisher = rospy.Publisher(
"/move_group/display_planned_path",
moveit_msgs.msg.DisplayTrajectory,
queue_size=20,
)

# We can get the name of the reference frame for this robot:
planning_frame = move_group.get_planning_frame()
print("============ Planning frame: %s" % planning_frame)

# We can also print the name of the end-effector link for this group:
eef_link = move_group.get_end_effector_link()
print("============ End effector link: %s" % eef_link)

# We can get a list of all the groups in the robot:
group_names = robot.get_group_names()
print("============ Available Planning Groups:", robot.get_group_names())

# Sometimes for debugging it is useful to print the entire state of the
# robot:
print("============ Printing robot state")
print(robot.get_current_state())
print("")

# # -- Plan arm joint goal --

# -- UNCOMMENT BELOW TO SEE THE SACLING FACTOR IN ACTION --
max_vel_scale_factor = 0.5
# max_vel_scale_factor = 1.0
# max_acc_scale_factor = 0.2
max_acc_scale_factor = 0.5
move_group.set_max_velocity_scaling_factor(max_vel_scale_factor)
move_group.set_max_acceleration_scaling_factor(max_acc_scale_factor)


# print(sys.path)
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/home/hjkwon/scripts/pythonPanda')


env = gym.make("PandaGraspDepthDense-v1")
env = PandaWrapper(env)
obs = env.reset()

for i in range(10):

    action = env.action_space.sample()
    obs, _, _, _ = env.step(5*action)
    body_name = env.robot.body_name
    joint_angles = np.array([env.sim.get_joint_angle(joint=i, body=body_name) for i in range(7)])
    
    joint_goal = move_group.get_current_joint_values()

    print("joint goal is: ", joint_goal)

    joint_goal[0] = joint_angles[0]
    joint_goal[1] = joint_angles[1]
    joint_goal[2] = joint_angles[2]
    joint_goal[3] = joint_angles[3]
    joint_goal[4] = joint_angles[4]
    joint_goal[5] = joint_angles[5]
    joint_goal[6] = joint_angles[6]

    move_group.set_joint_value_target(joint_goal)
    move_group.plan()
    move_group.go(wait=True)
    # Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()

    move_group.clear_pose_targets()

    # -- Plan hand goal --

    gripperWidth = env.robot.get_fingers_width()

    hand_move_group.set_joint_value_target([gripperWidth/2, gripperWidth/2])
    (hand_plan_retval, plan, _, error_code) = hand_move_group.plan()
    retval = hand_move_group.execute(plan, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()
    

    img = env.render()
    plt.imshow(img)
    plt.savefig(imgDir + str(i) + ".png")

