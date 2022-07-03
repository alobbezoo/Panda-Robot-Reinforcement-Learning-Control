########### NOTICE ###########
# pose handling is not yet implemented on the receiving side!

### ros_bridge_send.py
# Helper function to make digital twin interact with real world
# Input: simulated environment
# Output: publishes either joint angles or desired poses on ROS network

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import time

rospy.init_node('bridge', anonymous=True)
joint_pub = rospy.Publisher('/bridge/joint_angles', Float64MultiArray, queue_size=10)
pose_pub = rospy.Publisher('/bridge/pose', Pose, queue_size=10)
time.sleep(1) # initial wait to ensure first message received

def talker(kind, message):
	if kind == "joints":
		rospy.loginfo(message)
		joint_pub.publish(message)

	elif kind == "pose":
		rospy.loginfo(message)
		pose_pub.publish(message)

def listener():
	rospy.Subscriber("/bridge/joint_angles/success", String, clear)
	rospy.spin()

def clear():
	clear = True

def publish_joint(env):
	clear = False
	listener()
	body_name = env.robot.body_name
	joint_angles = np.array([env.sim.get_joint_angle(joint=i, body=body_name) for i in range(7)])

	joint_goals = [joint_angles[0], joint_angles[1], joint_angles[2], joint_angles[3], joint_angles[4], joint_angles[5], joint_angles[6]]

	message = Float64MultiArray()
	message.data = joint_goals

	talker("joints", message)
	while clear = False:
		pass

def publish_pose(env):
	quat = env.sim.get_link_orientation("panda", 11)
	pos = env.sim.get_link_position("panda", 11)

	pose_goal = Pose()

	pose_goal.orientation.x = quat[0]
	pose_goal.orientation.y = quat[1]
	pose_goal.orientation.z = quat[2]
	pose_goal.orientation.w = quat[3]

	pose_goal.position.x = pos[0]
	pose_goal.position.y = pos[1]
	pose_goal.position.z = pos[2]

	talker("pose", pose_goal)
