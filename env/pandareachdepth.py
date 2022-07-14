# from panda_gym.envs.tasks.reach import Reach
# from panda_gym.envs.core import RobotTaskEnv
from env.robottaskenv import RobotTaskEnv
from env.pandaCust import Panda
from env.pybulletCust import PyBullet
from env.pandaReach import Reach

import numpy as np

class PandaReachDepth(RobotTaskEnv):
    """Reach task wih Panda robot.
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render, background_color=np.array([255,255,255]))
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Reach(sim=sim, reward_type=reward_type, get_ee_position=robot.get_ee_position, distance_threshold=0.04)
        super().__init__(robot, task)
