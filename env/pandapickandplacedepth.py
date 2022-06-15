# from panda_gym.envs.tasks.reach import Reach
# from panda_gym.envs.core import RobotTaskEnv


from env.pandaCust import Panda #good 
from env.pybulletCust import PyBullet #good
from env.robottaskenv import RobotTaskEnv
from env.pandaPickAndPlace import PickAndPlace
import numpy as np

class PandaPickAndPlaceDepth(RobotTaskEnv):
    """Reach task wih Panda robot.
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim=sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace(sim=sim, reward_type=reward_type, get_ee_position=robot.get_ee_position, 
            gripper_width = robot.get_fingers_width, distance_threshold=0.04)
        super().__init__(robot, task)
