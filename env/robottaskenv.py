import gym_robotics
import gym
from panda_gym.envs.core import PyBulletRobot
from panda_gym.envs.core import Task
from typing import Any, Dict, Union, Iterator, Optional, Tuple
import numpy as np

"""
MAIN CHANGE IN THIS SCRIPT WAS DONE ON LINE 62
"""

class RobotTaskEnv(gym_robotics.GoalEnv):
    """Robotic task goal env, as the junction of a task and a robot.
    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, robot: PyBulletRobot, task: Task) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.robot = robot
        self.task = task
        obs = self.reset()  # required for init; seed can be changed later
        observation_shape = obs["observation"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["achieved_goal"].shape
        self.observation_space = gym.spaces.Dict(
            dict(
                observation=gym.spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                achieved_goal=gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
            )
        )
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()  # robot state
        task_obs = self.task.get_obs()  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal()
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
        }

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        self.task.np_random, seed = gym.utils.seeding.np_random(seed)
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        obs = self._get_obs()
        done = bool(self.task.is_success(obs["achieved_goal"], self.task.get_goal())) #False             
        
        #NOTE: Major change here, before done was fixed as done
        # print("done bool check: ", done)
        
        info = {"is_success": self.task.is_success(obs["achieved_goal"], self.task.get_goal())}
        reward = self.task.compute_reward(obs["achieved_goal"], self.task.get_goal(), info)
        assert isinstance(reward, float)  # needed for pytype cheking
        return obs, reward, done, info

    def close(self) -> None:
        self.sim.close()

    def render(
        self,
        mode: str,
        width: int = 720,
        height: int = 480,
        target_position: Optional[np.ndarray] = None,
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.
        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.
        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.
        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        target_position = target_position if target_position is not None else np.zeros(3)
        return self.sim.render(
            mode,
            width=width,
            height=height,
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )