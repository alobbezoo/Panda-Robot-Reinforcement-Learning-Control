
from panda_gym.envs.core import Task
from panda_gym.utils import distance
from env.pybulletCust import PyBullet 

import numpy as np
from typing import Union, Dict, Any

class Reach(Task):
    def __init__(
        self,
        get_ee_position,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.3,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.get_ee_position = get_ee_position
        self.distance_threshold = distance_threshold
        self.object_size = 0.035
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        self.sim.create_sphere(
            body_name="object",
            radius=0.03,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )

    def get_obs(self) -> np.ndarray:
        self.goal = self.sim.get_base_position("object") # Need to update goal as it changes
        object_position = self.sim.get_base_position("object")

        return object_position

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        object_position = self._sample_object()
        # self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0])) 

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""

        object_position = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        # object_position = np.array([0.0, 0.0, self.object_size / 2])
        # noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        # object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        
        achieved_goal = self.get_achieved_goal()

        d = distance(achieved_goal, desired_goal)

        return np.array(((d < self.distance_threshold)), dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        desired_goal = self.sim.get_base_position("object") 
        achieved_goal = self.get_achieved_goal()

        d = distance(achieved_goal, desired_goal)
        
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d