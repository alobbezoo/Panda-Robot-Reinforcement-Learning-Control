
from panda_gym.envs.core import Task
from panda_gym.utils import distance
from env.pybulletCust import PyBullet 

import numpy as np
from typing import Union, Dict, Any

class Grasp(Task):
    def __init__(
        self,
        get_ee_position,
        gripper_width, 
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.gripper_width = gripper_width
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        
        # self.sim.create_box(
        #     body_name="target",
        #     half_extents=np.ones(3) * self.object_size / 2,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.array([0.0, 0.0, 0.05]),
        #     rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        # )

    def get_obs(self) -> np.ndarray:
        self.goal = self.sim.get_base_position("object") # Need to update goal as it changes
        
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        # object_velocity = self.sim.get_base_velocity("object")
        # object_angular_velocity = self.sim.get_base_angular_velocity("object")
        # observation = np.concatenate([object_position, object_rotation, object_position_2]) 
        observation = np.concatenate([object_position, object_rotation])  # object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        # object_position = np.array(self.sim.get_base_position("object")
        # return object_position
        ee_position = np.array(self.get_ee_position())
        fingers_width = self.gripper_width()
        ee_position = np.concatenate((ee_position, [fingers_width])) #HERE
        return ee_position

    def reset(self) -> None:
        # self.goal = self._sample_goal()
        # NOTE: testing change here since not pick and place
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        # self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0])) 


    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2, 0])  # z offset for the cube center        #HERE
        # noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        # if self.np_random.random() < 0.3:
        #     noise[2] = 0.0
        # goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        
        min_gripper_width = 0.05
        achieved_goal = self.get_achieved_goal()
        gripper_width = achieved_goal[3]
        ## Better Approach

        d = distance(achieved_goal[0:3], desired_goal)
        # achieved_goal[3] > 0.1

        # return np.array(((d < self.distance_threshold) and (gripper_width > min_gripper_width)), dtype=np.float64)
        # return np.array(d < self.distance_threshold, dtype=np.float64)
        # print("is_sucess 1: ", np.array(d < self.distance_threshold, dtype=np.float64))
        # print("is_sucess 2: ", np.array(gripper_width > min_gripper_width, dtype=np.float64))
        # print("is_sucess: ", np.array(((d < self.distance_threshold), dtype=np.float64))
        # print("is_sucess: ", np.array(((d < self.distance_threshold) and (gripper_width > min_gripper_width)), dtype=np.float64))

        return np.array(((d < self.distance_threshold) and (gripper_width > min_gripper_width)), dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        desired_goal = self.sim.get_base_position("object") 
        achieved_goal = self.get_achieved_goal()[0:3] # Better Approach

        d = distance(achieved_goal, desired_goal)

        # Note: Could add a gripper based reward here
        # if self.get_achieved_goal()[3] < 0.05:
        #   d += 0.5
        
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d