
from panda_gym.envs.core import Task
from panda_gym.utils import distance
from env.pybulletCust import PyBullet 

import numpy as np
from typing import Union, Dict, Any
#@title Default title text

from panda_gym.envs.core import Task
# from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class PickAndPlace(Task):
    def __init__(
        self,
        get_ee_position,
        gripper_width, 
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.2,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.25,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.gripper_width = gripper_width
        self.object_size = 0.05
        self.target_size = 0.1
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
        
        self.sim.create_cylinder(
            body_name="object",
            radius=self.object_size / 2,
            height=self.object_size/1.5,
            mass=0.5,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
            lateral_friction = 1, # None,
            spinning_friction = 1, 
        )
        

        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.target_size / 2,
            mass=500.0,
            # ghost=True,
            position=np.array([0.05, 0.1, self.target_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
            lateral_friction = 1, # None,
            spinning_friction = 1, 
        )



    def get_obs(self) -> np.ndarray:
        self.goal = self.sim.get_base_position("target") + [0, 0, (self.target_size/2+0.01+self.object_size/2)] # Need to update goal as it changes
        
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        # object_rotation = self.sim.get_base_rotation("object")
        target_position = self.sim.get_base_position("target")
        # target_rotation = self.sim.get_base_rotation("target")
        # observation = np.concatenate([object_position, object_rotation, target_position, target_rotation])  # object_velocity, object_angular_velocity]) 
        observation = np.concatenate([object_position, target_position])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position
    
    def get_desired_goal(self) -> np.ndarray:
        target_position_mid = np.array(self.sim.get_base_position("target"))
        target_position = target_position_mid + [0, 0, (self.target_size/2+0.01 + self.object_size/2)]
        return target_position

    def reset(self) -> None:
        
        object_position = self._sample_object()     
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.goal = self.get_desired_goal()

        return self.get_ee_position 

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([-0.20, -0.20, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        
        achieved_goal = self.get_achieved_goal()  
        desired_goal = self.get_desired_goal()

        d = distance(achieved_goal, desired_goal)

        # print("is_sucess: ", np.array(((d < self.distance_threshold) and (gripper_width > min_gripper_width)), dtype=np.float64))

        return np.array((d < self.distance_threshold), dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        achieved_goal = self.get_achieved_goal()  
        desired_goal = self.get_desired_goal()

        # Distance between block and the target
        d1 = distance(achieved_goal, desired_goal)
        
        # Add term for extra negative if the block is not lifted
        if (achieved_goal[2]<desired_goal[2]):
            d1+= (desired_goal[2] - achieved_goal[2])*2
        
        # Distance between the end effector and the block
        d2 = distance(self.get_ee_position(), achieved_goal)

        d = d1 + d2

        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
