import os
import gym

from gym.envs.registration import register

from env.pandareachdepth import PandaReachDepth
from env.pandagraspdepth import PandaGraspDepth
from env.pandapickandplacedepth import PandaPickAndPlaceDepth

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="PandaReachDepth{}{}-v1".format(control_suffix, reward_suffix),
            entry_point=PandaReachDepth,
            kwargs=kwargs,
            max_episode_steps=50,
        )
        
        register(
            id="PandaGraspDepth{}{}-v1".format(control_suffix, reward_suffix),
            entry_point=PandaGraspDepth, 
            kwargs=kwargs,
            max_episode_steps=50,
        )
        
        register(
            id="PandaPickAndPlaceDepth{}{}-v1".format(control_suffix, reward_suffix),
            entry_point=PandaPickAndPlaceDepth, 
            kwargs=kwargs,
            max_episode_steps=50,
        )
