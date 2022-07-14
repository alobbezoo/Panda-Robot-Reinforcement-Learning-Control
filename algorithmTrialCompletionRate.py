"""
File for testing the pre-trained networks
"""

# CUSTOM IMPORTS: 
from SB3.vectEnvs import make_panda_env
import register

from SB3.customCnnShallow_V0 import CustomCNN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

from SB3.callbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3 import PPO, SAC, TD3

# STANDARD IMPORTS
from tabnanny import verbose
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
import torch
import numpy as np
import gym
import os

from SB3.recordVideo import record_video_single, record_video_multiple

#INPUTS:
# from wrappers.pandaTrainedBlockGraspWrapperVect_BW_D_0_Supervised import PandaWrapper
from wrappers.pandaWrapperVect import PandaWrapperFunction
# from wrappers.pandaWrapperBW_D import PandaWrapper

from stable_baselines3.common.evaluation import evaluate_policy

from SB3.vectEnvs import make_panda_env
import register

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


# POLICY = PPO
POLICY = SAC
# POLICYNAME = "PPO"
POLICYNAME = "SAC"

# ENVNAME = "PandaReach"
# ENVNAME = "PandaGrasp"
ENVNAME = "PandaPickandPlace"
# ENVNAME = "PPO_Semi_Supervised_PandaGrasp"

# TYPE = "CNN"
TYPE = "Vect"

REWARDTYPE = "Dense"
# REWARDTYPE = "Sparse"

PandaWrapper = PandaWrapperFunction(10)

# # VECT

# # Reach:
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Vect/PPO/Dense/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Vect/SAC/Dense/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Vect/PPO/Sparse/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Vect/SAC/Sparse/best_model.zip"

# # Grasp:
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaGrasp/Vect/PPO/Dense/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaGrasp/Vect/SAC/Dense/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaGrasp/Vect/PPO/Sparse/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaGrasp/Vect/SAC/Sparce/best_model.zip"

# # Pick and Place
MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaPickandPlace/Vect/SAC/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaPickandPlace/Vect/PPO/PPO_PandaPickandPlace1_Vect/best_model.zip"


# # CNN

# # Reach:
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Cnn/PPO/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Cnn/SAC/best_model.zip"

# # Grasp:
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaGrasp/Cnn/PPO/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaGrasp/Cnn/SAC/best_model.zip"

loaded_model = POLICY.load(MODELPATH, device="cpu")
MODEL = loaded_model.policy

# ENV_ID = "PandaReachDepthDense-v1" 
# ENV_ID = "PandaReachDepth-v1" 

# ENV_ID = "PandaPickAndPlaceDepthBlockDense-v1" 
ENV_ID = "PandaPickAndPlaceDepthDense-v1" 
# ENV_ID = "PandaGraspDepthDense-v1" 
# ENV_ID = "PandaGraspDepthBlockDense-v1" 
# ENV_ID = "PandaGraspDepthBlock-v1" 

RENDER = False
NEPISODES = 100
GOALNENVS = 5


n_envs = loaded_model.n_envs
print("n_envs = ", n_envs)

eval_env = make_panda_env(env_id=ENV_ID, n_envs=GOALNENVS, wrapper=PandaWrapper)
eval_env = VecFrameStack(eval_env, n_stack=GOALNENVS)


eval_env_test = PandaWrapper(gym.make(ENV_ID))
print("eval_env_test.observation_space: ", eval_env_test.observation_space)

# print("eval_env.observation_space", eval_env.observation_space)
print("n_envs = ", eval_env.num_envs)


episode_rewards, episode_lengths = evaluate_policy(env=eval_env, model=MODEL, n_eval_episodes=NEPISODES, \
    return_episode_rewards=True, deterministic=False)

yescount = 0
counter = 0

for i in episode_lengths: 
    if i < 50:
        yescount += 1
    counter+=1
    
modelCompletionRate = (yescount)/(NEPISODES)

print("\nRewards for ",ENVNAME, TYPE, POLICYNAME, REWARDTYPE, "are:")
# print("Episode rewards are: ", episode_rewards)
print("Episode lengths are: ", episode_lengths)
print("Episode modelCompletionRate are: ", modelCompletionRate, "\n")