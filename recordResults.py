"""
File for testing the pre-trained networks
"""

# CUSTOM IMPORTS: 
from SB3.vectEnvs import make_panda_env
import register

from SB3.customCnnShallow_V0 import CustomCNN

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
# from wrappers.pandaWrapperBW_D import PandaWrapper #Note: confirm you have the correct wrapper
# from wrappers.pandaPPOTrainedGraspWrapperVect_BW_D_0_Supervised import PandaWrapper
from wrappers.pandaWrapperVect import PandaWrapperFunction

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

PandaWrapper = PandaWrapperFunction(19)

# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/cnnTraining/2022-06-24/SAC/callback/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PPO_PandaGrasp/Vect/PPO_PandaGrasp_Vect/best_model.zip"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/vectTraining/2022-06-27/PPOGrasp/callback/best_model.zip"
MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/vectTraining/2022-06-26/SACOptunaPandaPickAndPlaceDepthDense-v1/optimizedCallbackDir/best_model.zip"
#MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Vect/PPO/Dense/best_model.zip"

# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/vectTraining/2022-06-26/SACOptunaPandaPickAndPlaceDepthDense-v1/optimizedCallbackDir/best_model.zip"

# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/vectTraining/2022-06-28/SACPickandPlace/callback/best_model.zip"

loaded_model = POLICY.load(MODELPATH, device="cpu")

# ENV_ID = "PandaReachDepthDense-v1" 
# ENV_ID = "PandaGraspDepthDense-v1" 
ENV_ID = "PandaPickAndPlaceDepthBlockDense-v1" 

RENDER = False

n_envs = loaded_model.n_envs

# SET DIRECTORIES: 
import os
from datetime import datetime
date = datetime.now().date() 

homeDir = str(os.getcwd() + "/")

pathDir = homeDir + "trainedModelsPlots/"
if not os.path.exists(pathDir):
    os.mkdir(pathDir)

nameDir1 = pathDir + ENVNAME + "/"
if not os.path.exists(nameDir1):
    os.mkdir(nameDir1)

nameDir2 = nameDir1 + TYPE + "/"
if not os.path.exists(nameDir2):
    os.mkdir(nameDir2)

nameDir3 = nameDir2 + POLICYNAME + "unconverged/"
if not os.path.exists(nameDir3):
    os.mkdir(nameDir3)

vidDir = nameDir3 


print("starting video recorder: ")

record_video_single(model=loaded_model, env_id=ENV_ID, video_length = 400, video_dir=vidDir, \
    render=True, wrapper=PandaWrapper, prefix=ENVNAME + "-" + ENV_ID)

# record_video_multiple(model=loaded_model, n_envs = n_envs, env_id=ENV_ID, video_length = 400, video_dir=vidDir, \
#     render=False, wrapper=PandaWrapper, prefix=POLICYNAME + "-" + ENV_ID)
