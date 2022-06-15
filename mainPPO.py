"""
Goal of this script is to train a Panda Reach SAC model with the trained hyperparameters found during
hyperparameter tuning. Directories are created for saving the model and images created during training.  

"""

# SET DIRECTORIES: 
import os
from datetime import datetime
date = datetime.now().date() 

homeDir = str(os.getcwd() + "/")

pathDir = homeDir + str(date) + "/"
if not os.path.exists(pathDir):
    os.mkdir(pathDir)

modelHomeDir = pathDir + "PPOTuned/"
if not os.path.exists(modelHomeDir):
    os.mkdir(modelHomeDir)


imgDir = modelHomeDir + "images/"
if not os.path.exists(imgDir):
    os.mkdir(imgDir)

callbackDir =  modelHomeDir + "callback/"
if not os.path.exists(callbackDir):
    os.mkdir(os.path.join(callbackDir))


# CUSTOM IMPORTS: 
from SB3.vectEnvs import make_panda_env
import register
from SB3.customCnnShallow_V0 import CustomCNN
from wrappers.pandaWrapperBW_D import PandaWrapper
from SB3.callbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3 import PPO 

# STANDARD IMPORTS
from tabnanny import verbose
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
import torch
import numpy as np


# SETTING PARAMS:
RENDER = True
DEVICE = "cpu"
VERBOSE = 2
ENV_ID = "PandaReachDepthDense-v1"
N_ENVS = 3
EVAL_FREQ = 2e3
OPTIMIZED_N_TIMESTEPS = 6e6

# TUNED HYPERPARAMETER KWARGS

"""
Number of finished trials:  25
Best trial:
  Value:  -6.933871199999999
  Params: 
    gamma: 0.9053598073803767
    gae_lambda: 0.8710966254908115
    lr: 0.0008579475601380259
    n_epochs: 11
    net_arch_width_int: 7
    net_arch_depth: 3
    n_envs: 3
  User attrs:
    vf_coef: 0.75
    clip_range: 0.075
    max_grad_norm: 0.5
    batch_size: 2048
    n_steps: 16384
    ent_coef: 1e-06
    net_arch_width: 128
"""

net_arch_width_int = 7
net_arch_width = 2 ** net_arch_width_int
net_arch_depth = 3
net_arch_array = np.ones(net_arch_depth,dtype=int)
net_arch = [{"pi": (net_arch_array*net_arch_width).tolist(), "vf": (net_arch_array*net_arch_width).tolist()}]

# SETTING HYPERPARAMETERS
kwargs = {
    "policy": "CnnPolicy",
    "device": DEVICE,
    "verbose": VERBOSE,
    "gamma": 0.9053598073803767,
    "gae_lambda": 0.8710966254908115,
    "learning_rate": 0.0008579475601380259,
    "ent_coef": 1e-06,
    "vf_coef": 0.75,
    "clip_range": 0.075,
    "max_grad_norm": 0.5,
    "batch_size": 2048,
    "n_steps": 16384,
    "n_epochs": 11,
    "policy_kwargs": {
        "net_arch": net_arch,
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs":dict(features_dim=128),
    },
}

# RUNNING CODE: 
if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

    vec_env = make_panda_env(env_id=ENV_ID, n_envs=N_ENVS, seed=0, monitor_dir=callbackDir)
    vec_env = VecFrameStack(vec_env, n_stack=N_ENVS)

    print("kwargs: ", kwargs)
    print("n_envs: ", N_ENVS)

    model= PPO(env=vec_env, **kwargs)

    callback = SaveOnBestTrainingRewardCallback(check_freq=EVAL_FREQ, log_dir=callbackDir, verbose=VERBOSE)

    model.learn(total_timesteps=int(OPTIMIZED_N_TIMESTEPS), callback=callback)

    results_plotter.plot_results([callbackDir], OPTIMIZED_N_TIMESTEPS, results_plotter.X_TIMESTEPS, "PPO Training")
    plt.savefig(imgDir +'OptimizedTraining.png')
    


