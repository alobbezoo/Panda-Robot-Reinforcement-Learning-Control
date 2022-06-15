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

modelHomeDir = pathDir + "TD3/"
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
from stable_baselines3 import TD3 

# STANDARD IMPORTS
from tabnanny import verbose
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
import torch

# SETTING PARAMS:
RENDER = True
VERBOSE = 2
ENV_ID = "PandaGraspDepthDense-v1"
OPTIMIZED_N_TIMESTEPS = 2e4
N_ENV = 3

# ROUGHLY TUNED HYPERPARAMETER KWARGS
kwargs = {
    "policy": "CnnPolicy",
    "device": "cpu",
    "gamma": 0.99,
    "tau": 0.005, 
    "learning_rate": 0.001,
    "learning_starts": 1000,
    "batch_size": 100,
    "buffer_size": 1000,
    "train_freq": 64,
    # "optimize_memory_usage": "True",
    "verbose": 2,
    "policy_kwargs": {
        "net_arch": [128, 128],
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": dict(features_dim=128),
    },
}

# RUNNING CODE: 
if __name__ == "__main__":

    vec_env = make_panda_env(env_id=ENV_ID, n_envs=N_ENV, seed=0, monitor_dir=callbackDir)
    # vec_env = VecFrameStack(vec_env, n_stack=N_ENV)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=callbackDir, verbose=2)

    model = TD3(env=vec_env, **kwargs)

    # Train the agent
    model.learn(total_timesteps=int(OPTIMIZED_N_TIMESTEPS), callback=callback, log_interval=1000)


    results_plotter.plot_results([callbackDir], OPTIMIZED_N_TIMESTEPS, results_plotter.X_TIMESTEPS, "TD3 Training")
    plt.savefig(imgDir +'OptimizedTraining.png')
    


