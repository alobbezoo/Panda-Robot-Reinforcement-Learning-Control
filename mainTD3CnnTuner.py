"""
Goal of this script is to search the space of hyperaparameters for a Panda Reach TD3 model. 
Directories are created for saving the model, images, and videos created during training.  
After an optimal combination of hyperparameters are found (after specified number of exploratory 
hyperparameter combinations) the optimal hyperparameters are trained for several million steps. 
After training the optimal set of hyperparameters, videos of the exploratory agent are recorded. 

"""
# Create Folders
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

optunaCallbackDir =  modelHomeDir + "optunaCallbackDir/"
if not os.path.exists(optunaCallbackDir):
    os.mkdir(os.path.join(optunaCallbackDir))

optimizedCallbackDir =  modelHomeDir + "optimizedCallbackDir/"
if not os.path.exists(optimizedCallbackDir):
    os.mkdir(os.path.join(optimizedCallbackDir))

# Custom Imports
from SB3.TD3.optimizeTD3CNNDense import TrainOpt
from SB3.TD3.exploreOptunaTD3CnnDense import OptunaFunc
from SB3.customCnnShallow_V0 import CustomCNN

# Standard Imports
from wrappers.pandaWrapperBW_D import PandaWrapper
from SB3.callbacks import TrialEvalCallback
from SB3.optunaPlotter import Plotter

from tabnanny import verbose
import torch
import optuna
import register

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


#Training Constants
N_TRIALS = 60 #NOTE: Check and Update Me
N_STARTUP_TRIALS = 5
N_TIMESTEPS = 8e4 #NOTE: Check and Update Me
EVAL_FREQ = 1e3
N_EVAL_EPISODES = 10
RENDER = False
VERBOSE = 0
ENV_ID = "PandaGraspDepthDense-v1"
DEVICE = "cuda"


OPTIMIZED_N_TIMESTEPS = 1e6 #NOTE: Check and Update Me

optunaClass = OptunaFunc(callbackDir = optunaCallbackDir, n_trials = N_TRIALS, 
        n_timesteps = N_TIMESTEPS, eval_freq =EVAL_FREQ, 
        n_eval_episodes = N_EVAL_EPISODES, env_id = ENV_ID, 
        policy = "CnnPolicy", device=DEVICE, render = RENDER, verbose = VERBOSE,
    )

objective = optunaClass.objective  

if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=10)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    #Exploring the hyperparameter space
    try:
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True) # gc = garbage collection, saves memory
    except KeyboardInterrupt:
        pass

    Plotter(study, imgDir) #Plotting out the outputs

    print("\n \nNumber of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
    
    print("\n \nTRAINING OPTIMIZED STUDY \n \n")


    #Training with the optimized hyperparameters 
    try:
        OptimizedStudy = TrainOpt(evalFreq = EVAL_FREQ, modelHomeDir = modelHomeDir, callbackDir = optimizedCallbackDir, trial=trial, env_ID=ENV_ID, 
            timeSteps=OPTIMIZED_N_TIMESTEPS, verbose=VERBOSE, render=RENDER, device=DEVICE)
        OptimizedStudy.train()
    except KeyboardInterrupt:
        pass


    print("\n \nRECORDING OPTIMIZED STUDY \n \n")

    OptimizedStudy.record_video(video_length=400, prefix='TD3-PandaReach')


