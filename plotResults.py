#IMPORTS:
from SB3.plotter import plot_results
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np

# # INPUTS:
# POLICY = "PPO"
# TRAINING = "Vect"
# PROBLEM = "PandaPickandPlace"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaPickandPlace/Vect/PPO/PPO_PandaPickandPlace2_Vect/"
# MAXLENGTH_PERCENTAGE = 0.3 #1 #0.28

# # INPUTS:
# POLICY = "SAC"
# TRAINING = "Vect"
# PROBLEM = "PandaPickandPlace"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaPickandPlace/Vect/SAC/"
# MAXLENGTH_PERCENTAGE = 1 #0.28

# #INPUTS:
# POLICY = "SAC"
# TRAINING = "Vect"
# PROBLEM = "PandaReach"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Vect/SAC/Sparse/"
# MAXLENGTH_PERCENTAGE = 0.2 #0.28

# #INPUTS: ONLY PLOTTED FOR FEW STEPS, POOR PERFORMANCE
# POLICY = "PPO"
# TRAINING = "Vect"
# PROBLEM = "PandaGrasp"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/vectTraining/2022-04-22/optimizedCustVect/"
# MAXLENGTH_PERCENTAGE = 1

# #INPUTS:
# POLICY = "PPO"
# TRAINING = "CNN"
# PROBLEM = "PandaGrasp"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaGrasp/CNN/PPO/images/"
# MAXLENGTH_PERCENTAGE = 1

# #INPUTS:
# POLICY = "SAC"
# TRAINING = "CNN"
# PROBLEM = "PandaGrasp"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/cnnTraining/2022-06-02/SAC/optimizedCallbackDir/"
# MAXLENGTH_PERCENTAGE = 1 #0.5 #0.28

#INPUTS:
POLICY = "SAC"
TRAINING = "CNN"
PROBLEM = "PandaReach"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/cnnTraining/2022-07-13/SACCNNReach/callback/"
MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Cnn/SAC"
MAXLENGTH_PERCENTAGE = 1 #0.5 #0.28

# evaluations = np.load('/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/cnnTraining/2022-07-13/SACCNNReach/callback/results.npy')
# print("evaluations are: ", evaluations)

# #INPUTS:
# PROBLEM = "PandaReach"
# TRAINING = "Vect"
# POLICY = "SAC"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Vect/SAC/"
# MAXLENGTH_PERCENTAGE = 0.25 #0.28

# #INPUTS:
# PROBLEM = "PandaReach"
# TRAINING = "Vect"
# POLICY = "PPO"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaReach/Vect/PPO/Sparse/"
# MAXLENGTH_PERCENTAGE = 1 #0.25 #0.28

# # INPUTS:
# PROBLEM = "PandaGrasp"
# TRAINING = "Vect"
# POLICY = "SAC"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/trainedModelsPlots/PandaGrasp/Vect/SAC/"
# MAXLENGTH_PERCENTAGE = .6 #0.5 #0.28

# ## INPUTS:
# PROBLEM = "PandaGrasp"
# TRAINING = "CNN"
# POLICY = "SAC"
# MODELPATH = "/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/cnnTraining/2022-06-16/SAC/optimizedCallbackDir/"
# MAXLENGTH_PERCENTAGE = 1 #0.5 #0.28

def plotTitle(window = 1, policy=POLICY, training=TRAINING, problem=PROBLEM): 
    return str(policy + " " + training + problem + " " + "Tuned Training Curve 3 Window=" + str(window))

def saveTitle(window = 1, policy=POLICY, training=TRAINING, problem=PROBLEM): 
    return str(policy + "_" + training + problem + "_" + "Tuned_Training_Curve3_Window_" + str(window))

# SET DIRECTORIES: 
import os
from datetime import datetime
date = datetime.now().date() 

homeDir = str(os.getcwd() + "/")

pathDir = homeDir + "trainedModelsPlots/"
if not os.path.exists(pathDir):
    os.mkdir(pathDir)

nameDir1 = pathDir + PROBLEM + "/"
if not os.path.exists(nameDir1):
    os.mkdir(nameDir1)

nameDir2 = nameDir1 + TRAINING + "/"
if not os.path.exists(nameDir2):
    os.mkdir(nameDir2)

nameDir3 = nameDir2 + POLICY + "/"
if not os.path.exists(nameDir3):
    os.mkdir(nameDir3)


imgDir = nameDir3 + "images/"
if not os.path.exists(imgDir):
    os.mkdir(imgDir)

#PLOTTING
plot_results(log_folder=MODELPATH, window_size=1, title=plotTitle(window=1), img_folder=imgDir,\
    img_name = saveTitle(window=1), maxlengthpercentage=MAXLENGTH_PERCENTAGE)

plot_results(log_folder=MODELPATH, window_size=50, title=plotTitle(window=50), img_folder=imgDir,\
    img_name = saveTitle(window=50), maxlengthpercentage=MAXLENGTH_PERCENTAGE)

plot_results(log_folder=MODELPATH, window_size=100, title=plotTitle(window=100), img_folder=imgDir,\
    img_name =  saveTitle(window=100), maxlengthpercentage=MAXLENGTH_PERCENTAGE)

plot_results(log_folder=MODELPATH, window_size=500, title=plotTitle(window=500), img_folder=imgDir,\
    img_name =  saveTitle(window=500), maxlengthpercentage=MAXLENGTH_PERCENTAGE)
