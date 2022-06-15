from stable_baselines3 import PPO

# Create Folders
import os
from datetime import datetime

date = datetime.now().date() 
homeDir = str(os.getcwd() + "/")

# Load model from callback saver
model = PPO.load(str(homeDir + "2022-04-22/savedModels/model_20220422_162247"))
print(model)

"""
# Load model from optimized callback saver
model2 = PPO.load(str(homeDir + "2022-04-22/optimizedCustVect/best_model.zip"))
print(model2)
"""
