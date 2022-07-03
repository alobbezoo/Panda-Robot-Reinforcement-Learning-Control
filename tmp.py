import gym
import gym.spaces
import os
import torch
import time
time.sleep(3)
env = gym.make("PandaReachDense-v2", render=True)

obs= env.reset()

# print("here \n \n")
done=False
N_SAMPLES = 10000
i=0

for i in range(200):
    while done==False:
        print(obs)
        # action = -(obs["observation"][0:3] - obs["observation"][3:6] ) 
        action = env.action_space.sample()
        # action = -(obs["observation"] - obs["desired_goal"])

        obs, _, done, _ = env.step(3*action)
        
        # print("obs is: ", obs)
        i+=1
        time.sleep(0.2)
    print("i is: ", i)
    env.reset()
    done = False


env.close()
