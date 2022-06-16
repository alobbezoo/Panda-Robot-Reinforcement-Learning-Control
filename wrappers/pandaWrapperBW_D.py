"""
Script to serve as a wrapper for the panda robot. Inital state feedback is vector, but wrapper changes state
feedback to greyscale and depth (2 channel image)

"""

import gym
import numpy as np
from gym import spaces
import cv2 
from PIL import Image
from torchvision import transforms
import torch 

import matplotlib.pyplot as plt

class PandaWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, env):
    # Call the parent constructor, so we can access self.env later
    super(PandaWrapper, self).__init__(env)
    # NOTE: The super() builtin allows us to access methods of the base class.
    self.observation_space = spaces.Box(low=np.zeros((2,100,100)), high=np.ones((2,100,100))*255, shape=(2,100,100), dtype=np.uint8)
    self._max_episode_steps = 50
    self.step_counter = 0

    
  def image_process(self, img, depth):

    depth_1 = (np.expand_dims(depth, axis=2)*255).astype(np.uint8) # 0-255 int intensity
    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #convert to greyscale
    img_2 = np.expand_dims(img_1, axis=2)
    bw_d_rearrange = np.append(img_2, depth_1, axis=2)
    zeros = np.zeros([np.shape(bw_d)[0], np.shape(bw_d)[1], 1])
    bw_d = np.moveaxis(bw_d_rearrange,2,0)

    return bw_d
      
  def reset(self):
    """
    Reset the environment 
    """
    obs_dict = self.env.reset()

    img, depth = self.env.render(mode = 'rgb_array',width = 100, height= 100, distance = 1, 
      target_position = [-0.1, 0, 0.1], yaw = 60, pitch = -30) # better orientation for pick and place

    obs = self.image_process(img, depth)

    return obs
  
  def render(self, mode='rgb_array'): # NOTE: Render function is needed for rendering out the video

    img, depth = self.env.render(mode = 'rgb_array', width = 1080, height= 720) 
    return img

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """

    # modify for Panda specific state vector
    obs_dict, reward, done, info = self.env.step(action) 
      
    img, depth = self.env.render(mode = 'rgb_array', width = 100, height= 100, distance = 1, 
      target_position = [-0.1, 0, 0.1], yaw = 60, pitch = -30) 


    obs = self.image_process(img, depth)


    return obs, reward, done, info
