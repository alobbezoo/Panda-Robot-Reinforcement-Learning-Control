"""
Script is used to create a wrapper for a 3 channel greyscale, depth and 0 channel image. The ideal is that the image 
feedback from this code can be implemented for pre-built pybullet Resnet, Googlenet, etc

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
    self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)

  def image_process(self, img, depth):
    depth_2 = np.expand_dims(depth, axis=2)
    img_2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_3 = np.expand_dims(img_2, axis=2)
    bw_d = np.append(img_3, depth_2, axis=2)
    zeros = np.zeros([np.shape(bw_d)[0], np.shape(bw_d)[1], 1])
    bw_d_0 = np.concatenate((bw_d, zeros), 2)
    PIL_image = Image.fromarray(np.uint8(bw_d_0)).convert('RGB')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(PIL_image)
    img = input_tensor.numpy()

    return img
   
  def render_bw_d_0(self):

    img, depth = self.env.render(mode = 'rgb_array', width = 100, height= 100, distance = 1, 
      target_position = [0.3, 0, 0.15], yaw = 70, pitch = -20)

    bw_d_0 = self.image_process(img, depth)
    
    return bw_d_0
  
  def render(self, mode='rgb_array'):

    img, depth = self.env.render(mode = 'rgb_array', width = 1080, height= 720, distance = 2, 
      target_position = [-0.25, 0, 0], yaw = 0, pitch = -5) 
    # im = Image.fromarray(img)
    # print("CALL!")
    # im.save("test.png")

    return img

  def reset(self):
    """
    Reset the environment 
    """
    obs_dict = self.env.reset()
    obs = obs_dict["observation"]

    return obs

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """

    # # modify for Panda specific state vector
    obs_dict, reward, done, info = self.env.step(action)
    obs = obs_dict["observation"]

    return obs, reward, done, info
