# WRAPPER IMPORTS
import gym
import numpy as np
from gym import spaces

import cv2 
from PIL import Image
from torchvision import transforms
import torch 

import matplotlib.pyplot as plt

# SUPERVISED LEARNING IMPORTS
from torch import nn
import torch as th
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils import data
import torch.optim as optim
import torchvision.models as models


class PandaWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, env):
    # Call the parent constructor, so we can access self.env later
    super(PandaWrapper, self).__init__(env)
    self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
    self.count = 0

    SqueezeNet = models.squeezenet1_1(pretrained=False)
    self.modelSqueeze = SqueezeNet


    final_conv = nn.Conv2d(512, 3, kernel_size=1)
    self.modelSqueeze.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.Tanh(), nn.AdaptiveAvgPool2d((1, 1))) 
    self.modelSqueeze.load_state_dict(torch.load("/home/hjkwon/Documents/Panda-Robot-RL-Control-with-RGBD-Sensor/supervisedLearning/pandaGraspTrainingData/trainedModelsSqueezenet_Original/model_20220625_115828_400"))

    self.device = torch.device("cpu") # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    self.modelSqueeze.to(self.device )
    print(f'Using {self.device } for inference')


  def image_process(self, img, depth):
    self.count += 1
    new_img = np.delete(img,3,2) 
    img_2 = img
    depth_2 = (depth*255).astype(np.uint8)
    depth_3 = np.expand_dims(depth_2, axis=2)
    img_3 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)
    img_4 = np.expand_dims(img_3, axis=2)
    bw_d = np.append(img_4, depth_3, axis=2)
    zeros = np.zeros([np.shape(bw_d)[0], np.shape(bw_d)[1], 1])
    bw_d_0 = np.concatenate((bw_d, zeros), 2)
    PIL_image = Image.fromarray(np.uint8(bw_d_0)).convert('RGB')

    return PIL_image
   
  def preprocess(self, bw_d_0):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(bw_d_0).unsqueeze(0)
   

  def render_bw_d_0(self):
    img, depth = self.env.render(mode = 'rgb_array', width = 100, height= 100, distance = 1, 
        target_position = [0.3, 0, 0.15], yaw = 75, pitch = -20)
    bw_d_0 = self.image_process(img, depth)
    return bw_d_0

  def render(self, mode='rgb_array'):
    img, depth = self.env.render(mode = 'rgb_array', width = 1080, height= 720) 
    return img
  
  def vect_extractor(self, bw_d_0, obs):
    bw_d_0 = self.preprocess(bw_d_0)
    bw_d_0_GPU = bw_d_0.to(self.device)

    dummy_labels = self.modelSqueeze.forward(bw_d_0_GPU)

    print("observation is: ", obs)
    print("env.sim.get_base_position(object)  is: ", self.env.sim.get_base_position("object"))
    print("dummy_labels.detach().numpy() is: ", dummy_labels.cpu().detach().numpy())

    obs[4:7] =dummy_labels.cpu().detach().numpy()
    return obs

  def reset(self):
    """
    Reset the environment 
    """
    obs_dict = self.env.reset()
    obs = obs_dict["observation"]
    bw_d_0 = self.render_bw_d_0()

    extracted_vector = self.vect_extractor(bw_d_0=bw_d_0, obs=obs)

    return extracted_vector

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """

    # # modify for Panda specific state vector
    obs_dict, reward, done, info = self.env.step(action) 
    obs = obs_dict["observation"]
    bw_d_0 = self.render_bw_d_0()

    extracted_vector = self.vect_extractor(bw_d_0=bw_d_0, obs=obs)

    return extracted_vector, reward, done, info
