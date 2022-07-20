from pprint import pp
from stable_baselines3 import PPO, SAC, TD3
from SB3.customCnnShallow_V0 import CustomCNN

import numpy as np
        
class Hyperparameters():
  def __init__(self, envName, algorithm, type):
    self.envName = envName
    self.algorithm = algorithm
    self.type = type
    print("true/false: ", (self.envName == "PandaGraspDepthDense-v1") and (self.algorithm == SAC) \
      and (self.type == "vect"))


  def ppoNetArch(self, net_arch_width_int, net_arch_depth):
    net_arch_width = 2 ** net_arch_width_int
    net_arch_array = np.ones(net_arch_depth,dtype=int)
    net_arch = [{"pi": (net_arch_array*net_arch_width).tolist(), "vf": (net_arch_array*net_arch_width).tolist()}]
    return net_arch
  
  def sacNetArch(self, net_arch_width_int, net_arch_depth):
    net_arch_width = 2 ** net_arch_width_int   
    net_arch_array = np.ones(net_arch_depth,dtype=int)

    net_arch = (net_arch_array*net_arch_width).tolist()
    return net_arch


  def hyperparams(self):


# VECT TUNING


  # PICK AND PLACE

    if (self.envName == (self.envName == "PandaPickAndPlaceDepthBlockDense-v1") or (self.envName == "PandaPickAndPlaceDepthDense-v1") \
      and (self.algorithm == PPO) and (self.type == "vect")):

      """
      Best trial:
        Value:  -13.7
        Params: 
          gamma: 0.9270335099881851
          gae_lambda: 0.8237685826071883
          lr: 0.0013378119211869901
          n_epochs: 9
          batch_size: 10
          net_arch_width_int: 7
          net_arch_depth: 5
          n_envs: 3
        User attrs:
          vf_coef: 0.75
          clip_range: 0.075
          max_grad_norm: 0.5
          batch_size: 1024
          n_steps: 32768
          ent_coef: 1e-06
          net_arch_width: 128

      """
      net_arch = self.ppoNetArch(net_arch_width_int=7, net_arch_depth=5)

      kwargs = {
          "gamma": 0.9270335099881851,
          "gae_lambda": 0.8237685826071883,
          "learning_rate": 0.0013378119211869901,
          "ent_coef": 1e-06,
          "max_grad_norm": 0.5,
          "vf_coef": 0.75,
          "clip_range": 0.075,
          "batch_size": 1024,
          "n_steps": 32768,
          "n_epochs": 9,
          "verbose": 1,
          "policy_kwargs": {
              "net_arch": net_arch,
          },
      }

      print("HEREREER")

      return kwargs




    elif (self.envName == (self.envName == "PandaPickAndPlaceDepthBlockDense-v1") or (self.envName == "PandaPickAndPlaceDepthDense-v1") \
      and (self.algorithm == SAC) and (self.type == "vect")):

      """
      {'policy': 'MlpPolicy', 'device': 'cpu', 'verbose': 0, 'use_sde': 'True', 'use_sde_at_warmup': 'True', 1
      'ent_coef': 0.0011449869603157097, 'gamma': 0.9909845566299015, 'tau': 0.0014339706991093016, 
      'learning_rate': 0.0010905402525877687, 'batch_size': 8192, 'learning_starts': 6000, 'buffer_size': 50000, 
      'train_freq': 50, 'gradient_steps': 50, 'policy_kwargs': {'net_arch': [64, 64, 64, 64]}}
      n_envs:  5
      action_noise:  0.15000000000000002

      Trial 27 finished with value: -7.5215306 and parameters: 
      {'gamma': 0.9909845566299015, 'tau': 0.0014339706991093016, 'lr': 0.0010905402525877687, 
      'ent_coef': 0.0011449869603157097, 'batch_size_num': 13, 'learning_starts': 6, 
      'action_noise_int': 4, 'net_arch_width_int': 6, 'net_arch_depth': 4}. 
      Best is trial 27 with value: -7.5215306.[0m

 

      """

      net_arch = self.sacNetArch(net_arch_width_int=6, net_arch_depth=4)

      kwargs = {
          "ent_coef": 0.0011449869603157097,
          "gradient_steps": 50, 
          "train_freq": 50,
          "gamma": 0.9909845566299015,
          "tau": 0.0014339706991093016,
          "learning_rate": 0.0010905402525877687,
          "batch_size": 8192,
          "learning_starts": 6000,
          "buffer_size": 50000,
          "policy_kwargs": {
              "net_arch": net_arch,
          },
      }

      actionNoiseInt = 0.15
      n_envs = 5
      

      return kwargs, n_envs, actionNoiseInt



  #GRASP AND REACH

    # elif (self.envName == (self.envName == "PandaGraspDepthBlockDense-v1") or (self.envName == "PandaGraspDepthDense-v1") \
    #   or (self.envName == "PandaReachDepthDense-v1") and (self.algorithm == PPO) and (self.type == "vect")):

    #   """
    #   Number of finished trials:  22
    #   Best trial:
    #     Value:  -3.0011657
    #     Params: 
    #       gamma: 0.9093373345474445
    #       gae_lambda: 0.9290620915181558
    #       lr: 0.003576327533197799
    #       n_epochs: 7
    #       batch_size: 12
    #       net_arch_width_int: 7
    #       net_arch_depth: 5
    #       n_envs: 4
    #     User attrs:
    #       vf_coef: 0.75
    #       clip_range: 0.075
    #       max_grad_norm: 0.5
    #       batch_size: 4096
    #       n_steps: 16384
    #       ent_coef: 1e-06
    #       net_arch_width: 128
    #       n_envs = 4
    #   """
    #   print("HEREREER")

    #   net_arch = self.ppoNetArch(net_arch_width_int=7, net_arch_depth=5)


    #   kwargs = {
    #       "gamma": 0.9093373345474445,
    #       "gae_lambda": 0.9290620915181558,
    #       "learning_rate": 0.003576327533197799,
    #       "ent_coef": 1e-06,
    #       "max_grad_norm": 0.5,
    #       "vf_coef": 0.75,
    #       "clip_range": 0.075,
    #       "batch_size": 4096,
    #       "n_steps": 16384,
    #       "n_epochs": 7,
    #       "verbose": 1,
    #       "policy_kwargs": {
    #           "net_arch": net_arch,
    #       },
    #   }

    #   return kwargs


    elif (self.envName == (self.envName == "PandaGraspDepthBlock-v1") or (self.envName == "PandaGraspDepth-v1") \
          or (self.envName == "PandaReachDepth-v1") and (self.algorithm == PPO) and (self.type == "vect")):

          """
          Best trial:
            Value:  -13.7
            Params: 
              gamma: 0.9270335099881851
              gae_lambda: 0.8237685826071883
              lr: 0.0013378119211869901
              n_epochs: 9
              batch_size: 10
              net_arch_width_int: 7
              net_arch_depth: 5
              n_envs: 3
            User attrs:
              vf_coef: 0.75
              clip_range: 0.075
              max_grad_norm: 0.5
              batch_size: 1024
              n_steps: 32768
              ent_coef: 1e-06
              net_arch_width: 128
          """

          net_arch = self.ppoNetArch(net_arch_width_int=7, net_arch_depth=5)


          kwargs = {
              "gamma": 0.9270335099881851,
              "gae_lambda": 0.8237685826071883,
              "learning_rate": 0.0013378119211869901,
              "ent_coef": 1e-06,
              "max_grad_norm": 0.5,
              "vf_coef": 0.75,
              "clip_range": 0.075,
              "batch_size": 1024,
              "n_steps": 32768,
              "n_epochs": 9,
              "verbose": 1,
              "policy_kwargs": {
                  "net_arch": net_arch,
              },
          }

          return kwargs


    elif (self.envName == (self.envName == "PandaGraspDepthBlockDense-v1") or (self.envName == "PandaGraspDepthDense-v1") \
      or (self.envName == "PandaReachDepthDense-v1") and (self.algorithm == SAC) and (self.type == "vect")):  
  
      """
      Number of finished trials:  47
      Best trial:
      Value:  -0.3323002
      Params: 
          gamma: 0.9562861389598207
          tau: 0.020984209655387977
          lr: 0.0016125789033604766
          batch_size_num: 11
          learning_starts: 7
          train_freq_int: 2
          action_noise_int: 1
          net_arch_width_int: 6
          net_arch_depth: 4
          n_envs: 2
      User attrs:
          learning_rate: 0.0016125789033604766
          batch_size: 2048
          learning_starts: 7000
          buffer_size: 50000
          train_freq: 100
          net_arch_width: 64
          net_arch_depth: 4
          gradient_steps: 100
          action_noise: 0.0
      """

      net_arch = self.sacNetArch(net_arch_width_int=7, net_arch_depth=3)


      kwargs = {
          "gradient_steps": 100, 
          "train_freq": 100,
          "gamma": 0.9562861389598207,
          "tau": 0.020984209655387977,
          "learning_rate": 0.0016125789033604766,
          "batch_size": 2048,
          "learning_starts": 10000,
          "buffer_size": 50000,
          "policy_kwargs": {
              "net_arch": net_arch,
          },
      }

      actionNoiseInt = 0.0
      # actionNoiseInt = 0.15
      n_envs = 2

      print("HEREREER")

      return kwargs, n_envs, actionNoiseInt




    elif (self.envName == (self.envName == "PandaGraspDepthBlock-v1") or (self.envName == "PandaGraspDepth-v1") \
      or (self.envName == "PandaReachDepth-v1") and (self.algorithm == SAC) and (self.type == "vect")):  
  
      """
      Number of finished trials:  100
      Best trial:
      Value:  -3.0
      Params: 
          gamma: 0.9795669800650622
          tau: 0.0028732553225876887
          lr: 0.0009069077086214123
          ent_coef: 0.008820073767094497
          batch_size_num: 8
          learning_starts: 9
          action_noise_int: 4
          n_envs: 4
      User attrs:
          learning_rate: 0.0009069077086214123
          batch_size: 256
          learning_starts: 9000
          buffer_size: 50000
          train_freq: 50
          net_arch_width: 256
          net_arch_depth: 3
          gradient_steps: 50
          action_noise: 0.15000000000000002
          ent_coef: 0.008820073767094497
          n_envs: 4
      """

      net_arch = self.sacNetArch(net_arch_width_int=8, net_arch_depth=3)


      kwargs = {
          "gradient_steps": 50, 
          "train_freq": 50,
          "gamma": 0.9795669800650622,
          "tau": 0.0028732553225876887,
          "learning_rate": 0.0009069077086214123,
          "ent_coef": 0.008820073767094497,
          "batch_size": 256,
          "learning_starts": 9000,
          "buffer_size": 50000,
          "policy_kwargs": {
              "net_arch": net_arch,
          },
      }

      actionNoise = 0.15

      print("HERE")

      return kwargs, actionNoise





# CNN TUNING

  # REACH

    elif (self.envName == (self.envName == "PandaReachDepthBlockDense-v1") or (self.envName == "PandaReachDepthDense-v1") \
      and (self.algorithm == PPO) and (self.type == "cnn")):  

      net_arch = self.ppoNetArch(net_arch_width_int=7, net_arch_depth=3)


      kwargs = {
          "gamma": 0.9053598073803767,
          "gae_lambda": 0.8710966254908115,
          "learning_rate": 0.0008579475601380259,
          "ent_coef": 1e-06,
          "max_grad_norm": 0.5,
          "vf_coef": 0.75,
          "clip_range": 0.075,
          "batch_size": 2048,
          "n_steps": 16384,
          "n_epochs": 11,
          "policy_kwargs": {
              "net_arch": net_arch,
          },
      }

      return kwargs


    elif (self.envName == (self.envName == "PandaReachDepthBlockDense-v1") or (self.envName == "PandaReachDepthDense-v1")  \
       and (self.algorithm == SAC) and (self.type == "cnn")):  

      """
      {'policy': 'CnnPolicy', 'device': 'cuda', 'verbose': 0, 'use_sde': 'True', 'use_sde_at_warmup': 'True', 
      'ent_coef': 'auto', 'gamma': 0.9815349183392321, 'tau': 0.02884082120552075, 'learning_rate': 1.0566976937791386e-05, 
      'batch_size': 256, 'learning_starts': 5000, 'buffer_size': 50000, 'train_freq': 50, 'gradient_steps': 50, 
      'policy_kwargs': {'net_arch': [256, 256, 256, 256], 
      'features_extractor_class': <class 'SB3.customCnnShallow_V0.CustomCNN'>, 'features_extractor_kwargs': {'features_dim': 128}}}
      n_envs:  4
      action_noise:  0.05

      Trial 11 finished with value: -13.5151115 and parameters: {'gamma': 0.9815349183392321, 
      'tau': 0.02884082120552075, 'lr': 1.0566976937791386e-05, 'batch_size_num': 8, 'learning_starts': 5, 
      'train_freq': 1, 'action_noise_int': 2, 'net_arch_width_int': 8, 'net_arch_depth': 4}. 
      Best is trial 10 with value: -6.748622999999999.
      """
      
      """
      PANDAREACH SAC CNN July 12
      Number of finished trials:  19
      Best trial:
      Value:  -6.303952600000001
      Params: 
          gamma: 0.9864406015682582
          tau: 0.03292173459733876
          lr: 5.218340576030402e-07
          batch_size_num: 8
      User attrs:
          learning_rate: 5.218340576030402e-07
          batch_size: 256
          learning_starts: 5000
          buffer_size: 50000
          train_freq: 100
          net_arch_width: 128
          net_arch_depth: 4
          gradient_steps: 100
          action_noise: 0.1
          ent_coef: auto

      """


      net_arch = self.sacNetArch(net_arch_width_int=7, net_arch_depth=4)


      kwargs = {
          "gradient_steps": 100, 
          "train_freq": 100,
          "gamma": 0.9864406015682582,
          "tau": 0.03292173459733876,
          "learning_rate": 5.218340576030402e-07,
          "batch_size": 256,
          "learning_starts": 5000,
          "buffer_size": 50000,
          "ent_coef": "auto",
          "policy_kwargs": {
              "net_arch": net_arch,
              "features_extractor_class": CustomCNN,
              "features_extractor_kwargs": dict(features_dim=128),
          },
      }

      actionNoise = 0.1
      n_envs = 4

      return kwargs, n_envs, actionNoise




  # GRASP

    elif (self.envName == (self.envName == "PandaGraspDepthBlockDense-v1") or (self.envName == "PandaGraspDepthDense-v1") \
      and (self.algorithm == SAC) and (self.type == "cnn")):  

      """
      Value:  -5.4568439
      Params: 
          gamma: 0.9703915849602966
          tau: 0.0005415492504332224
          lr: 0.001490086797041379
          learning_starts: 1
          buffer_size_num: 4
          action_noise_int: 1
      User attrs:
          learning_rate: 0.001490086797041379
          batch_size: 512
          learning_starts: 1000
          buffer_size: 20000
          train_freq: 100
          net_arch_width: 256
          net_arch_depth: 3
          gradient_steps: 100
          action_noise: 0.0
          n_envs = 4

      """

      net_arch = self.sacNetArch(net_arch_width_int=8, net_arch_depth=3)


      kwargs = {
          "gradient_steps": 100, 
          "train_freq": 100,
          "gamma": 0.9703915849602966,
          "tau": 0.0005415492504332224,
          "learning_rate": 0.001490086797041379,
          "batch_size": 512,
          "learning_starts": 1000,
          "buffer_size": 20000,
          "policy_kwargs": {
              "net_arch": net_arch,
              "features_extractor_class": CustomCNN,
              "features_extractor_kwargs": dict(features_dim=128),
          },
      }

      actionNoiseInt = 0.0
      n_envs = 4

      return kwargs, n_envs, actionNoiseInt


