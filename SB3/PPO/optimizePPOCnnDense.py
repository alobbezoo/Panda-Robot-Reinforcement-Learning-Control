# NOTE: Goal of this script is to create class for sampling and training optuna hyperparameter combinations

# Custom Imports
from wrappers.pandaWrapperBW_D import PandaWrapper
from SB3.customCnnShallow_V0 import CustomCNN
from SB3.PPO.exploreOptunaPPOCnnDense import make_panda_env
from SB3.callbacks import SaveOnBestTrainingRewardCallback

#Standard Imports
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

import matplotlib.pyplot as plt
import numpy as np
import os
import gym

# Video Rendering: 
import base64
from pathlib import Path
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


class TrainOpt():
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(
        self, trial, env_ID, timeSteps, modelHomeDir, callbackDir, policy = "CnnPolicy", device="cpu", verbose=2, render = False
    ):

        self.trial = trial
        self.env_ID = env_ID
        self.timeSteps = timeSteps
        self.policy = policy
        self.device = device
        self.verbose = verbose
        self.render = render
        self.modelHomeDir = modelHomeDir
        self.callbackDir = callbackDir


    def optimal(self):
        learning_rate = self.trial.params['lr']
        gae_lambda = self.trial.params['gae_lambda']
        n_epochs = self.trial.params['n_epochs']
        gamma = self.trial.params['gamma']
        net_arch_depth = self.trial.params['net_arch_depth']
        

        vf_coef = self.trial.user_attrs['vf_coef']
        clip_range = self.trial.user_attrs['clip_range']
        max_grad_norm = self.trial.user_attrs['max_grad_norm']
        batch_size = self.trial.user_attrs['batch_size']
        n_steps = self.trial.user_attrs['n_steps']
        ent_coef = self.trial.user_attrs['ent_coef']
        net_arch_width = self.trial.user_attrs['net_arch_width']

        net_arch_array = np.ones(net_arch_depth,dtype=int)
        net_arch = [{"pi": (net_arch_array*net_arch_width).tolist(), "vf": (net_arch_array*net_arch_width).tolist()}]

        self.policy_kwargs = {
            "net_arch": net_arch,
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs":dict(features_dim=128),
            }

        self.n_envs = self.trial.params['n_envs']

        kwargs = {
            "policy": self.policy, 
            "device": self.device,
            "verbose": self.verbose, 
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "clip_range": clip_range,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_size, 
            "n_steps": n_steps, 
            "n_epochs": n_epochs, 
            "policy_kwargs": self.policy_kwargs,
        }

        return kwargs

    def train(self):

        kwargs = self.optimal()
        print(kwargs)

        self.vec_env = make_panda_env(env_id=self.env_ID, n_envs=self.n_envs, seed=0, monitor_dir=self.callbackDir)
        self.vec_env_stack = VecFrameStack(self.vec_env, n_stack=self.n_envs)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.callbackDir, verbose=1)

        self.model= PPO(env=self.vec_env_stack, **kwargs)

        self.model.learn(total_timesteps=int(self.timeSteps), callback=callback)

        results_plotter.plot_results([self.callbackDir], self.timeSteps, results_plotter.X_TIMESTEPS, str("PPO" + self.env_ID))
        plt.savefig(self.modelHomeDir +'images/optimize_plot.png')


    def record_video(self, video_length=400, prefix=''):
        """
        :param env_id: (str)
        :param model: (RL model)
        :param video_length: (int)
        :param prefix: (str)
        :param video_folder: (str)
        """
        video_folder= self.modelHomeDir+'videos/'
        eval_env = DummyVecEnv([lambda: PandaWrapper(gym.make(self.env_ID))]) 
        #NOTE: Had to add the wrapper for proper reading
        
        # Start the video at step=0 and record 500 steps
        eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                    record_video_trigger=lambda step: step == 0, video_length=video_length,
                                    name_prefix=prefix)

        obs = eval_env.reset()
        for _ in range(video_length):
            action, _ = self.model.predict(obs)
            obs, _, _, _ = eval_env.step(action)
