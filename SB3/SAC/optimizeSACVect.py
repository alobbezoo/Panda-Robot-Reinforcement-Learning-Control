# NOTE: Goal of this script is to create class for sampling and training optuna hyperparameter combinations

# Custom Imports
# from wrappers.pandaWrapperVect_Grasp_BW_D_0 import PandaWrapper #NOTE: Unique to Vect
from curses import wrapper
# from SB3.customCnnShallow_V0 import CustomCNN
from SB3.PPO.exploreOptunaPPOCnnDense import make_panda_env
from SB3.callbacks import SaveOnBestTrainingRewardCallback

#Standard Imports
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise, OrnsteinUhlenbeckActionNoise

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

# Create Folders
import os


class TrainOpt():
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(
        self, evalFreq, trial, env_ID, timeSteps, modelHomeDir, callbackDir, wrapper, policy = "CnnPolicy", \
            device="cpu", verbose=2, render = False
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
        self.evalFreq = evalFreq
        self.wrapper=wrapper


    def optimal(self):
        gamma = self.trial.params['gamma']
        tau = self.trial.params['tau']

        net_arch_depth = self.trial.user_attrs['net_arch_depth']
        learning_rate = self.trial.user_attrs['learning_rate']
        batch_size= self.trial.user_attrs['batch_size']
        learning_starts = self.trial.user_attrs['learning_starts']
        buffer_size = self.trial.user_attrs['buffer_size']
        train_freq = self.trial.user_attrs['train_freq']
        net_arch_width = self.trial.user_attrs['net_arch_width']
        gradient_steps = self.trial.user_attrs["gradient_steps"]
        ent_coef = self.trial.user_attrs["ent_coef"]
        
        self.n_envs = self.trial.user_attrs['n_envs']

        net_arch_array = np.ones(net_arch_depth,dtype=int)
        net_arch = (net_arch_array*net_arch_width).tolist()

        self.policy_kwargs = {
            "net_arch": net_arch,
            }

        self.action_noise = self.trial.user_attrs['action_noise']

        kwargs = {
            "policy": self.policy, 
            "device": self.device,
            "verbose": self.verbose, 
            "use_sde": "True",
            "use_sde_at_warmup": "True",
            # "ent_coef": "auto",
            "ent_coef":ent_coef, 
            "gradient_steps": gradient_steps, 
            "gamma": gamma,
            "tau": tau,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "buffer_size": buffer_size,
            "train_freq": train_freq,
            "policy_kwargs": self.policy_kwargs,
        }

        return kwargs

    def train(self):

        kwargs = self.optimal()
        
        print(kwargs)

        vec_env = make_panda_env(env_id=self.env_ID, n_envs=self.n_envs, seed=0, monitor_dir=self.callbackDir, wrapper=self.wrapper)
        # self.vec_env_stack = VecFrameStack(vec_env, n_stack=self.n_envs)

        callback = SaveOnBestTrainingRewardCallback(check_freq=self.evalFreq, log_dir=self.callbackDir) #, verbose=self.verbose)

        # Add some action noise for exploration
        n_actions = vec_env.action_space.shape[-1]
        base_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=self.action_noise * np.ones(n_actions))
        action_noise = VectorizedActionNoise(base_noise=base_noise,  n_envs=self.n_envs)

        self.model= SAC(env=vec_env, action_noise=action_noise, **kwargs)

        self.model.learn(total_timesteps=int(self.timeSteps), callback=callback)
        # self.model.save()

        results_plotter.plot_results([self.callbackDir], self.timeSteps, results_plotter.X_TIMESTEPS, str("SAC" + self.env_ID))
        plt.savefig(self.modelHomeDir +'images/optimize_plot.png')
        # self.model.save(path=(self.modelHomeDir +'final_resulting_model.png'))


    def record_video(self, video_length=400, prefix=''):
        """
        :param env_id: (str)
        :param model: (RL model)
        :param video_length: (int)
        :param prefix: (str)
        :param video_folder: (str)
        """
        video_folder= self.modelHomeDir+'videos/'
        eval_env = DummyVecEnv([lambda: self.wrapper(gym.make(self.env_ID))]) 

        #NOTE: Had to add the wrapper for proper reading
        
        # Start the video at step=0 and record 500 steps
        eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                    record_video_trigger=lambda step: step == 0, video_length=video_length,
                                    name_prefix=prefix)

        obs = eval_env.reset()
        for _ in range(video_length):
            action, _ = self.model.predict(obs)
            obs, _, _, _ = eval_env.step(action)

