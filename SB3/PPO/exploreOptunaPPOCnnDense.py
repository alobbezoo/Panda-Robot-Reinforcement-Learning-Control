
# Custom Imports
from wrappers.pandaWrapperBW_D import PandaWrapper
from SB3.customCnnShallow_V0 import CustomCNN
from SB3.callbacks import TrialEvalCallback
from SB3.vectEnvs import make_panda_env

# Standard Imports
import optuna

from stable_baselines3 import PPO

# Custom Callback:
from stable_baselines3.common.vec_env import VecEnv


# from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
# from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

import random
import gym
import numpy as np
from typing import Any, Dict
import os

class OptunaFunc():
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """
    def __init__(
        self, callbackDir, n_trials:int = 50, n_timesteps = 1000, eval_freq =100, 
        n_eval_episodes = 3, env_id = "PandaReachDepth-v1", render = False, verbose = 2, 
        policy = "CnnPolicy", device="cuda", 
    ):
        self.callbackDir = callbackDir
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.env_id = env_id
        self.policy = policy
        self.device = device
        self.verbose = verbose
        self.render = render

        self.default_hyperparams = {
            "policy": self.policy, 
            "device": self.device,
            "verbose": self.verbose,
}

    def sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sampler for A2C hyperparameters."""
        gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99, log=True)
        learning_rate = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        n_epochs = trial.suggest_int("n_epochs", 6, 14, log=True) 

        vf_coef = 0.75 #trial.suggest_float("vf_coef", 0.35, 1, log=True)
        clip_range = 0.075 #trial.suggest_float("clip_range", 0.05, 0.4, log=True)
        max_grad_norm = 0.5  #trial.suggest_float("max_grad_norm", 0.3, 3.0, log=True)
        ent_coef = 1e-6 #trial.suggest_float("ent_#coef", 1e-8 , 1e-1, log=True)
        n_steps = 2 ** 10 #NOTE
        # trial.suggest_int("exponent_n_steps", self.exponent_n_steps_min, 
        # self.exponent_n_steps_max, log=True) # trying 10 and 14 instead of 10

        # batch_size_num = 2**trial.suggest_int("batch_size_num", 1, 4, log=True)
        batch_size = 2 ** 9 #2048 #NOTE
        #int(n_steps / 8) #int(n_steps / 8)
        # NOTE: I believe batch size of 8192 was causing the gpu to crash


        net_arch_width_int = trial.suggest_int("net_arch_width_int", 6, 8)
        net_arch_width = 2 ** net_arch_width_int
        net_arch_depth = trial.suggest_int("net_arch_depth", 3, 5)
        net_arch_array = np.ones(net_arch_depth,dtype=int)
        net_arch = [{"pi": (net_arch_array*net_arch_width).tolist(), "vf": (net_arch_array*net_arch_width).tolist()}]

        # Display true values not shown otherwise
        trial.set_user_attr("vf_coef", vf_coef)
        trial.set_user_attr("clip_range", clip_range)
        trial.set_user_attr("max_grad_norm", max_grad_norm)
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("n_steps", n_steps)
        trial.set_user_attr("ent_coef", ent_coef)
        trial.set_user_attr("net_arch_width", net_arch_width)

        """ 
        NOTE: Rollout buffer size is number of steps*envs, should be in range of 2048-409600
        in this case we are testing only 1 enviroment at a time

        NOTE: Batch_size corresponds to how many experiences are used for each gradient descent update.
        his should always be a fraction of the buffer_size. If you are using a continuous action space,
        this value should be large (in 1000s). 
        """

        return {
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
            "clip_range": clip_range,
            "batch_size": batch_size, 
            "n_epochs": n_epochs,
            "policy_kwargs":
            {
                "net_arch": net_arch,
                "features_extractor_class": CustomCNN,
                "features_extractor_kwargs":dict(features_dim=128),
            }
        }

    def objective(self, trial: optuna.Trial) -> float:

        kwargs = self.default_hyperparams.copy()
        # Sample hyperparameters

        kwargs.update(self.sample_params(trial))
        # Create the RL model

        print("\n \n")
        print(kwargs)

        # NOTE: Single Enviroment
        # eval_env = gym.make(self.env_id, render = self.render) #, render=True)
        # # Create the callback that will periodically evaluate and report the performance

        # eval_env = PandaWrapper(eval_env)

        # # ADDED IN THE MONITOR SINCE THIS WAS THROWING AN ERROR
        # eval_env = Monitor(eval_env, log_dir)


        # NOTE: Multi Enviroment
        n_envs = trial.suggest_int("n_envs", 3, 6, log=True) 
        # trial.set_user_attr("n_envs", n_envs)
        print("n_envs: ", n_envs)
        print("\n \n")

        vec_env = make_panda_env(env_id=self.env_id, n_envs=n_envs, seed=0, monitor_dir=self.callbackDir)
        # vec_env_stack = VecFrameStack(vec_env, n_stack=n_envs)

        model = PPO(env=vec_env, **kwargs)

        eval_callback = TrialEvalCallback(
            vec_env, trial, n_eval_episodes=self.n_eval_episodes, eval_freq=self.eval_freq, deterministic=True, callbackDir = self.callbackDir
        )

        nan_encountered = False

        try:
            model.learn(total_timesteps = self.n_timesteps, callback=eval_callback)
        except AssertionError as e: # Sometimes, random hyperparams can generate NaN
            print(e) 
            nan_encountered = True
        finally:
            # Free memory
            model.env.close()
            # eval_env.close()

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        print("TRIAL Finished: Objective Trial Time")

        return eval_callback.last_mean_reward
    


