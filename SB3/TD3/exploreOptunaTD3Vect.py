
# Custom Imports
from SB3.customCnnShallow_V0 import CustomCNN
from SB3.callbacks import TrialEvalCallback
from SB3.vectEnvs import make_panda_env

# Standard Imports
import optuna

from stable_baselines3 import TD3

# Custom Callback:
from stable_baselines3.common.vec_env import VecEnv


# from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
# from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise, OrnsteinUhlenbeckActionNoise

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
        """Sampler for hyperparameters."""

        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        tau = trial.suggest_float("tau", 0.0005, 0.1, log=True)
        learning_rate = trial.suggest_float("lr", 1e-5, 4e-3, log=True)
        
        batch_size = 2**trial.suggest_int("batch_size_num", 6, 12, log=True)
        learning_starts = 1000*trial.suggest_int("learning_starts", 1, 10, log=True) 
        buffer_size = 50000 #5000*trial.suggest_int("learning_starts", 1, 10, log=True) 
        train_freq = 50 #10*trial.suggest_int("train_freq", 1, 20, log=True)
        gradient_steps = 50

        action_noise = 0.05*trial.suggest_int("action_noise_int", 1, 3, log=True) - 0.05 #minium is zero noise
        self.action_noise = action_noise

        net_arch_width_int = trial.suggest_int("net_arch_width_int", 5, 7)
        net_arch_width = 2 ** net_arch_width_int
        net_arch_depth = trial.suggest_int("net_arch_depth", 3, 7)
        net_arch_array = np.ones(net_arch_depth,dtype=int)
        # print("here: \n: ", {(net_arch_array*net_arch_width).tolist()})
        net_arch = (net_arch_array*net_arch_width).tolist()

        # Display true values not shown otherwise
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("learning_starts", learning_starts)
        trial.set_user_attr("buffer_size", buffer_size)
        trial.set_user_attr("train_freq", train_freq)
        trial.set_user_attr("gradient_steps", gradient_steps)
        trial.set_user_attr("net_arch_width", net_arch_width)
        trial.set_user_attr("net_arch_depth", net_arch_depth)
        trial.set_user_attr("action_noise", action_noise)


        return {
            "gamma": gamma,
            "tau": tau,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "buffer_size": buffer_size,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps, 
            "policy_kwargs":
            {
                "net_arch": net_arch,
                # "features_extractor_class": CustomCNN,
                # "features_extractor_kwargs":dict(features_dim=128),
            }
        }

    def objective(self, trial: optuna.Trial) -> float:

        kwargs = self.default_hyperparams.copy()
        # Sample hyperparameters

        kwargs.update(self.sample_params(trial))
        # Create the RL model

        print("\n \n")
        print(kwargs)


        # NOTE: Multi Enviroment
        n_envs = 4 #trial.suggest_int("n_envs", 2, 6, log=True) 
        print("n_envs: ", n_envs)
        print("action_noise: ", self.action_noise)
        print("\n \n")

        vec_env = make_panda_env(env_id=self.env_id, n_envs=n_envs, seed=0, monitor_dir=self.callbackDir)
        vec_env_stack = VecFrameStack(vec_env, n_stack=n_envs)
        
        callback = TrialEvalCallback(
            vec_env, trial, n_eval_episodes=self.n_eval_episodes, eval_freq=self.eval_freq, deterministic=True, 
            callbackDir = self.callbackDir
        )
        
        # Add some action noise for exploration
        n_actions = vec_env.action_space.shape[-1]
        base_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=self.action_noise * np.ones(n_actions))
        action_noise = VectorizedActionNoise(base_noise=base_noise,  n_envs=n_envs)

        model = TD3(env=vec_env, action_noise=action_noise, **kwargs)

        nan_encountered = False

        try:
            model.learn(total_timesteps = self.n_timesteps, callback=callback)
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

        if callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        print("TRIAL Finished: Objective Trial Time")

        return callback.last_mean_reward
    


