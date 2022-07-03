

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import results_plotter
import optuna
import gym

import numpy as np
import os

from datetime import datetime
date = datetime.now().date() 


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = results_plotter.ts2xy(results_plotter.load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    
                    self.best_mean_reward = mean_reward

                    # Added new callback for tracking the model along its training progress
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.model_path = self.log_dir + "model_" + str(timestamp) + "_numTimesteps_"+ str(self.num_timesteps)
                    print("\n Most recently saved model: ", self.model_path, "\n")

                    self.model.save(path=self.model_path)


                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 500, # EVALUATE MORE FREQUENTLY
        deterministic: bool = True,
        verbose: int = 2, # TRACKING THE RESULTS
        callbackDir: str = "temp/",
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        

        # New Entry
        self.callbackDir = callbackDir
        self.save_path = os.path.join(callbackDir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1

            # Retrieve training reward
            x, y = results_plotter.ts2xy(results_plotter.load_results(self.callbackDir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])

            # self.trial.report(self.last_mean_reward, self.eval_idx)
            
            self.trial.report(mean_reward, self.eval_idx)

            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True