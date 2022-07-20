# Base Imports
import gym
import os
from SB3.vectEnvs import make_panda_env

# Video Rendering: 
import base64
from pathlib import Path
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

def record_video_single(model, env_id, video_dir, wrapper, render=False, video_length=400, prefix=''):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """

    eval_env = DummyVecEnv([lambda: wrapper(gym.make(env_id, render=render))]) 
    

    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_dir,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    print("record vid sigal reset called")
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

def record_video_multiple(model, n_envs, env_id, video_dir, wrapper, render=False, video_length=400, prefix=''):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """

    monitorDir = video_dir + "monitor/"
    if not os.path.exists(monitorDir):
        os.mkdir(monitorDir)

    eval_env = make_panda_env(env_id=env_id, n_envs=n_envs, seed=0, monitor_dir=monitorDir, wrapper=wrapper)
    # eval_env = VecFrameStack(eval_env, n_stack=n_envs)
    

    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_dir,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)
