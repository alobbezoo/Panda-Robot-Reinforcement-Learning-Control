# from wrappers.pandaWrapperBW_D import PandaWrapper
from typing import Any, Callable, Dict, Optional, Type, Union
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from wrappers.pandaWrapperBW_D import PandaWrapper


class PandaTopWrapper(gym.Wrapper):
    """
    Panda preprocessings
    Specifically:
    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    :param env: gym environment
    :param noop_max: max number of no-ops
    :param frame_skip: the frequency at which the agent experiences the game.
    :param screen_size: resize Atari frame
    :param terminal_on_life_loss: if True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    """

    def __init__(
        self,
        env: gym.Env,
        base_wrapper=PandaWrapper,
        # noop_max: int = 30,
        frame_skip: int = 4,
        # screen_size: int = 84,
        # terminal_on_life_loss: bool = True,
        # clip_reward: bool = True,
    ):
        env = base_wrapper(env)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        # if terminal_on_life_loss:
        #     env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        # env = WarpFrame(env, width=screen_size, height=screen_size)
        # if clip_reward:
        #     env = ClipRewardEnv(env)

        super().__init__(env)

def make_panda_env(
    env_id: Union[str, Type[gym.Env]],
    wrapper = PandaTopWrapper, 
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.
    :param env_id: the environment ID or the environment class
        n_envs = trial.suggest_int("n_envs", 3, 6, log=True) 
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def panda_wrapper(env: gym.Env) -> gym.Env:
        env = wrapper(env, **wrapper_kwargs)
        return env
    

    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=panda_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
    )


