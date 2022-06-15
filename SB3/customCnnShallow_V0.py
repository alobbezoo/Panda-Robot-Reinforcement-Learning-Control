from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gym
from collections import OrderedDict
import torchvision.models as models
import gym.spaces
from torchvision import transforms


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 1000, ):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        # # NOTE: REPLACED THIS WITH RESNET
        self.cnn = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(n_input_channels, 32, kernel_size=7, stride=2, padding=0),
            'bn1': nn.BatchNorm2d(32),
            'rl1': nn.ReLU(),
            'conv2': nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            'bn2': nn.BatchNorm2d(64),
            'rl2': nn.ReLU(),
            'conv3': nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=0),
            'bn3': nn.BatchNorm2d(128),
            'rl3': nn.ReLU(),
            'flt': nn.Flatten()
        }))


        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        # print("CNN Network Shape: ", self.cnn)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # return self.cnn(observations)
        return self.linear(self.cnn(observations))
