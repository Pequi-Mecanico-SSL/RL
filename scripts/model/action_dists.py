import numpy as np
import gym
import torch
from math import log
import os

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import List, Union, ModelConfigDict
from torch import Tensor as TensorType
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution

from ray.rllib.utils.numpy import SMALL_NUMBER

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

class TorchBetaTest(TorchDistributionWrapper):
    """
    A Beta distribution is defined on the interval [0, 1] and parameterized by
    shape parameters alpha and beta (also called concentration parameters).
    PDF(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
        with Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
        and Gamma(n) = (n - 1)!
    """

    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
        low: float = -1.0,
        high: float = 1.0,
        signal: list = [1, 1, 1, 1],
    ):
        super().__init__(inputs, model)
        # Stabilize input parameters (possibly coming from a linear layer).
        self.inputs = torch.clamp(self.inputs, log(SMALL_NUMBER), -log(SMALL_NUMBER))
        self.inputs = torch.log(torch.exp(self.inputs) + 1.0) + 1.0
        self.low = low
        self.high = high
        alpha, beta = torch.chunk(self.inputs, 2, dim=-1)
        # Note: concentration0==beta, concentration1=alpha (!)
        self.dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
        self.signal = torch.tensor(signal)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = self._squash(self.dist.mean)
        return self.signal.to(self.device) * self.last_sample

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        # Use the reparameterization version of `dist.sample` to allow for
        # the results to be backprop'able e.g. in a loss term.
        normal_sample = self.dist.rsample()
        self.last_sample = self._squash(normal_sample)

        return self.signal.to(self.device) * self.last_sample

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        unsquashed_values = self._unsquash(x)
        return torch.sum(self.dist.log_prob(unsquashed_values), dim=-1)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        return self.dist.entropy().sum(-1)

    def _squash(self, raw_values: TensorType) -> TensorType:
        return raw_values * (self.high - self.low) + self.low

    def _unsquash(self, values: TensorType) -> TensorType:
        return (values - self.low) / (self.high - self.low)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape) * 2


class TorchBetaTest_yellow(TorchBetaTest):
    def __init__(self, inputs: List[TensorType], model: TorchModelV2):
        super().__init__(inputs, model, signal=[-1, 1, -1, 1])

class TorchBetaTest_blue(TorchBetaTest):
    def __init__(self, inputs: List[TensorType], model: TorchModelV2):
        super().__init__(inputs, model, signal=[1, 1, 1, 1])