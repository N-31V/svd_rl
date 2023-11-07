from abc import ABC, abstractmethod
from svdtrainer.state import State


class Reward(ABC):
    """Calculate reward by final state."""
    @abstractmethod
    def __call__(self, state: State) -> float:
        raise NotImplementedError


class MetricBoundReward(Reward):
    def __init__(self, bound: float):
        self.bound = bound

    def __call__(self, state: State) -> float:
        if state.f1 < self.bound:
            return 0
        else:
            return 1 - state.size


class SizeBoundReward(Reward):
    def __init__(self, bound: float):
        self.bound = bound

    def __call__(self, state: State) -> float:
        if state.size > self.bound:
            return 0
        else:
            return state.f1


class MetricSizeReward(Reward):
    def __init__(self, size_factor: float):
        self.size_factor = size_factor

    def __call__(self, state: State) -> float:
        return state.f1 + self.size_factor * (1 - state.size)
