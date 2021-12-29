""" 
Custom added maze tasks with dense rewards and progressively farther goals
For creating expert demonstrations

"""
from typing import Dict, List, Tuple, Type

import numpy as np

from mujoco_maze.custom_maze_task import (
    GoalRewardLargeUMaze,
    GoalRewardRoom3x5,
    GoalRewardRoom3x10,
    DistReward,
)
from mujoco_maze.maze_task import MazeGoal, MazeTask


class DistCurriculumRoom3x5(GoalRewardRoom3x5):
    INNER_REWARD_SCALING: float = 0.01
    PENALTY: float = 0

    def __init__(self, scale: float, goal: Tuple[float, float]) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale, threshold=0.6)]

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class DistCurriculumRoom3x10(GoalRewardRoom3x10):
    INNER_REWARD_SCALING: float = 0.01
    PENALTY: float = 0

    def __init__(self, scale: float, goal: Tuple[float, float]) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale, threshold=0.6)]

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class DistCurriculumLargeUMaze(GoalRewardLargeUMaze):
    INNER_REWARD_SCALING: float = 0.01
    PENALTY: float = 0

    def __init__(self, scale: float, goal: Tuple[float, float]) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale, threshold=0.6)]

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class ExpertTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "DistRoom3x5_1Goals": DistCurriculumRoom3x5,
        "DistRoom3x10_1Goals": DistCurriculumRoom3x10,
        "DistLargeUMaze_2Goals": DistCurriculumLargeUMaze,
        "DistLargeUMaze_4Goals": DistCurriculumLargeUMaze,
    }
    GOALS = {
        "DistRoom3x5_1Goals": [(4, 0.0)],
        "DistRoom3x10_1Goals": [(9, 0.0)],
        "DistLargeUMaze_2Goals": [(2, 2), (0, 4)],
        "DistLargeUMaze_4Goals": [(2, 1), (2, 2), (2, 3), (0, 4)],
    }
    REWARD_THRESHOLDS = {
        "DistRoom3x5_1Goals": DistReward([-70], [-70], None),
        "DistRoom3x10_1Goals": DistReward([-70], [-690], None),
        "DistLargeUMaze_2Goals": DistReward([-300, -700], [-50, -100], None),
        "DistLargeUMaze_4Goals": DistReward(
            [-200, -400, -600, -800], [-25, -50, -75, -100], None
        ),
    }

    @staticmethod
    def keys() -> List[str]:
        return list(ExpertTaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return ExpertTaskRegistry.REGISTRY[key]

    @staticmethod
    def goals(key: str) -> List[Type[MazeTask]]:
        return ExpertTaskRegistry.GOALS[key]

    @staticmethod
    def reward_thresholds(key: str) -> List[Type[MazeTask]]:
        return ExpertTaskRegistry.REWARD_THRESHOLDS[key]
