""" Custom added maze tasks """
from typing import Dict, List, Type

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell
from mujoco_maze.maze_task import MazeTask, Scaling, MazeGoal, RED, BLUE, GREEN

from mujoco_maze.custom_maze_task import (
    GoalRewardRoom3x5,
    GoalRewardRoom3x10,
    GoalRewardLargeUMaze,
)


class DistCurriculumRoom3x5(GoalRewardRoom3x5):
    INNER_REWARD_SCALING: float = 0.01
    REWARD_THRESHOLD: float = -70
    PENALTY: float = 0

    def __init__(self, scale: float, n_goals: int, goal_index: int) -> None:
        super().__init__(scale)
        self.all_goals = [(i, 0.0) for i in np.linspace(4 / n_goals, 4, n_goals)]
        self.goals = [
            MazeGoal(np.array(self.all_goals[goal_index]) * scale, threshold=0.6)
        ]

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class DistCurriculumRoom3x10(GoalRewardRoom3x10):
    INNER_REWARD_SCALING: float = 0.01
    # Point = -70, Ant = -690
    # REWARD_THRESHOLD: float = -70
    REWARD_THRESHOLD: float = -690
    PENALTY: float = 0

    def __init__(self, scale: float, n_goals: int, goal_index: int) -> None:
        super().__init__(scale)
        self.all_goals = [(i, 0.0) for i in np.linspace(4 / n_goals, 4, n_goals)]
        self.goals = [
            MazeGoal(np.array(self.all_goals[goal_index]) * scale, threshold=0.6)
        ]

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class DistCurriculumLargeUMaze(GoalRewardLargeUMaze):
    INNER_REWARD_SCALING: float = 0.01
    REWARD_THRESHOLD: float = -72
    PENALTY: float = 0

    def __init__(self, scale: float, n_goals: int, goal_index: int) -> None:
        super().__init__(scale)
        self.all_goals = [
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (2, 4),
            (1, 4),
            (0, 5),
        ]
        self.goals = [
            MazeGoal(np.array(self.all_goals[goal_index]) * scale, threshold=0.6)
        ]

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class ExpertTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "DistRoom3x5_1Goals": DistCurriculumRoom3x5,
        "DistRoom3x10_1Goals": DistCurriculumRoom3x10,
        "DistLargeUMaze_8Goals": DistCurriculumLargeUMaze,
    }
    N_GOALS = {
        "DistRoom3x5_1Goals": 1,
        "DistRoom3x10_1Goals": 1,
        "DistLargeUMaze_8Goals": 8,
    }

    @staticmethod
    def keys() -> List[str]:
        return list(ExpertTaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return ExpertTaskRegistry.REGISTRY[key]

    @staticmethod
    def n_goals(key: str) -> List[Type[MazeTask]]:
        return ExpertTaskRegistry.N_GOALS[key]
