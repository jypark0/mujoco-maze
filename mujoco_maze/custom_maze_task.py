""" Custom added maze tasks """
from typing import Dict, List, Type

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell
from mujoco_maze.maze_task import MazeTask, Scaling, MazeGoal, RED, BLUE, GREEN


class GoalRewardRoom3x5(MazeTask):
    INNER_REWARD_SCALING: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = 0

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([4.0, 0.0]) * scale)]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B],
            [B, E, E, E, E, E, B],
            [B, R, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B],
        ]


class DistRewardRoom3x5(MazeTask):
    INNER_REWARD_SCALING: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = 0
    pass


class ShapedRewardRoom3x5(MazeTask):
    # INNER_REWARD_SCALING: float = 0
    INNER_REWARD_SCALING: float = 0.01
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    REWARD_THRESHOLD: float = 93
    PENALTY: float = -0.001

    def __init__(self, scale: float, n_goals: int, goal_index: int) -> None:
        super().__init__(scale)
        self.all_goals = [(i, 0.0) for i in np.linspace(4 / n_goals, 4, n_goals)]
        self.goals = [
            MazeGoal(np.array(self.all_goals[goal_index]) * scale, threshold=0.6)
        ]

    def reward(self, obs: np.ndarray) -> float:
        reward = self.PENALTY

        if self.termination(obs):
            reward = 100.0
        return reward

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B],
            [B, E, E, E, E, E, B],
            [B, R, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B],
        ]


class DistShapedRewardRoom3x5(ShapedRewardRoom3x5):
    TOP_DOWN_VIEW: bool = True
    INNER_REWARD_SCALING: float = 0
    REWARD_THRESHOLD: float = -70
    PENALTY: float = 0

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class CustomTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "Room3x5": [DistRewardRoom3x5, GoalRewardRoom3x5],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(CustomTaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return CustomTaskRegistry.REGISTRY[key]


class ExpertTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "Room3x5Expert4Goals": ShapedRewardRoom3x5,
        "Room3x5Expert8Goals": ShapedRewardRoom3x5,
        "DistRoom3x5Expert1Goals": DistShapedRewardRoom3x5,
        "DistRoom3x5Expert2Goals": DistShapedRewardRoom3x5,
        "DistRoom3x5Expert4Goals": DistShapedRewardRoom3x5,
    }
    N_GOALS = {
        "Room3x5Expert4Goals": 4,
        "Room3x5Expert8Goals": 8,
        "DistRoom3x5Expert1Goals": 1,
        "DistRoom3x5Expert2Goals": 2,
        "DistRoom3x5Expert4Goals": 4,
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
