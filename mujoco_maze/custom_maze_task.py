""" Custom added maze tasks """
from typing import Dict, List, Type

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell
from mujoco_maze.maze_task import (
    MazeTask,
    Scaling,
    MazeGoal,
    DistRewardMixIn,
    RED,
    BLUE,
    GREEN,
)


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


class GoalRewardRoom3x10(MazeTask):
    INNER_REWARD_SCALING: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = 0
    RENDER_WIDTH = 1000
    RENDER_HEIGHT = 500
    VIEWER_SETUP_KWARGS = {"distance": 0.6, "elevation": -60, "azimuth": 90}

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([9.0, 0.0]) * scale)]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, E, E, E, E, B],
            [B, R, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B],
        ]


class DistRewardRoom3x10(GoalRewardRoom3x10):
    INNER_REWARD_SCALING: float = 0.01
    REWARD_THRESHOLD: float = -80
    PENALTY: float = 0

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class GoalRewardLargeUMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    PENALTY: float = -0.0001
    RENDER_HEIGHT = 500
    RENDER_WIDTH = 500
    VIEWER_SETUP_KWARGS = {"distance": 1.1, "elevation": -60, "azimuth": 90}

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 3.5 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B],
            [B, E, E, E, E, B],
            [B, R, E, E, E, B],
            [B, E, E, E, E, B],
            [B, B, B, B, E, B],
            [B, B, B, B, E, B],
            [B, E, E, E, E, B],
            [B, E, E, E, E, B],
            [B, E, E, E, E, B],
            [B, B, B, B, B, B],
        ]


class DistRewardLargeUMaze(GoalRewardLargeUMaze, DistRewardMixIn):
    pass


class CustomTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "Room3x5": [GoalRewardRoom3x5, DistRewardRoom3x5],
        "Room3x10": [GoalRewardRoom3x10, DistRewardRoom3x10],
        "LargeUMaze": [GoalRewardLargeUMaze, DistRewardLargeUMaze],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(CustomTaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return CustomTaskRegistry.REGISTRY[key]
