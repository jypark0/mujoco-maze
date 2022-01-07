""" Custom added maze tasks """
from typing import Dict, List, Type

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell
from mujoco_maze.task_common import (
    GREEN,
    DistRewardMixIn,
    WayPointMixIn,
    MazeGoal,
    MazeTask,
    Scaling,
    RewardThreshold,
    euc_dist,
)


class GoalRewardRoom3x5(MazeTask):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0.9, 0.9, 0.9)
    PENALTY: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    INNER_REWARD_SCALING: float = 0
    RENDER_WIDTH: int = 500
    RENDER_HEIGHT: int = 500
    VIEWER_SETUP_KWARGS = {"distance": 1.0, "elevation": -60, "azimuth": 90}

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


class DistRewardRoom3x5(GoalRewardRoom3x5, DistRewardMixIn):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0, 0, 0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goal_reward = 500


class WayPointRoom3x5(GoalRewardRoom3x5, WayPointMixIn):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0, 0, 0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        waypoints = [(2, 0)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.goal_reward = 500
        self.waypoint_reward = 100


class GoalRewardRoom3x10(MazeTask):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0.9, 0.9, 0.9)
    PENALTY: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    INNER_REWARD_SCALING: float = 0
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


class DistRewardRoom3x10(GoalRewardRoom3x10, DistRewardMixIn):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-70, -690, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goal_reward = 500


class WayPointRoom3x10(GoalRewardRoom3x10, WayPointMixIn):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-70, -690, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        waypoints = [(2, 0), (4, 0), (6, 0), (8, 0)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.goal_reward = 500
        self.waypoint_reward = 100


class GoalRewardLargeUMaze(MazeTask):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0.9, 0.9, 0.9)
    PENALTY: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(8.0, 8.0, 4.0)
    INNER_REWARD_SCALING: float = 0
    RENDER_HEIGHT = 500
    RENDER_WIDTH = 500
    VIEWER_SETUP_KWARGS = {"distance": 0.9, "elevation": -60, "azimuth": 90}

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 4.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        # return [
        #     [B, B, B, B, B, B],
        #     [B, E, E, E, E, B],
        #     [B, R, E, E, E, B],
        #     [B, E, E, E, E, B],
        #     [B, B, B, B, E, B],
        #     [B, B, B, B, E, B],
        #     [B, E, E, E, E, B],
        #     [B, E, E, E, E, B],
        #     [B, E, E, E, E, B],
        #     [B, B, B, B, B, B],
        # ]
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, E, E, E, B],
            [B, B, B, E, B],
            [B, E, E, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ]


class DistRewardLargeUMaze(GoalRewardLargeUMaze, DistRewardMixIn):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-700, None, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goal_reward = 500


class WayPointLargeUMaze(GoalRewardLargeUMaze, WayPointMixIn):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-700, None, None)
    INNER_REWARD_SCALING: float = 1

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        waypoints = [(1.5, 1), (2, 2), (1.5, 3)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.goal_reward = 500
        self.waypoint_reward = 100


class CustomTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "Room3x5": [GoalRewardRoom3x5, DistRewardRoom3x5, WayPointRoom3x5],
        "Room3x10": [GoalRewardRoom3x10, DistRewardRoom3x10, WayPointRoom3x10],
        "LargeUMaze": [GoalRewardLargeUMaze, DistRewardLargeUMaze, WayPointLargeUMaze],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(CustomTaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return CustomTaskRegistry.REGISTRY[key]
