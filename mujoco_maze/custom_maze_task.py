""" Custom added maze tasks """
from typing import Dict, List, NamedTuple, Optional, Sequence, Type, Tuple

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell
from mujoco_maze.maze_task import (
    GREEN,
    DistRewardMixIn,
    MazeGoal,
    MazeTask,
    Scaling,
)


def euc_dist(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


class DistReward(NamedTuple):
    ant: Optional[Sequence[float]]
    point: Optional[Sequence[float]]
    swimmer: Optional[Sequence[float]]


class GoalRewardRoom3x5(MazeTask):
    INNER_REWARD_SCALING: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = 0
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
    MAZE_SIZE_SCALING: Scaling = Scaling(8.0, 8.0, 4.0)
    PENALTY: float = -0.0001
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


class DistLargeUMazeWayPoint(GoalRewardLargeUMaze):
    def __init__(self, scale: float, goal: Tuple[float, float], waypoints=None) -> None:
        super().__init__(scale, goal, waypoints)
        self.goals = [MazeGoal(np.array([0, 4]) * scale, threshold=0.6)]
        self.waypoints = []
        for waypoint in [(2, 2)]:
            self.waypoints.append(
                MazeGoal(
                    np.array(waypoint) * scale,
                    rgb=GREEN,
                    custom_size=0.1 * scale / 2,
                )
            )
        self.visited = np.zeros(len(self.waypoints), dtype=bool)

        # Precalculate distances b/w waypoints
        self.rews = np.zeros(len(self.waypoints) + 1)
        self.rews[0] = -euc_dist(self.waypoints[0].pos, [0, 0])
        for i in range(1, len(self.waypoints)):
            self.rews[i] = -euc_dist(self.waypoints[i - 1].pos, self.waypoints[i].pos)
        self.rews[-1] = -euc_dist(self.waypoints[-1].pos, self.goals[0].pos)
        self.rews /= self.scale

    def reward(self, obs: np.ndarray) -> float:
        # If all subgoals were visited
        if self.visited.all():
            reward = -self.goals[0].euc_dist(obs) / self.scale
            if self.termination(obs):
                reward = 100
        else:
            # Choose next subgoal
            goal_idx = np.argmax(~self.visited)
            # Add all remaining distances
            reward = np.sum(self.rews[goal_idx + 1 :])

            if self.waypoints[goal_idx].neighbor(obs):
                self.visited[goal_idx] = True
                reward += 100
            else:
                reward += -self.waypoints[goal_idx].euc_dist(obs) / self.scale
        return reward


class DistRewardLargeUMaze(GoalRewardLargeUMaze, DistRewardMixIn):
    pass


class CustomTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "Room3x5": [GoalRewardRoom3x5, DistRewardRoom3x5],
        "Room3x10": [GoalRewardRoom3x10, DistRewardRoom3x10],
        "LargeUMaze": [GoalRewardLargeUMaze, DistRewardLargeUMaze],
        "LargeUMazeWayPoint": [DistLargeUMazeWayPoint],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(CustomTaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return CustomTaskRegistry.REGISTRY[key]
