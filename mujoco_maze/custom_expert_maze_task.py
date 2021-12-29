""" 
Custom added maze tasks with dense rewards and progressively farther goals
For creating expert demonstrations

"""
from typing import Dict, List, NamedTuple, Optional, Sequence, Type, Tuple

import numpy as np

from mujoco_maze.custom_maze_task import (
    GoalRewardLargeUMaze,
    GoalRewardRoom3x5,
    GoalRewardRoom3x10,
    DistReward,
    euc_dist,
)
from mujoco_maze.maze_task import MazeGoal, MazeTask, GREEN


class DistRewardList(NamedTuple):
    ant: Optional[Sequence[float]]
    point: Optional[Sequence[float]]
    swimmer: Optional[Sequence[float]]


class Room3x5(GoalRewardRoom3x5):
    INNER_REWARD_SCALING: float = 0.01
    PENALTY: float = 0

    def __init__(self, scale: float, goal: Tuple[float, float], waypoints=None) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale, threshold=0.6)]

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class Room3x5WayPoint(Room3x5):
    def __init__(self, scale: float, goal: Tuple[float, float], waypoints=None) -> None:
        super().__init__(scale, goal, waypoints)
        self.goals = [MazeGoal(np.array(goal) * scale, threshold=0.6)]
        self.waypoints = []
        for waypoint in waypoints:
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
        self.rews[0] = -euc_dist(self.waypoints[0].pos, [0, 0]) / self.scale
        for i in range(1, len(self.waypoints)):
            self.rews[i] = (
                -euc_dist(self.waypoints[i - 1].pos, self.waypoints[i].pos) / self.scale
            )
        self.rews[-1] = (
            -euc_dist(self.waypoints[-1].pos, self.goals[0].pos) / self.scale
        )

    def reward(self, obs: np.ndarray) -> float:
        # If all waypoints were visited
        if self.visited.all():
            reward = -self.goals[0].euc_dist(obs) / self.scale
            if self.termination(obs):
                reward = 100
        else:
            # Choose next waypoint
            goal_idx = np.argmax(~self.visited)
            # Add all remaining distances
            reward = np.sum(self.rews[goal_idx + 1 :])

            if self.waypoints[goal_idx].neighbor(obs):
                self.visited[goal_idx] = True
                reward += 100
            else:
                reward += -self.waypoints[goal_idx].euc_dist(obs) / self.scale
        return reward


class Room3x10(GoalRewardRoom3x10):
    INNER_REWARD_SCALING: float = 0.01
    PENALTY: float = 0

    def __init__(self, scale: float, goal: Tuple[float, float], waypoints=None) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale, threshold=0.6)]

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class LargeUMaze(GoalRewardLargeUMaze):
    INNER_REWARD_SCALING: float = 0.01
    PENALTY: float = 0

    def __init__(self, scale: float, goal: Tuple[float, float], waypoints=None) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale, threshold=0.6)]

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = 100.0
        return reward


class ExpertTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "DistRoom3x5_1Goals": Room3x5,
        "DistRoom3x5WayPoint_3Goals": Room3x5WayPoint,
        "DistRoom3x10_1Goals": Room3x10,
        "DistLargeUMaze_2Goals": LargeUMaze,
        "DistLargeUMaze_4Goals": LargeUMaze,
    }
    GOALS = {
        "DistRoom3x5_1Goals": [(4, 0)],
        "DistRoom3x5WayPoint_3Goals": [(1, 0), (2, 0), (4, 0)],
        "DistRoom3x10_1Goals": [(9, 0)],
        "DistLargeUMaze_2Goals": [(2, 2), (0, 4)],
        "DistLargeUMaze_4Goals": [(2, 1), (2, 2), (2, 3), (0, 4)],
    }
    REWARD_THRESHOLDS = {
        "DistRoom3x5_1Goals": DistReward([-70], [-70], None),
        "DistRoom3x5WayPoint_3Goals": DistReward([-20, -40, 70], [-20, -40, -70], None),
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
