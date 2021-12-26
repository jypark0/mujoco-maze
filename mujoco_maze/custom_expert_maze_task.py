""" Custom added maze tasks """
from typing import Dict, List, Type

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell
from mujoco_maze.maze_task import MazeTask, Scaling, MazeGoal, RED, BLUE, GREEN

from mujoco_maze.custom_maze_task import GoalRewardRoom3x5, GoalRewardRoom3x10


class DistCurriculumRoom3x5(GoalRewardRoom3x5):
    INNER_REWARD_SCALING: float = 0.01
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
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
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    REWARD_THRESHOLD: float = -72
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


class ExpertTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "DistRoom3x5Expert1Goals": DistCurriculumRoom3x5,
        "DistRoom3x10Expert1Goals": DistCurriculumRoom3x10,
    }
    N_GOALS = {
        "DistRoom3x5Expert1Goals": 1,
        "DistRoom3x10Expert1Goals": 1,
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
