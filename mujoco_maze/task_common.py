from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Sequence

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell


def euc_dist(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)


class MazeGoal:
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 0.6,
        custom_size: Optional[float] = None,
        region_size=None,
    ) -> None:
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = rgb
        self.threshold = threshold
        self.custom_size = custom_size
        self.region_size = region_size

    def neighbor(self, obs: np.ndarray) -> float:
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5


class Scaling(NamedTuple):
    ant: Optional[float]
    point: Optional[float]
    swimmer: Optional[float]


class RewardThreshold(NamedTuple):
    ant: Optional[float]
    point: Optional[float]
    swimmer: Optional[float]


class RewardThresholdList(NamedTuple):
    ant: Optional[Sequence[float]]
    point: Optional[Sequence[float]]
    swimmer: Optional[Sequence[float]]


class MazeTask(ABC):
    REWARD_THRESHOLD: float
    PENALTY: Optional[float] = None
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=8.0, point=4.0, swimmer=4.0)
    INNER_REWARD_SCALING: float = 0.01
    # For Fall/Push/BlockMaze
    OBSERVE_BLOCKS: bool = False
    # For Billiard
    OBSERVE_BALLS: bool = False
    OBJECT_BALL_SIZE: float = 1.0
    # Unused now
    PUT_SPIN_NEAR_AGENT: bool = False
    TOP_DOWN_VIEW: bool = False
    # For render
    RENDER_WIDTH: int = 500
    RENDER_HEIGHT: int = 500
    VIEWER_SETUP_KWARGS = {"distance": 0.6, "elevation": -60, "azimuth": 90}

    def __init__(self, scale: float) -> None:
        self.goals = []
        self.scale = scale
        self.goal_reward = 1.0

    def sample_goals(self) -> bool:
        return False

    def termination(self, obs: np.ndarray) -> bool:
        for goal in self.goals:
            if goal.neighbor(obs):
                return True
        return False

    @abstractmethod
    def reward(self, obs: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass


class DistRewardMixIn:
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0, 0, 0)
    INNER_REWARD_SCALING: float = 0.1  # or 1
    PENALTY: float = 0
    goals: List[MazeGoal]
    scale: float
    goal_reward: float

    def reward(self, obs: np.ndarray) -> float:
        reward = -self.goals[0].euc_dist(obs) / self.scale
        if self.termination(obs):
            reward = self.goal_reward
        return reward


class WayPointMixIn:
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0, 0, 0)
    INNER_REWARD_SCALING: float = 0.1  # or 1
    PENALTY: float = 0

    def create_waypoints(self, waypoints=[]):
        self.waypoints = []
        for waypoint in waypoints:
            self.waypoints.append(
                MazeGoal(
                    np.array(waypoint) * self.scale,
                    rgb=GREEN,
                    custom_size=0.2 * self.scale / 2,
                )
            )

        self.waypoint_idx = 0
        if len(waypoints) >= 1:
            self.current_goal = self.waypoints[self.waypoint_idx]
        else:
            self.current_goal = None

    def precalculate_distances(self):
        # Precalculate distances b/w waypoints

        self.rews = np.zeros(len(self.waypoints) + 1)
        self.rews[0] = -euc_dist(self.waypoints[0].pos, [0, 0])
        for i in range(1, len(self.waypoints)):
            self.rews[i] = -euc_dist(self.waypoints[i - 1].pos, self.waypoints[i].pos)
        self.rews[-1] = -euc_dist(self.waypoints[-1].pos, self.goals[0].pos)
        self.rews /= self.scale

    def termination(self, obs: np.ndarray) -> bool:
        if self.waypoint_idx == len(self.waypoints) and self.goals[0].neighbor(obs):
            return True
        return False

    def update_current_goal(self, obs: np.ndarray) -> None:
        """Used to update desired_goal before calculating reward in goalEnv

        For both GoalEnv and MazeEnv, call this function before calling reward()
        """
        if self.waypoint_idx != len(self.waypoints):
            if self.current_goal.neighbor(obs):
                self.waypoint_idx += 1
                if self.waypoint_idx == len(self.waypoints):
                    self.current_goal = self.goals[0]
                else:
                    self.current_goal = self.waypoints[self.waypoint_idx]

    def reward(self, obs: np.ndarray) -> float:
        reward = np.sum(self.rews[self.waypoint_idx + 1 :])
        if self.current_goal.neighbor(obs):
            if self.waypoint_idx == len(self.waypoints):
                reward += self.goal_reward
            else:
                reward += self.waypoint_reward
        else:
            reward += -self.current_goal.euc_dist(obs) / self.scale

        return reward
