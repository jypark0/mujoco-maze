""" Custom added maze tasks """
from typing import Dict, List, Type

import numpy as np

from mujoco_maze.maze_env_utils import MazeCell
from mujoco_maze.task_common import (
    DistRewardMixIn,
    MazeGoal,
    MazeTask,
    RewardThreshold,
    Scaling,
    WayPointMixIn,
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
        self.goal_reward = 1000

    def reward(self, obs: np.ndarray) -> float:
        return self.goal_reward if self.termination(obs) else self.PENALTY

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


class DistRewardRoom3x5(DistRewardMixIn, GoalRewardRoom3x5):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 1000, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goal_reward = 1000


class WayPointRoom3x5(WayPointMixIn, GoalRewardRoom3x5):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 1000, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        waypoints = [(2, 0)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.goal_reward = 1000
        self.waypoint_reward = 0


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
        self.goal_reward = 1000

    def reward(self, obs: np.ndarray) -> float:
        return self.goal_reward if self.termination(obs) else self.PENALTY

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


class DistRewardRoom3x10(DistRewardMixIn, GoalRewardRoom3x10):
    # for dt0.03, gear10
    # REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-10, 170, None)
    # for dt0.02, gear30
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 170, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goal_reward = 1000


class WayPointRoom3x10(WayPointMixIn, GoalRewardRoom3x10):
    # for dt0.03, gear10
    # REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-10, 170, None)
    # for dt0.02, gear30
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 170, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        waypoints = [(2, 0), (4, 0), (6, 0), (8, 0)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.waypoint_reward = 0


class GoalRewardWallRoom5x11(MazeTask):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0.9, 0.9, 0.9)
    PENALTY: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    INNER_REWARD_SCALING: float = 0
    RENDER_WIDTH = 500
    RENDER_HEIGHT = 300
    VIEWER_SETUP_KWARGS = {"distance": 0.9, "elevation": -60, "azimuth": 90}

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        # Goal region
        # self.goals = [
        #     MazeGoal(
        #         np.array([10.25, 0.0]) * scale,
        #         region_size=(scale / 4, scale / 4),
        #     )
        # ]

        # Goal point
        self.goals = [MazeGoal(np.array([10.0, 0.0]) * scale)]
        self.goal_reward = 1000

    # def termination(self, obs: np.ndarray) -> bool:
    #     if (
    #         obs[0] >= 10.0 * self.scale
    #         and -0.25 * self.scale <= obs[1] <= 0.25 * self.scale
    #     ):
    #         return True
    #     return False

    def reward(self, obs: np.ndarray) -> float:
        return self.goal_reward if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, R, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B],
        ]


class DistRewardWallRoom5x11(DistRewardMixIn, GoalRewardWallRoom5x11):
    # for dt0.03, gear10
    # REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-280, -70, None)
    # for dt0.02, gear30
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, -70, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goal_reward = 1000


class WayPointWallRoom5x11(WayPointMixIn, GoalRewardWallRoom5x11):
    # for dt0.03, gear10
    # REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-280, -70, None)
    # for dt0.02, gear30
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, -70, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        waypoints = [(2, 0), (4, 0), (6, 0), (8, 0)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.goal_reward = 1000
        self.waypoint_reward = 0


class GoalRewardChasmRoom5x11(GoalRewardWallRoom5x11):
    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, C, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.CHASM, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, C, E, E, E, E, E, B],
            [B, E, E, E, E, E, C, E, E, E, E, E, B],
            [B, R, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, C, E, E, E, E, E, B],
            [B, E, E, E, E, E, C, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B],
        ]


class DistRewardChasmRoom5x11(DistRewardMixIn, GoalRewardChasmRoom5x11):
    # for dt0.03, gear10
    # REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-280, -70, None)
    # for dt0.02, gear30
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, -70, None)
    pass


class WayPointChasmRoom5x11(WayPointMixIn, GoalRewardChasmRoom5x11):
    # for dt0.03, gear10
    # REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-280, -70, None)
    # for dt0.02, gear30
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, -70, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        waypoints = [(2, 0), (4, 0), (6, 0), (8, 0)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.waypoint_reward = 0


class GoalRewardWallRoom7x15(MazeTask):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0.9, 0.9, 0.9)
    PENALTY: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    OBSERVE_BLOCKS: bool = True
    INNER_REWARD_SCALING: float = 0
    RENDER_WIDTH = 800
    RENDER_HEIGHT = 500
    VIEWER_SETUP_KWARGS = {"distance": 0.8, "elevation": -90, "azimuth": 90}

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [
            MazeGoal(
                np.array([14.25, 1.75]) * scale,
                region_size=(scale / 4, 1.75 * scale),
            )
        ]
        self.goal_reward = 1000

    def termination(self, obs: np.ndarray) -> bool:
        if obs[0] >= 14 * self.scale and obs[1] >= 0.0 * self.scale:
            return True
        return False

    def reward(self, obs: np.ndarray) -> float:
        return self.goal_reward if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, B],
            [B, E, E, E, E, B, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, B],
            [B, R, E, E, E, B, E, E, E, E, E, B, E, E, E, E, B],
            [B, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, B, E, E, E, E, B],
            [B, E, E, E, E, B, E, E, E, E, E, B, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        ]


class DistRewardWallRoom7x15(DistRewardMixIn, GoalRewardWallRoom7x15):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 1000, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goal_reward = 1000


class WayPointWallRoom7x15(WayPointMixIn, GoalRewardWallRoom7x15):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 1000, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        waypoints = [(4, 2), (10, -2)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.goal_reward = 1000
        self.waypoint_reward = 0


class GoalRewardLargeUMaze(MazeTask):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0.9, 0.9, 0.9)
    PENALTY: float = 0
    # MAZE_SIZE_SCALING: Scaling = Scaling(8.0, 8.0, 4.0)
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)
    INNER_REWARD_SCALING: float = 0
    RENDER_HEIGHT = 500
    RENDER_WIDTH = 500
    VIEWER_SETUP_KWARGS = {"distance": 1.2, "elevation": -60, "azimuth": 90}

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        # Goal region
        # self.goals = [
        #     MazeGoal(
        #         np.array([-0.25, 4.25]) * scale,
        #         region_size=(scale / 4, scale / 4),
        #     )
        # ]

        # Goal point
        self.goals = [MazeGoal(np.array([0.0, 6.0]) * scale)]
        self.goal_reward = 1000

    # def termination(self, obs: np.ndarray) -> bool:
    #     if obs[0] <= 0 and obs[1] >= 4.0 * self.scale:
    #         return True
    #     return False

    def reward(self, obs: np.ndarray) -> float:
        return self.goal_reward if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        # return [
        #     [B, B, B, B, B],
        #     [B, R, E, E, B],
        #     [B, E, E, E, B],
        #     [B, B, B, E, B],
        #     [B, E, E, E, B],
        #     [B, E, E, E, B],
        #     [B, B, B, B, B],
        # ]
        return [
            [B, B, B, B, B, B, B],
            [B, R, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, E, B],
            [B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B],
        ]


class DistRewardLargeUMaze(DistRewardMixIn, GoalRewardLargeUMaze):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 1000, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goal_reward = 1000


class WayPointLargeUMaze(WayPointMixIn, GoalRewardLargeUMaze):
    # for dt0.03, gear10 with smaller maze
    # REWARD_THRESHOLD: RewardThreshold = RewardThreshold(-10, 210, None)
    # for dt0.02, gear30
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 1000, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        # smaller umaze
        # waypoints = [(1.5, 1), (2, 2), (1.5, 3)]
        waypoints = [(3, 1.5), (4, 3), (3, 4.5)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.goal_reward = 1000
        self.waypoint_reward = 0


class GoalRewardCorridor7x7(MazeTask):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(0.9, 0.9, 0.9)
    PENALTY: float = 0
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=4.0, point=4.0, swimmer=1.0)
    INNER_REWARD_SCALING: float = 0
    RENDER_WIDTH: int = 500
    RENDER_HEIGHT: int = 400
    VIEWER_SETUP_KWARGS = {"distance": 1.2, "elevation": -60, "azimuth": 90}

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([6.0, 6.0]) * scale)]
        self.goal_reward = 1000

    def reward(self, obs: np.ndarray) -> float:
        return self.goal_reward if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, R, E, B, E, E, E, E, B],
            [B, E, E, B, E, E, E, E, B],
            [B, E, E, E, E, E, B, B, B],
            [B, E, E, E, E, E, E, E, B],
            [B, B, B, E, E, E, E, E, B],
            [B, E, E, E, E, B, E, E, B],
            [B, E, E, E, E, B, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]


class DistRewardCorridor7x7(DistRewardMixIn, GoalRewardCorridor7x7):
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 240, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goal_reward = 1000


class WayPointRewardCorridor7x7(WayPointMixIn, GoalRewardCorridor7x7):
    # for dt0.03, gear10
    # REWARD_THRESHOLD: RewardThreshold = RewardThreshold(125, 225, None)
    # for dt0.02, gear30
    REWARD_THRESHOLD: RewardThreshold = RewardThreshold(1000, 225, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)

        waypoints = [(3, 3)]
        self.create_waypoints(waypoints)
        self.precalculate_distances()

        self.goal_reward = 1000
        self.waypoint_reward = 0


class CustomTaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "Room3x5": [GoalRewardRoom3x5, DistRewardRoom3x5, WayPointRoom3x5],
        "Room3x10": [GoalRewardRoom3x10, DistRewardRoom3x10, WayPointRoom3x10],
        "WallRoom5x11": [
            GoalRewardWallRoom5x11,
            DistRewardWallRoom5x11,
            WayPointWallRoom5x11,
        ],
        "ChasmRoom5x11": [
            GoalRewardChasmRoom5x11,
            DistRewardChasmRoom5x11,
            WayPointChasmRoom5x11,
        ],
        "WallRoom7x15": [
            GoalRewardWallRoom7x15,
            DistRewardWallRoom7x15,
            WayPointWallRoom7x15,
        ],
        "LargeUMaze": [GoalRewardLargeUMaze, DistRewardLargeUMaze, WayPointLargeUMaze],
        "Corridor7x7": [
            GoalRewardCorridor7x7,
            DistRewardCorridor7x7,
            WayPointRewardCorridor7x7,
        ],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(CustomTaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return CustomTaskRegistry.REGISTRY[key]
