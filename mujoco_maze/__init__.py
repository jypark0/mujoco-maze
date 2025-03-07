"""
Mujoco Maze
----------

A maze environment using mujoco that supports custom tasks and robots.
"""


from gym.envs import register

from mujoco_maze.ant import AntEnv
from mujoco_maze.custom_maze_task import CustomTaskRegistry
from mujoco_maze.maze_task import TaskRegistry
from mujoco_maze.point import PointEnv
from mujoco_maze.point_fixed_start import PointFixedStartEnv
from mujoco_maze.reacher import ReacherEnv
from mujoco_maze.swimmer import SwimmerEnv

max_episode_steps = 400


def orig_register(task_registry):
    for maze_id in task_registry.keys():
        for i, task_cls in enumerate(task_registry.tasks(maze_id)):
            point_scale = task_cls.MAZE_SIZE_SCALING.point
            if point_scale is not None:
                # Point
                register(
                    id=f"Point{maze_id}-v{i}",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=PointEnv,
                        maze_task=task_cls,
                        maze_size_scaling=point_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=max_episode_steps,
                    reward_threshold=task_cls.REWARD_THRESHOLD,
                )

            ant_scale = task_cls.MAZE_SIZE_SCALING.ant
            if ant_scale is not None:
                # Ant
                register(
                    id=f"Ant{maze_id}-v{i}",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=AntEnv,
                        maze_task=task_cls,
                        maze_size_scaling=ant_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=max_episode_steps,
                    reward_threshold=task_cls.REWARD_THRESHOLD,
                )

            swimmer_scale = task_cls.MAZE_SIZE_SCALING.swimmer
            if swimmer_scale is not None:
                # Reacher
                register(
                    id=f"Reacher{maze_id}-v{i}",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=ReacherEnv,
                        maze_task=task_cls,
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=max_episode_steps,
                    reward_threshold=task_cls.REWARD_THRESHOLD,
                )
                # Swimmer
                register(
                    id=f"Swimmer{maze_id}-v{i}",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=SwimmerEnv,
                        maze_task=task_cls,
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=max_episode_steps,
                    reward_threshold=task_cls.REWARD_THRESHOLD,
                )


def custom_register(
    task_registry, entry_point="mujoco_maze.maze_env:MazeEnv", prefix=""
):
    for maze_id in task_registry.keys():
        for i, task_cls in enumerate(task_registry.tasks(maze_id)):
            point_scale = task_cls.MAZE_SIZE_SCALING.point
            point_reward_threshold = task_cls.REWARD_THRESHOLD.point
            if point_scale is not None and point_reward_threshold is not None:
                # Point
                register(
                    id=f"{prefix}Point{maze_id}-v{i}",
                    entry_point=entry_point,
                    kwargs=dict(
                        model_cls=PointEnv,
                        maze_task=task_cls,
                        maze_size_scaling=point_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=max_episode_steps,
                    reward_threshold=point_reward_threshold,
                )

            ant_scale = task_cls.MAZE_SIZE_SCALING.ant
            ant_reward_threshold = task_cls.REWARD_THRESHOLD.ant
            if ant_scale is not None and ant_reward_threshold is not None:
                # Ant
                register(
                    id=f"{prefix}Ant{maze_id}-v{i}",
                    entry_point=entry_point,
                    kwargs=dict(
                        model_cls=AntEnv,
                        maze_task=task_cls,
                        maze_size_scaling=ant_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=max_episode_steps,
                    reward_threshold=ant_reward_threshold,
                )

            swimmer_scale = task_cls.MAZE_SIZE_SCALING.swimmer
            swimmer_reward_threshold = task_cls.REWARD_THRESHOLD.swimmer
            if swimmer_scale is not None and swimmer_reward_threshold is not None:
                # Reacher
                register(
                    id=f"{prefix}Reacher{maze_id}-v{i}",
                    entry_point=entry_point,
                    kwargs=dict(
                        model_cls=ReacherEnv,
                        maze_task=task_cls,
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=max_episode_steps,
                    reward_threshold=swimmer_reward_threshold,
                )
                # Swimmer
                register(
                    id=f"{prefix}Swimmer{maze_id}-v{i}",
                    entry_point=entry_point,
                    kwargs=dict(
                        model_cls=SwimmerEnv,
                        maze_task=task_cls,
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=max_episode_steps,
                    reward_threshold=swimmer_reward_threshold,
                )


orig_register(TaskRegistry)
custom_register(CustomTaskRegistry, "mujoco_maze.maze_env:MazeEnv", "")
custom_register(CustomTaskRegistry, "mujoco_maze.goal_maze_env:GoalMazeEnv", "Goal")

# Register FixedStart envs separately
for maze_id in ["Room3x10", "WallRoom5x11", "ChasmRoom5x11"]:
    for i, task_cls in enumerate(CustomTaskRegistry.tasks(maze_id)):
        point_scale = task_cls.MAZE_SIZE_SCALING.point
        point_reward_threshold = task_cls.REWARD_THRESHOLD.point
        if point_scale is not None and point_reward_threshold is not None:
            # Point
            register(
                id=f"Point{maze_id}-FixedStart-v{i}",
                entry_point="mujoco_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=PointFixedStartEnv,
                    maze_task=task_cls,
                    maze_size_scaling=point_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=max_episode_steps,
                reward_threshold=point_reward_threshold,
            )
            register(
                id=f"GoalPoint{maze_id}-FixedStart-v{i}",
                entry_point="mujoco_maze.goal_maze_env:GoalMazeEnv",
                kwargs=dict(
                    model_cls=PointFixedStartEnv,
                    maze_task=task_cls,
                    maze_size_scaling=point_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=max_episode_steps,
                reward_threshold=point_reward_threshold,
            )

__version__ = "0.2.0"
