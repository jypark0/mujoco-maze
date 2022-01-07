"""
Mujoco Maze
----------

A maze environment using mujoco that supports custom tasks and robots.
"""


import gym

from mujoco_maze.ant import AntEnv
from mujoco_maze.maze_task import TaskRegistry
from mujoco_maze.custom_maze_task import CustomTaskRegistry
from mujoco_maze.custom_expert_maze_task import ExpertTaskRegistry
from mujoco_maze.point import PointEnv
from mujoco_maze.reacher import ReacherEnv
from mujoco_maze.swimmer import SwimmerEnv


def register(task_registry):
    for maze_id in task_registry.keys():
        for i, task_cls in enumerate(task_registry.tasks(maze_id)):
            point_scale = task_cls.MAZE_SIZE_SCALING.point
            if point_scale is not None:
                # Point
                gym.envs.register(
                    id=f"Point{maze_id}-v{i}",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=PointEnv,
                        maze_task=task_cls,
                        maze_size_scaling=point_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=task_cls.REWARD_THRESHOLD,
                )

            ant_scale = task_cls.MAZE_SIZE_SCALING.ant
            if ant_scale is not None:
                # Ant
                gym.envs.register(
                    id=f"Ant{maze_id}-v{i}",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=AntEnv,
                        maze_task=task_cls,
                        maze_size_scaling=ant_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=task_cls.REWARD_THRESHOLD,
                )

            swimmer_scale = task_cls.MAZE_SIZE_SCALING.swimmer
            if swimmer_scale is not None:
                # Reacher
                gym.envs.register(
                    id=f"Reacher{maze_id}-v{i}",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=ReacherEnv,
                        maze_task=task_cls,
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=task_cls.REWARD_THRESHOLD,
                )
                # Swimmer
                gym.envs.register(
                    id=f"Swimmer{maze_id}-v{i}",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=SwimmerEnv,
                        maze_task=task_cls,
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
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
                gym.envs.register(
                    id=f"{prefix}Point{maze_id}-v{i}",
                    entry_point=entry_point,
                    kwargs=dict(
                        model_cls=PointEnv,
                        maze_task=task_cls,
                        maze_size_scaling=point_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=point_reward_threshold,
                )

            ant_scale = task_cls.MAZE_SIZE_SCALING.ant
            ant_reward_threshold = task_cls.REWARD_THRESHOLD.ant
            if ant_scale is not None and ant_reward_threshold is not None:
                # Ant
                gym.envs.register(
                    id=f"{prefix}Ant{maze_id}-v{i}",
                    entry_point=entry_point,
                    kwargs=dict(
                        model_cls=AntEnv,
                        maze_task=task_cls,
                        maze_size_scaling=ant_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=ant_reward_threshold,
                )

            swimmer_scale = task_cls.MAZE_SIZE_SCALING.swimmer
            swimmer_reward_threshold = task_cls.REWARD_THRESHOLD.swimmer
            if swimmer_scale is not None and swimmer_reward_threshold is not None:
                # Reacher
                gym.envs.register(
                    id=f"{prefix}Reacher{maze_id}-v{i}",
                    entry_point=entry_point,
                    kwargs=dict(
                        model_cls=ReacherEnv,
                        maze_task=task_cls,
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=swimmer_reward_threshold,
                )
                # Swimmer
                gym.envs.register(
                    id=f"{prefix}Swimmer{maze_id}-v{i}",
                    entry_point=entry_point,
                    kwargs=dict(
                        model_cls=SwimmerEnv,
                        maze_task=task_cls,
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=swimmer_reward_threshold,
                )


def expert_register(expert_task_registry):
    for maze_id in expert_task_registry.keys():
        task_cls = expert_task_registry.tasks(maze_id)
        goals = expert_task_registry.goals(maze_id)
        reward_thresholds = expert_task_registry.reward_thresholds(maze_id)

        point_scale = task_cls.MAZE_SIZE_SCALING.point
        ant_scale = task_cls.MAZE_SIZE_SCALING.ant
        swimmer_scale = task_cls.MAZE_SIZE_SCALING.swimmer

        for i, goal in enumerate(goals):
            if point_scale is not None and reward_thresholds.point:
                # Point
                gym.envs.register(
                    id=f"Point{maze_id}_{i}-v0",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=PointEnv,
                        maze_task=task_cls,
                        task_kwargs={"goal": goal, "waypoints": goals[:i]},
                        maze_size_scaling=point_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=reward_thresholds.point[i],
                )

            if ant_scale is not None and reward_thresholds.ant:
                # Ant
                gym.envs.register(
                    id=f"Ant{maze_id}_{i}-v0",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=AntEnv,
                        maze_task=task_cls,
                        task_kwargs={"goal": goal, "waypoints": goals[:i]},
                        maze_size_scaling=ant_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=reward_thresholds.ant[i],
                )

            if swimmer_scale is not None and reward_thresholds.swimmer:
                # Reacher
                gym.envs.register(
                    id=f"Reacher{maze_id}_{i}-v0",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=ReacherEnv,
                        maze_task=task_cls,
                        task_kwargs={"goal": goal, "waypoints": goals[:i]},
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=reward_thresholds.swimmer[i],
                )
                # Swimmer
                gym.envs.register(
                    id=f"Swimmer{maze_id}_{i}-v0",
                    entry_point="mujoco_maze.maze_env:MazeEnv",
                    kwargs=dict(
                        model_cls=SwimmerEnv,
                        maze_task=task_cls,
                        task_kwargs={"goal": goal, "waypoints": goals[:i]},
                        maze_size_scaling=swimmer_scale,
                        inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                    ),
                    max_episode_steps=1000,
                    reward_threshold=reward_thresholds.swimmer[i],
                )


register(TaskRegistry)
custom_register(CustomTaskRegistry, "mujoco_maze.maze_env:MazeEnv", "")
custom_register(CustomTaskRegistry, "mujoco_maze.goal_maze_env:GoalMazeEnv", "Goal")
expert_register(ExpertTaskRegistry)

__version__ = "0.2.0"
