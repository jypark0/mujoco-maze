import itertools as it
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Tuple, Type
from collections import OrderedDict

import gym
from gym.spaces import Dict, Box
import numpy as np
from mujoco_maze import maze_env_utils, maze_task
from mujoco_maze.agent_model import AgentModel
from mujoco_maze.maze_env import MazeEnv


class GoalMazeEnv(MazeEnv):
    """Make MazeEnv like GoalEnv, but with only positions for goals"""

    def _get_obs_space(self) -> gym.spaces.Dict:
        shape = self._get_obs()["observation"].shape
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        # Set velocity limits
        wrapped_obs_space = self.wrapped_env.observation_space
        high[: wrapped_obs_space.shape[0]] = wrapped_obs_space.high
        low[: wrapped_obs_space.shape[0]] = wrapped_obs_space.low
        # Set coordinate limits
        low[0], high[0], low[1], high[1] = self._xy_limits()
        # Set orientation limits

        observation_space = Box(low, high, shape=shape, dtype="float32")
        goal_space = Box(low[:2], high[:2], shape=(2,), dtype="float32")

        return Dict(
            OrderedDict(
                [
                    ("observation", observation_space),
                    ("desired_goal", goal_space),
                    ("achieved_goal", goal_space),
                ]
            )
        )

    def _get_obs(self) -> np.ndarray:
        observation = super()._get_obs()
        achieved_goal = self.wrapped_env.get_xy()

        if hasattr(self._task, "current_goal"):
            desired_goal = self._task.current_goal.pos
        else:
            # Just use first goal in self._task.goals
            desired_goal = self._task.goals[0].pos

        return OrderedDict(
            [
                ("observation", observation),
                ("desired_goal", desired_goal),
                ("achieved_goal", achieved_goal),
            ]
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.t += 1
        if self.wrapped_env.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            old_objballs = self._objball_positions()
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            new_objballs = self._objball_positions()
            # Checks that the new_position is in the wall
            collision = self._collision.detect(old_pos, new_pos)
            if collision is not None:
                pos = collision.point + self._restitution_coef * collision.rest()
                if self._collision.detect(old_pos, pos) is not None:
                    # If pos is also not in the wall, we give up computing the position
                    self.wrapped_env.set_xy(old_pos)
                else:
                    self.wrapped_env.set_xy(pos)
            # Do the same check for object balls
            for name, old, new in zip(self.object_balls, old_objballs, new_objballs):
                collision = self._objball_collision.detect(old, new)
                if collision is not None:
                    pos = collision.point + self._restitution_coef * collision.rest()
                    if self._objball_collision.detect(old, pos) is not None:
                        pos = old
                    idx = self.wrapped_env.model.body_name2id(name)
                    self.wrapped_env.data.xipos[idx][:2] = pos
        else:
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
        next_obs = self._get_obs()
        inner_reward = self._inner_reward_scaling * inner_reward
        outer_reward = self._task.reward(next_obs["observation"])
        done = self._task.termination(next_obs["observation"])
        info["position"] = self.wrapped_env.get_xy()
        return next_obs, inner_reward + outer_reward, done, info
