"""
A ball-like robot as an explorer in the maze.
Based on `models`_ and `rllab`_.

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _rllab: https://github.com/rll/rllab
"""

from typing import Optional, Tuple

import gym
import numpy as np

from mujoco_maze.agent_model import AgentModel


class PointEnv(AgentModel):
    FILE: str = "point.xml"
    ORI_IND: int = 2
    MANUAL_COLLISION: bool = True
    RADIUS: float = 0.4
    OBJBALL_TYPE: str = "hinge"

    VELOCITY_LIMITS: float = 10.0

    def __init__(
        self,
        file_path: Optional[str] = None,
        viewer_setup_kwargs: dict = None,
        reset_noise_scale: float = 0.1,
    ) -> None:
        super().__init__(file_path, 1, viewer_setup_kwargs)
        self._reset_noise_scale = reset_noise_scale

        high = np.inf * np.ones(6, dtype=np.float32)
        # high[3:] = self.VELOCITY_LIMITS * 1.2
        high[self.ORI_IND] = np.pi
        low = -high
        self.observation_space = gym.spaces.Box(low, high)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        qpos = self.sim.data.qpos.copy()
        # ctrlrange=[-1, 1], scale to [-0.25, 0.25] here
        qpos[2] += 0.25 * action[1]
        # Clip orientation
        if qpos[2] < -np.pi:
            qpos[2] += np.pi * 2
        elif np.pi < qpos[2]:
            qpos[2] -= np.pi * 2
        ori = qpos[2]
        # Compute increment in each direction.
        dx = np.cos(ori) * 0.2 * action[0]
        dy = np.sin(ori) * 0.2 * action[0]
        # Ensure that the robot is within reasonable range.
        qpos[0] = np.clip(qpos[0] + dx, -100, 100)
        qpos[1] = np.clip(qpos[1] + dy, -100, 100)
        qvel = self.sim.data.qvel

        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            self.sim.step()
        next_obs = self._get_obs()
        reward = 0.0
        done = False
        info = {}

        return next_obs, reward, done, info

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:3],  # Only point-relevant coords.
                self.sim.data.qvel.flat[:3],
            ]
        )

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.sim.model.nq, low=noise_low, high=noise_high
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale
            * self.np_random.standard_normal(self.sim.model.nv)
        )

        # For debugging
        # qvel = self.init_qvel
        # qpos = self.init_qpos

        # Set everything other than point to original position and 0 velocity.
        qpos[3:] = self.init_qpos[3:]
        qvel[3:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_xy(self):
        return self.sim.data.qpos[:2].copy()

    def set_xy(self, xy: np.ndarray) -> None:
        qpos = self.sim.data.qpos.copy()
        qpos[:2] = xy
        self.set_state(qpos, self.sim.data.qvel)

    def get_ori(self):
        return self.sim.data.qpos[self.ORI_IND]
