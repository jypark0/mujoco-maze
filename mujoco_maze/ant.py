"""
A four-legged robot as an explorer in the maze.
Based on `models`_ and `gym`_ (both ant and ant-v3).

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _gym: https://github.com/openai/gym
"""

from typing import Callable, Tuple

import numpy as np

from mujoco_maze.agent_model import AgentModel

ForwardRewardFn = Callable[[float, float], float]


def forward_reward_vabs(xy_velocity: float) -> float:
    return np.sum(np.abs(xy_velocity))


def forward_reward_vnorm(xy_velocity: float) -> float:
    return np.linalg.norm(xy_velocity)


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class AntEnv(AgentModel):
    FILE: str = "ant_dt2_gear30.xml"
    ORI_IND: int = 3
    MANUAL_COLLISION: bool = False
    OBJBALL_TYPE: str = "freejoint"

    def __init__(
        self,
        file_path: str,
        viewer_setup_kwargs: dict = None,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1,
        terminate_when_unhealthy: bool = False,
        healthy_z_range: Tuple[float, float] = (0.2, 1.0),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        forward_reward_fn: ForwardRewardFn = forward_reward_vnorm,
    ) -> None:
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._forward_reward_fn = forward_reward_fn
        super().__init__(file_path, 5, viewer_setup_kwargs)

    def _forward_reward(self, xy_pos_before: np.ndarray) -> Tuple[float, np.ndarray]:
        xy_pos_after = self.get_body_com("torso")[:2].copy()
        xy_velocity = (xy_pos_after - xy_pos_before) / self.dt
        return self._forward_reward_fn(xy_velocity)

    # Ref: https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)

        # Modify forward reward to include both (x,y)
        forward_reward = self._forward_reward(xy_position_before)
        healthy_reward = self.healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        # No cfrc observation
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:15],  # Ensures only ant obs.
                self.sim.data.qvel.flat[:14],
            ]
        )

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_ori(self) -> np.ndarray:
        ori = [0, 1, 0, 0]
        rot = self.sim.data.qpos[self.ORI_IND : self.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = np.arctan2(ori[1], ori[0])
        return ori

    def set_xy(self, xy: np.ndarray) -> None:
        qpos = self.sim.data.qpos.copy()
        qpos[:2] = xy
        self.set_state(qpos, self.sim.data.qvel)

    def get_xy(self) -> np.ndarray:
        return np.copy(self.sim.data.qpos[:2])
