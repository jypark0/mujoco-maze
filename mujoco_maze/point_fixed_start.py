from mujoco_maze.point import PointEnv


class PointFixedStartEnv(PointEnv):
    def reset_model(self):
        qvel = self.init_qvel
        qpos = self.init_qpos

        # Set everything other than point to original position and 0 velocity.
        qpos[3:] = self.init_qpos[3:]
        qvel[3:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()
