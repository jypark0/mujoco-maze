"""Common APIs for defining mujoco robot.
"""
from abc import ABC, abstractmethod
from typing import Optional

import mujoco_py
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.utils import EzPickle

DEFAULT_SIZE = 500


class AgentModel(ABC, MujocoEnv, EzPickle):
    FILE: str
    MANUAL_COLLISION: bool
    ORI_IND: Optional[int] = None
    RADIUS: Optional[float] = None
    OBJBALL_TYPE: Optional[str] = None

    def __init__(self, file_path: str, frame_skip: int, viewer_setup_kwargs) -> None:
        MujocoEnv.__init__(self, file_path, frame_skip)
        EzPickle.__init__(self)
        self.viewer_setup_kwargs = viewer_setup_kwargs

    def close(self):
        if self.viewer is not None and hasattr(self.viewer, "window"):
            import glfw

            glfw.destroy_window(self.viewer.window)
        super().close()

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Returns the observation from the model."""
        pass

    def get_xy(self) -> np.ndarray:
        """Returns the coordinate of the agent."""
        pass

    def set_xy(self, xy: np.ndarray) -> None:
        """Set the coordinate of the agent."""
        pass

    def top_down_viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.distance = self.model.stat.extent
        self.viewer.cam.distance = self.model.stat.extent / 2
        for i in range(3):
            self.viewer.cam.lookat[i] = self.model.stat.center[i]
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = (
            self.model.stat.extent * self.viewer_setup_kwargs["distance"]
        )

        # View center of model
        self.viewer.cam.lookat[0] = self.model.stat.center[0]
        self.viewer.cam.lookat[1] = self.model.stat.center[1]
        self.viewer.cam.lookat[2] = self.model.stat.center[2]

        self.viewer.cam.elevation = self.viewer_setup_kwargs["elevation"]
        self.viewer.cam.azimuth = self.viewer_setup_kwargs["azimuth"]

    # Workaround to get camera to track agent when using mode="human"
    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "top_down":
            geom_pos = self.model.geom_pos
            min_x, max_x = np.min(geom_pos[:, 0]), np.max(geom_pos[:, 0])
            min_y, max_y = np.min(geom_pos[:, 1]), np.max(geom_pos[:, 1])
            width = int(max_x - min_x) * 10
            height = int(max_y - min_y) * 10

        if mode in ["rgb_array", "depth_array", "top_down"]:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array" or mode == "top_down":
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            # self._get_viewer(mode).render(camera_id=camera_id)
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
                self.viewer_setup()
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
                self.viewer_setup()
            elif mode == "top_down":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
                self.top_down_viewer_setup()

            self._viewers[mode] = self.viewer

        return self.viewer
