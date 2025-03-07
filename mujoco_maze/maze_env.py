"""
Mujoco Maze environment.
Based on `models`_ and `rllab`_.

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _rllab: https://github.com/rll/rllab
"""

import itertools as it
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Tuple, Type

import gym
import numpy as np
from gym.utils import seeding

from mujoco_maze import maze_env_utils, maze_task
from mujoco_maze.agent_model import AgentModel
from mujoco_maze.task_common import RED

# Directory that contains mujoco xml files.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets"


class MazeEnv(gym.Env):
    def __init__(
        self,
        model_cls: Type[AgentModel],
        maze_task: Type[maze_task.MazeTask] = maze_task.MazeTask,
        random_start: bool = False,
        init_position: Optional[Tuple[float, float]] = None,
        init_rowcol: Optional[Tuple[int, int]] = None,
        include_position: bool = True,
        maze_height: float = 0.5,
        maze_size_scaling: float = 4.0,
        inner_reward_scaling: float = 1.0,
        restitution_coef: float = 0.8,
        task_kwargs: dict = {},
        websock_port: Optional[int] = None,
        camera_move_x: Optional[float] = None,
        camera_move_y: Optional[float] = None,
        camera_zoom: Optional[float] = None,
        image_shape: Tuple[int, int] = (600, 480),
        **kwargs,
    ) -> None:
        self.t = 0  # time steps
        self._task = maze_task(maze_size_scaling, **task_kwargs)
        # Expose task as public attribute
        self.task = self._task

        cond = (
            int(random_start)
            + int(init_position is not None)
            + int(init_rowcol is not None)
        )
        if cond > 1:
            raise ValueError(
                "Only one of random_start, init_position, and init_rowcol should be given"
            )

        self.random_start = random_start
        self.init_position = init_position
        self.init_rowcol = init_rowcol

        self._maze_height = height = maze_height
        self._maze_size_scaling = size_scaling = maze_size_scaling
        self._inner_reward_scaling = inner_reward_scaling
        self._observe_blocks = self._task.OBSERVE_BLOCKS
        self._put_spin_near_agent = self._task.PUT_SPIN_NEAR_AGENT

        # Viewer setup args
        self.viewer_setup_kwargs = self._task.VIEWER_SETUP_KWARGS

        # Observe other objectives
        self._observe_balls = self._task.OBSERVE_BALLS
        self._top_down_view = self._task.TOP_DOWN_VIEW
        self._restitution_coef = restitution_coef

        self._maze_structure = (
            self.maze_structure
        ) = structure = self._task.create_maze()
        # Elevate the maze to allow for falling.
        self.elevated = any(maze_env_utils.MazeCell.CHASM in row for row in structure)
        # Are there any movable blocks?
        self.blocks = any(any(r.can_move() for r in row) for row in structure)

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y
        self._init_positions = [
            (x - torso_x, y - torso_y) for x, y in self._find_all_robots()
        ]

        if model_cls.MANUAL_COLLISION:
            if model_cls.RADIUS is None:
                raise ValueError("Manual collision needs radius of the model")
            self._collision = maze_env_utils.CollisionDetector(
                structure,
                size_scaling,
                torso_x,
                torso_y,
                model_cls.RADIUS,
            )
            self._objball_collision = maze_env_utils.CollisionDetector(
                structure,
                size_scaling,
                torso_x,
                torso_y,
                self._task.OBJECT_BALL_SIZE,
            )
        else:
            self._collision = None

        self._xy_to_rowcol = lambda x, y: (
            2 + (y + size_scaling / 2) / size_scaling,
            2 + (x + size_scaling / 2) / size_scaling,
        )
        # walls (immovable), chasms (fall), movable blocks
        self._view = np.zeros([5, 5, 3])

        # Let's create MuJoCo XML
        xml_path = os.path.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        height_offset = 0.0
        if self.elevated:
            # Increase initial z-pos of ant.
            height_offset = height * size_scaling
            torso = tree.find(".//body[@name='torso']")
            torso.set("pos", f"0 0 {0.75 + height_offset:.2f}")
        if self.blocks:
            # If there are movable blocks, change simulation settings to perform
            # better contact detection.
            default = tree.find(".//default")
            default.find(".//geom").set("solimp", ".995 .995 .01")

        self.movable_blocks = []
        self.object_balls = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                struct = structure[i][j]
                if struct.is_robot() and self._put_spin_near_agent:
                    struct = maze_env_utils.MazeCell.SPIN
                x, y = j * size_scaling - torso_x, i * size_scaling - torso_y
                h = height / 2 * size_scaling
                size = size_scaling * 0.5
                if self.elevated and not struct.is_chasm():
                    # Create elevated platform.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"elevated_{i}_{j}",
                        pos=f"{x} {y} {h}",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.9 0.9 1",
                    )
                if struct.is_block():
                    # Unmovable block.
                    # Offset all coordinates so that robot starts at the origin.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {h + height_offset}",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1",
                    )
                elif struct.can_move():
                    # Movable block.
                    self.movable_blocks.append(f"movable_{i}_{j}")
                    _add_movable_block(
                        worldbody,
                        struct,
                        i,
                        j,
                        size_scaling,
                        x,
                        y,
                        h,
                        height_offset,
                    )
                elif struct.is_object_ball():
                    # Movable Ball
                    self.object_balls.append(f"objball_{i}_{j}")
                    if model_cls.OBJBALL_TYPE == "hinge":
                        _add_objball_hinge(
                            worldbody,
                            i,
                            j,
                            x,
                            y,
                            self._task.OBJECT_BALL_SIZE,
                        )
                    elif model_cls.OBJBALL_TYPE == "freejoint":
                        _add_objball_freejoint(
                            worldbody,
                            i,
                            j,
                            x,
                            y,
                            self._task.OBJECT_BALL_SIZE,
                        )
                    else:
                        raise ValueError(
                            f"OBJBALL_TYPE is not registered for {model_cls}"
                        )

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if "name" not in geom.attrib:
                raise Exception("Every geom of the torso must have a name")

        # Set goals
        for i, goal in enumerate(self._task.goals):
            if goal.dim >= 3:
                z = goal.pos[2]
            elif self.elevated:
                z = maze_height * maze_size_scaling
            else:
                z = 0.0

            # Goal region
            if goal.region_size is not None:
                h = 0.01
                ET.SubElement(
                    worldbody,
                    "geom",
                    name=f"goal_region{i}",
                    pos=f"{goal.pos[0]} {goal.pos[1]} {z}",
                    size=f"{goal.region_size[0]} {goal.region_size[1]} {h}",
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba=goal.rgb.rgba_str(),
                )
            else:
                # Point goals
                if goal.custom_size is None:
                    size = f"{maze_size_scaling * 0.15}"
                else:
                    size = f"{goal.custom_size}"
                ET.SubElement(
                    worldbody,
                    "site",
                    name=f"goal_site{i}",
                    pos=f"{goal.pos[0]} {goal.pos[1]} {z}",
                    size=size,
                    rgba=goal.rgb.rgba_str(),
                )

        # Set waypoints
        if hasattr(self._task, "waypoints"):
            for i, goal in enumerate(self._task.waypoints):
                if goal.dim >= 3:
                    z = goal.pos[2]
                elif self.elevated:
                    z = maze_height * maze_size_scaling
                else:
                    z = 0.0
                if goal.custom_size is None:
                    size = f"{maze_size_scaling * 0.1}"
                else:
                    size = f"{goal.custom_size}"
                ET.SubElement(
                    worldbody,
                    "site",
                    name=f"waypoint_site{i}",
                    pos=f"{goal.pos[0]} {goal.pos[1]} {z}",
                    size=size,
                    rgba=goal.rgb.rgba_str(),
                )

        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)
        self.world_tree = tree
        self.wrapped_env = model_cls(
            file_path=file_path,
            viewer_setup_kwargs=self._task.VIEWER_SETUP_KWARGS,
            **kwargs,
        )
        self.observation_space = self._get_obs_space()
        self._websock_port = websock_port
        self._camera_move_x = camera_move_x
        self._camera_move_y = camera_move_y
        self._camera_zoom = camera_zoom
        self._image_shape = image_shape
        self._mj_offscreen_viewer = None
        self._websock_server_pipe = None

        # Check that init_position is valid
        xmin, xmax, ymin, ymax = self._xy_limits()
        if init_position is not None:
            if not (xmin + 1 <= self.init_position[0] <= xmax - 1) or not (
                ymin + 1 <= self.init_position[1] <= ymax - 1
            ):
                raise ValueError(
                    f"Init_position {self.init_position} is not within limits"
                )

        # Check that init_rowcol is valid
        if init_rowcol is not None:
            if (
                not (
                    0 <= init_rowcol[0] < len(structure)
                    and 0 <= init_rowcol[1] < len(structure[0])
                )
                or structure[init_rowcol[0]][init_rowcol[1]].is_wall_or_chasm()
            ):
                raise ValueError(f"Init_rowcol {self.init_rowcol} is not valid")

        # Added to enable video_recording
        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array", "top_down"],
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_top_down_map(self):
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling

        tile_size = 1

        # Calculate cell_size
        cell_size = int(tile_size * size_scaling)

        # Map size: W x H x 3 (RGB)
        map_view = 255 * (
            np.ones(
                (cell_size * len(structure), cell_size * len(structure[0]), 3),
                dtype=np.uint8,
            )
        )

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j].is_block():
                    # Walls are gray
                    map_view[
                        cell_size * i : cell_size * (i + 1),
                        cell_size * j : cell_size * (j + 1),
                        :,
                    ] = 100

        # Need to flip map y-axis to match camera view
        map_view = np.flip(map_view, axis=0)

        # X is horizontal, y is vertical for map (camera azimuth is 90)
        # Will flip y coordinates later, account for this in shift_y
        shift_x = tile_size * self._init_torso_x + (cell_size * 0.5) - 0.5
        shift_y = (
            map_view.shape[0]
            - (tile_size * self._init_torso_y + (cell_size * 0.5))
            - 0.5
        )

        return map_view, tile_size, (shift_x, shift_y)

    @property
    def has_extended_obs(self) -> bool:
        return self._top_down_view or self._observe_blocks or self._observe_balls

    def get_ori(self) -> float:
        return self.wrapped_env.get_ori()

    def _get_obs_space(self) -> gym.spaces.Box:
        shape = self._get_obs().shape
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        # Set velocity limits
        wrapped_obs_space = self.wrapped_env.observation_space
        high[: wrapped_obs_space.shape[0]] = wrapped_obs_space.high
        low[: wrapped_obs_space.shape[0]] = wrapped_obs_space.low
        # Set coordinate limits
        low[0], high[0], low[1], high[1] = self._xy_limits()
        # Set orientation limits
        return gym.spaces.Box(low, high)

    def _xy_limits(self) -> Tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = 100, 100, -100, -100
        structure = self._maze_structure
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_block():
                continue
            xmin, xmax = min(xmin, j), max(xmax, j)
            ymin, ymax = min(ymin, i), max(ymax, i)
        x0, y0 = self._init_torso_x, self._init_torso_y
        scaling = self._maze_size_scaling
        xmin, xmax = (xmin - 0.5) * scaling - x0, (xmax + 0.5) * scaling - x0
        ymin, ymax = (ymin - 0.5) * scaling - y0, (ymax + 0.5) * scaling - y0
        return xmin, xmax, ymin, ymax

    # Public method
    def xy_limits(self) -> Tuple[float, float, float, float]:
        return self._xy_limits()

    def get_top_down_view(self) -> np.ndarray:
        self._view = np.zeros_like(self._view)

        def valid(row, col):
            return self._view.shape[0] > row >= 0 and self._view.shape[1] > col >= 0

        def update_view(x, y, d, row=None, col=None):
            if row is None or col is None:
                x = x - self._robot_x
                y = y - self._robot_y

                row, col = self._xy_to_rowcol(x, y)
                update_view(x, y, d, row=row, col=col)
                return

            row, row_frac, col, col_frac = int(row), row % 1, int(col), col % 1
            if row_frac < 0:
                row_frac += 1
            if col_frac < 0:
                col_frac += 1

            if valid(row, col):
                self._view[row, col, d] += (
                    min(1.0, row_frac + 0.5) - max(0.0, row_frac - 0.5)
                ) * (min(1.0, col_frac + 0.5) - max(0.0, col_frac - 0.5))
            if valid(row - 1, col):
                self._view[row - 1, col, d] += (max(0.0, 0.5 - row_frac)) * (
                    min(1.0, col_frac + 0.5) - max(0.0, col_frac - 0.5)
                )
            if valid(row + 1, col):
                self._view[row + 1, col, d] += (max(0.0, row_frac - 0.5)) * (
                    min(1.0, col_frac + 0.5) - max(0.0, col_frac - 0.5)
                )
            if valid(row, col - 1):
                self._view[row, col - 1, d] += (
                    min(1.0, row_frac + 0.5) - max(0.0, row_frac - 0.5)
                ) * (max(0.0, 0.5 - col_frac))
            if valid(row, col + 1):
                self._view[row, col + 1, d] += (
                    min(1.0, row_frac + 0.5) - max(0.0, row_frac - 0.5)
                ) * (max(0.0, col_frac - 0.5))
            if valid(row - 1, col - 1):
                self._view[row - 1, col - 1, d] += (max(0.0, 0.5 - row_frac)) * max(
                    0.0, 0.5 - col_frac
                )
            if valid(row - 1, col + 1):
                self._view[row - 1, col + 1, d] += (max(0.0, 0.5 - row_frac)) * max(
                    0.0, col_frac - 0.5
                )
            if valid(row + 1, col + 1):
                self._view[row + 1, col + 1, d] += (max(0.0, row_frac - 0.5)) * max(
                    0.0, col_frac - 0.5
                )
            if valid(row + 1, col - 1):
                self._view[row + 1, col - 1, d] += (max(0.0, row_frac - 0.5)) * max(
                    0.0, 0.5 - col_frac
                )

        # Draw ant.
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        self._robot_x = robot_x
        self._robot_y = robot_y

        structure = self._maze_structure
        size_scaling = self._maze_size_scaling

        # Draw immovable blocks and chasms.
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j].is_block():  # Wall.
                    update_view(
                        j * size_scaling - self._init_torso_x,
                        i * size_scaling - self._init_torso_y,
                        0,
                    )
                if structure[i][j].is_chasm():  # Chasm.
                    update_view(
                        j * size_scaling - self._init_torso_x,
                        i * size_scaling - self._init_torso_y,
                        1,
                    )

        # Draw movable blocks.
        for block_name in self.movable_blocks:
            block_x, block_y = self.wrapped_env.get_body_com(block_name)[:2]
            update_view(block_x, block_y, 2)

        return self._view

    def _get_obs(self) -> np.ndarray:
        wrapped_obs = self.wrapped_env._get_obs()
        if self._top_down_view:
            view = [self.get_top_down_view().flat]
        else:
            view = []

        additional_obs = []

        if self._observe_balls:
            for name in self.object_balls:
                additional_obs.append(self.wrapped_env.get_body_com(name))

        if self._observe_blocks:
            for name in self.movable_blocks:
                additional_obs.append(self.wrapped_env.get_body_com(name))

        obs = np.concatenate([wrapped_obs[:3]] + additional_obs + [wrapped_obs[3:]])

        # Update current_goal if current_goal was reached
        if hasattr(self._task, "current_goal"):
            self._task.update_current_goal(obs)

        return np.concatenate([obs, *view, np.array([self.t * 0.001])])

    def _valid_xy(self, xy):
        row, col = self._xy_to_rowcol(*xy)
        row, col = int(row), int(col)
        ret = True
        if self._maze_structure[row][col].is_block():
            ret = False

        return ret

    def valid_rowcol(self):
        valid_rowcol = []
        structure = self._maze_structure
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if not structure[i][j].is_wall_or_chasm():
                valid_rowcol.append((i, j))

        return valid_rowcol

    def reset(self) -> np.ndarray:
        self.t = 0
        self.wrapped_env.reset()

        # Reset waypoints if needed
        if hasattr(self._task, "waypoint_idx"):
            self._task.waypoint_idx = 0
            self._task.current_goal = self._task.waypoints[self._task.waypoint_idx]

        # Samples a new goal
        if self._task.sample_goals():
            self.set_marker()

        # Samples a new start position
        if len(self._init_positions) > 1:
            xy = self.np_random.choice(self._init_positions)
            self.wrapped_env.set_xy(xy)

        # If init_position is specified
        if self.init_position is not None:
            self.init_pos = self.init_position
            self.init_rc = self._xy_to_rowcol(*self.init_pos)

        # If init_rowcol is specified
        elif self.init_rowcol is not None:
            self.init_rc = self.init_rowcol
            r, c = self.init_rowcol
            structure = self._maze_structure

            # Transform r,c into random xy within size_scaling / 2
            x0, y0 = self._init_torso_x, self._init_torso_y
            scaling = self._maze_size_scaling
            xmin, xmax = (c - 0.5) * scaling - x0, (c + 0.5) * scaling - x0
            ymin, ymax = (r - 0.5) * scaling - y0, (r + 0.5) * scaling - y0

            # Because of collision radius, point has a chance to get stuck near the walls (can't get out)
            # Add 1 distance if next cell is wall or chasm
            # bottom
            if r > 0 and structure[r - 1][c].is_wall_or_chasm():
                ymin += 1
            # top
            if r < len(structure) - 1 and structure[r + 1][c].is_wall_or_chasm():
                ymax -= 1
            # left
            if c > 0 and structure[r][c - 1].is_wall_or_chasm():
                xmin += 1
            # right
            if c < len(structure[0]) - 1 and structure[r][c + 1].is_wall_or_chasm():
                xmax -= 1

            # Randomly sample anywhere within that cell
            xy = self.np_random.uniform([xmin, ymin], [xmax, ymax], 2)

            self.init_pos = xy

        # Samples random start position
        elif self.random_start:
            valid_rowcol = self.valid_rowcol()
            structure = self._maze_structure

            # Sample random row and col
            idx = self.np_random.randint(0, len(valid_rowcol))
            r, c = valid_rowcol[idx]

            # Transform r,c into random xy within size_scaling / 2
            x0, y0 = self._init_torso_x, self._init_torso_y
            scaling = self._maze_size_scaling
            xmin, xmax = (c - 0.5) * scaling - x0, (c + 0.5) * scaling - x0
            ymin, ymax = (r - 0.5) * scaling - y0, (r + 0.5) * scaling - y0

            # Because of collision radius, point has a chance to get stuck near the walls (can't get out)
            # Add 1 distance if next cell is wall or chasm
            x_change = False
            y_change = False
            # bottom
            if r > 0 and structure[r - 1][c].is_wall_or_chasm():
                ymin += 1
                y_change = True
            # top
            if r < len(structure) - 1 and structure[r + 1][c].is_wall_or_chasm():
                ymax -= 1
                y_change = True
            # left
            if c > 0 and structure[r][c - 1].is_wall_or_chasm():
                xmin += 1
                x_change = True
            # right
            if c < len(structure[0]) - 1 and structure[r][c + 1].is_wall_or_chasm():
                xmax -= 1
                x_change = True

            # # Also check corners (doesn't work)
            # # bottom left corner
            # if r > 0 and c > 0 and structure[r - 1][c - 1].is_wall_or_chasm():
            #     if not y_change:
            #         ymin += 1
            #     if not x_change:
            #         xmin += 1
            # # bottom right corner
            # if (
            #     r > 0
            #     and c < len(structure[0]) - 1
            #     and structure[r - 1][c + 1].is_wall_or_chasm()
            # ):
            #     if not y_change:
            #         ymin += 1
            #     if not x_change:
            #         xmax -= 1
            # # top left corner
            # if (
            #     r < len(structure[0]) - 1
            #     and c > 0
            #     and structure[r + 1][c - 1].is_wall_or_chasm()
            # ):
            #     if not y_change:
            #         ymax -= 1
            #     if not x_change:
            #         xmin += 1
            # # bottom right corner
            # if (
            #     r < len(structure[0]) - 1
            #     and c < len(structure[0]) - 1
            #     and structure[r + 1][c + 1].is_wall_or_chasm()
            # ):
            #     if not y_change:
            #         ymax -= 1
            #     if not x_change:
            #         xmax -= 1

            # Randomly sample anywhere within that cell
            xy = self.np_random.uniform([xmin, ymin], [xmax, ymax], 2)

            self.init_pos = xy
        else:
            # Store init_pos (for chasm: resetting to init pos)
            self.init_pos = self.wrapped_env.get_xy()

        # Set pos to self.init_position
        self.wrapped_env.set_xy(np.asarray(self.init_pos))

        return self._get_obs()

    def set_marker(self) -> None:
        for i, goal in enumerate(self._task.goals):
            idx = self.model.site_name2id(f"goal{i}")
            self.data.site_xpos[idx][: len(goal.pos)] = goal.pos

    def _render_image(self) -> np.ndarray:
        self._mj_offscreen_viewer._set_mujoco_buffers()
        self._mj_offscreen_viewer.render(*self._image_shape)
        pixels = self._mj_offscreen_viewer.read_pixels(*self._image_shape, depth=False)
        return np.asarray(pixels[::-1, :, :], dtype=np.uint8)

    def _maybe_move_camera(self, viewer: Any) -> None:
        from mujoco_py import const

        if self._camera_move_x is not None:
            viewer.move_camera(const.MOUSE_ROTATE_V, self._camera_move_x, 0.0)
        if self._camera_move_y is not None:
            viewer.move_camera(const.MOUSE_ROTATE_H, 0.0, self._camera_move_y)
        if self._camera_zoom is not None:
            viewer.move_camera(const.MOUSE_ZOOM, 0, self._camera_zoom)

    def render(self, mode="human", **kwargs) -> Optional[np.ndarray]:
        if mode == "human" and self._websock_port is not None:
            if self._mj_offscreen_viewer is None:
                from mujoco_py import MjRenderContextOffscreen as MjRCO

                from mujoco_maze.websock_viewer import start_server

                self._mj_offscreen_viewer = MjRCO(self.wrapped_env.sim)
                self._maybe_move_camera(self._mj_offscreen_viewer)
                self._websock_server_pipe = start_server(self._websock_port)
            return self._websock_server_pipe.send(self._render_image())
        else:
            width, height = self._task.RENDER_WIDTH, self._task.RENDER_HEIGHT
            if self.wrapped_env.viewer is None:
                # Use render_shape specified in maze_task
                self.wrapped_env.render(mode, width=width, height=height, **kwargs)
                self._maybe_move_camera(self.wrapped_env.viewer)
            # Use render_shape specified in maze_task
            return self.wrapped_env.render(mode, width=width, height=height, **kwargs)

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def _find_robot(self) -> Tuple[float, float]:
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                return j * size_scaling, i * size_scaling
        raise ValueError("No robot in maze specification.")

    def _find_all_robots(self) -> List[Tuple[float, float]]:
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        coords = []
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                coords.append((j * size_scaling, i * size_scaling))
        return coords

    def _objball_positions(self) -> None:
        return [
            self.wrapped_env.get_body_com(name)[:2].copy() for name in self.object_balls
        ]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.t += 1
        done = False
        if self.wrapped_env.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            old_objballs = self._objball_positions()
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            new_objballs = self._objball_positions()
            # Checks that the new_position is in the wall
            collision = self._collision.detect(old_pos, new_pos, "walls")
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

            # Check if new_position is in chasm
            falls = self._collision.detect(old_pos, new_pos, "chasm")
            if falls is not None:
                # Reset to this episodes initial position
                # Must also reset_model to reset velocities and orientation
                # new_pos = self.init_pos
                # self.wrapped_env.reset_model()
                # self.wrapped_env.set_xy(new_pos)

                # Just set done to True
                done = True

        else:
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)

        next_obs = self._get_obs()

        inner_reward = self._inner_reward_scaling * inner_reward
        outer_reward = self._task.reward(next_obs)
        done = done or self._task.termination(next_obs)
        info["position"] = self.wrapped_env.get_xy()
        info["is_success"] = self._task.termination(next_obs)
        return next_obs, inner_reward + outer_reward, done, info

    def close(self) -> None:
        self.wrapped_env.close()
        if self._websock_server_pipe is not None:
            self._websock_server_pipe.send(None)


def _add_objball_hinge(
    worldbody: ET.Element,
    i: str,
    j: str,
    x: float,
    y: float,
    size: float,
) -> None:
    body = ET.SubElement(worldbody, "body", name=f"objball_{i}_{j}", pos=f"{x} {y} 0")
    mass = 0.0001 * (size**3)
    ET.SubElement(
        body,
        "geom",
        type="sphere",
        name=f"objball_{i}_{j}_geom",
        size=f"{size}",  # Radius
        pos=f"0.0 0.0 {size}",  # Z = size so that this ball can move!!
        rgba=maze_task.BLUE.rgba_str(),
        contype="1",
        conaffinity="1",
        solimp="0.9 0.99 0.001",
        mass=f"{mass}",
    )
    ET.SubElement(
        body,
        "joint",
        name=f"objball_{i}_{j}_x",
        axis="1 0 0",
        pos="0 0 0",
        type="slide",
    )
    ET.SubElement(
        body,
        "joint",
        name=f"objball_{i}_{j}_y",
        axis="0 1 0",
        pos="0 0 0",
        type="slide",
    )
    ET.SubElement(
        body,
        "joint",
        name=f"objball_{i}_{j}_rot",
        axis="0 0 1",
        pos="0 0 0",
        type="hinge",
        limited="false",
    )


def _add_objball_freejoint(
    worldbody: ET.Element,
    i: str,
    j: str,
    x: float,
    y: float,
    size: float,
) -> None:
    body = ET.SubElement(worldbody, "body", name=f"objball_{i}_{j}", pos=f"{x} {y} 0")
    ET.SubElement(
        body,
        "geom",
        type="sphere",
        name=f"objball_{i}_{j}_geom",
        size=f"{size}",  # Radius
        pos=f"0.0 0.0 {size}",  # Z = size so that this ball can move!!
        rgba=maze_task.BLUE.rgba_str(),
        contype="1",
        conaffinity="1",
        solimp="0.9 0.99 0.001",
    )
    ET.SubElement(body, "freejoint", name=f"objball_{i}_{j}_root")


def _add_movable_block(
    worldbody: ET.Element,
    struct: maze_env_utils.MazeCell,
    i: str,
    j: str,
    size_scaling: float,
    x: float,
    y: float,
    h: float,
    height_offset: float,
) -> None:
    falling = struct.can_move_z()
    if struct.can_spin():
        h *= 0.1
        x += size_scaling * 0.25
        shrink = 0.1
    elif falling:
        # The "falling" blocks are shrunk slightly and increased in mass to
        # ensure it can fall easily through a gap in the platform blocks.
        shrink = 0.99
    elif struct.is_half_block():
        shrink = 0.5
    else:
        shrink = 1.0
    size = size_scaling * 0.5 * shrink
    movable_body = ET.SubElement(
        worldbody,
        "body",
        name=f"movable_{i}_{j}",
        pos=f"{x} {y} {h}",
    )
    ET.SubElement(
        movable_body,
        "geom",
        name=f"block_{i}_{j}",
        pos="0 0 0",
        size=f"{size} {size} {h}",
        type="box",
        material="",
        mass="0.001" if falling else "0.0002",
        contype="1",
        conaffinity="1",
        rgba="0.9 0.1 0.1 1",
    )
    if struct.can_move_x():
        ET.SubElement(
            movable_body,
            "joint",
            axis="1 0 0",
            name=f"movable_x_{i}_{j}",
            armature="0",
            damping="0.0",
            limited="true" if falling else "false",
            range=f"{-size_scaling} {size_scaling}",
            margin="0.01",
            pos="0 0 0",
            type="slide",
        )
    if struct.can_move_y():
        ET.SubElement(
            movable_body,
            "joint",
            armature="0",
            axis="0 1 0",
            damping="0.0",
            limited="true" if falling else "false",
            range=f"{-size_scaling} {size_scaling}",
            margin="0.01",
            name=f"movable_y_{i}_{j}",
            pos="0 0 0",
            type="slide",
        )
    if struct.can_move_z():
        ET.SubElement(
            movable_body,
            "joint",
            armature="0",
            axis="0 0 1",
            damping="0.0",
            limited="true",
            range=f"{-height_offset} 0",
            margin="0.01",
            name=f"movable_z_{i}_{j}",
            pos="0 0 0",
            type="slide",
        )
    if struct.can_spin():
        ET.SubElement(
            movable_body,
            "joint",
            armature="0",
            axis="0 0 1",
            damping="0.0",
            limited="false",
            name=f"spinable_{i}_{j}",
            pos="0 0 0",
            type="ball",
        )
