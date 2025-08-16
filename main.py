import math
import time
import random
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class GridDroneEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        grid_size=5,
        render: bool = True,
        drone_urdf_path="drone.urdf",
        lidar_num_rays=50,           # 2D lidar, horizontal plane
        lidar_max_range=8.0,
        max_vel=4.0,
        hover_z=1,
        timestep=1.0 / 60.0,
        episode_timeout=20.0,
        draw_lidar=True,
        action_smooth_alpha=0.2, # smoothing factor for velocity commands
        stage = 0   
    ):
        super().__init__()
        self.grid_size = grid_size
        self.render = bool(render)
        self.drone_urdf_path = drone_urdf_path
        self.lidar_num_rays = int(lidar_num_rays)
        self.lidar_max_range = float(lidar_max_range)
        self.max_vel = float(max_vel)
        self.hover_z = float(hover_z)
        self.timestep = float(timestep)
        self.episode_timeout = float(episode_timeout)
        self.prev_pos = None
        self.stuck_counter = 0
        self.min_movement_threshold = 0.05  # meters (small)
        self.stuck_penalty_scale = 0.1 * self.grid_size / 5  # scale for stuck penalty
        self.max_stuck_penalty = 5.0 * self.grid_size / 5 
        self.draw_lidar = bool(draw_lidar) and self.render
        self.start_distance = 0.0

        # smoothing alpha for actions (exponential smoothing)
        self.action_smooth_alpha = float(action_smooth_alpha)
        self.smoothed_action = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)

        # === Tuning params ===
        # Scaled down for stability
        self.counter = 0.0
        self.progress_scale = 15.0  * self.grid_size / 5       # dense shaping from progress (per step)
        self.goal_reward = 50.0 * self.grid_size / 5          # terminal goal reward (bigger to prioritize reaching)
        self.goal_reward_final = 120.0 * self.grid_size / 5    # final goal reward to reach the center
        self.collision_base_penalty = 25.0 * self.grid_size / 5  # stronger collision penalty
        self.obs_close_threshold = 0.6 * self.grid_size / 5    # lidar threshold for proximity penalty (meters)
        self.obs_penalty_weight = 0.5 * self.grid_size / 5     # weight for normalized obstacle proximity penalty
        self.forward_vel_weight = 0.1 * self.grid_size / 5    # reward for velocity component toward goal
        self.speed_penalty = 0.01 * self.grid_size / 5         # small penalty on raw speed to discourage reckless fast moves
        self.min_goal_dist_for_rdist = 0.05 * self.grid_size / 5 
        self.max_rdist = 0.3 * self.grid_size / 5           # cap on distance shaping term to avoid spikes
        self.smoothness_penalty_weight = 0.03 * self.grid_size / 5  # penalty on action change
        self.bound_penalty = 40  * self.grid_size / 5  # penalty to go outside the bounds

        # Stuck detection thresholds (seconds -> steps)
        self.stuck_warning_seconds = 2.0
        self.stuck_terminate_seconds = 4.0
        self.stuck_warning_steps = int(self.stuck_warning_seconds / self.timestep)
        self.stuck_terminate_steps = int(self.stuck_terminate_seconds / self.timestep)

        # Cell types
        self.EMPTY = 0
        self.L_WALL = 1
        self.S_WALL = 2
        self.MOVE_LEFT_RIGHT = 3
        self.MOVE_UP_DOWN = 4
        self.SPAWN = 5
        self.TARGET = 6

        self.cell_size = 1.0
        self.objects = []  # (body_id, x, y, type)
        self.moving_blocks = []
        self.spawn_pos = None
        self.target_pos = None

        self.elapsed_time = 0.0

        # Curriculum / stages
        # 0: no obstacles
        # 1: light obstacles (user provided weights)
        # 2: normal obstacles (user provided weights)
        # 3 : autochange 

        self.stage_level = stage
        self.curr_stage = 0
        self.stage_success_count = 0
        self.stage_episode_count = 0
        self.stage_min_episodes = 20  # require at least this many episodes before considering advancement

        self._connect()

        # Observation space: [vx, vy, px, py, lidar_distances..., goal_dx, goal_dy, prev_action_x, prev_action_y]
        obs_high = np.concatenate(
            [
                np.array([self.max_vel * 1.5, self.max_vel * 1.5], dtype=np.float32),
                np.array([self.grid_size * self.cell_size * 2, self.grid_size * self.cell_size * 2], dtype=np.float32),
                np.ones(self.lidar_num_rays, dtype=np.float32) * self.lidar_max_range,
                np.array([self.grid_size * self.cell_size * np.sqrt(2), self.grid_size * self.cell_size * np.sqrt(2)], dtype=np.float32),
                np.array([self.max_vel * 1.5, self.max_vel * 1.5], dtype=np.float32),  # prev_action bounds
            ]
        ).astype(np.float32)

        obs_low = -obs_high

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_vel, high=self.max_vel, shape=(2,), dtype=np.float32)

        self._compute_lidar_directions()

        self.reset()

    def _connect(self):
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # Disable gravity for true 2D dynamics
        p.setTimeStep(self.timestep)

    def create_black_ground(self):
        half_size = (self.grid_size * self.cell_size) / 2.0 + 0.01
        thickness = 0.01
        ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size, half_size, thickness])
        ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_size + 1, half_size + 1, thickness], rgbaColor=[0, 0, 0, 1])
        ground_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_collision, baseVisualShapeIndex=ground_visual, basePosition=[half_size, half_size, 0])
        return ground_id

    def _cell_to_world(self, x, y):
        world_x = x * self.cell_size + self.cell_size / 2.0
        world_y = y * self.cell_size + self.cell_size / 2.0
        world_z = 0.0
        return (world_x, world_y, world_z)

    def _spawn_wall(self, pos, l_shape=False):
        angle = random.choice([0, 90, 180, 270])
        orn = p.getQuaternionFromEuler([0, 0, math.radians(angle)])
        urdf = "l_wall.urdf" if l_shape else "s_wall.urdf"
        try:
            body_id = p.loadURDF(urdf, [pos[0], pos[1], 0], orn, useFixedBase=True)
        except Exception:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[0.5, 0.5, 0.5, 1])
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[pos[0], pos[1], 0.4])
        self.objects.append((body_id, pos[0], pos[1], 'wall'))

    def _spawn_moving_block(self, pos, horizontal=True, cell_x=None, cell_y=None):
        urdf = "slab_h.urdf" if horizontal else "slab_v.urdf"
        start_pos = [pos[0], pos[1], 1.5] if horizontal else [pos[0], pos[1], 0.75]
        try:
            body_id = p.loadURDF(urdf, start_pos)
        except Exception:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.2], rgbaColor=[0.7, 0.2, 0.2, 1])
            body_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=start_pos)
        obj = {"id": body_id, "start_pos": np.array(start_pos, dtype=np.float32), "horizontal": horizontal, "cell_x": cell_x, "cell_y": cell_y, "direction": 1, "speed": 0.5, "range": 0.4, "offset": 0}
        self.objects.append((body_id, pos[0], pos[1], 'moving_block'))
        self.moving_blocks.append(obj)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        self.objects.clear()
        self.moving_blocks.clear()
        self.spawn_pos = None
        self.target_pos = None
        self.elapsed_time = 0.0
        self.stuck_counter = 0
        self.prev_pos = None

        # keep smoothed action and prev_action but reset to zeros
        self.smoothed_action = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.floor_id = self.create_black_ground()

        grid = [[self.EMPTY for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        while True:
            spawn_x, spawn_y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            target_x, target_y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            dist = math.sqrt((target_x - spawn_x) ** 2 + (target_y - spawn_y) ** 2)
            if dist >= self.grid_size / 2:
                break
        grid[spawn_y][spawn_x] = self.SPAWN
        grid[target_y][target_x] = self.TARGET

        # choose obstacle sampling weights based on curriculum stage
        choices = [self.EMPTY, self.L_WALL, self.S_WALL, self.MOVE_LEFT_RIGHT, self.MOVE_UP_DOWN]
        
        if self.stage_level != 3:
            self.curr_stage = self.stage_level

        if self.curr_stage == 0:
            weights = [1.0, 0.0, 0.0, 0.0, 0.0]  # no obstacles
        elif self.curr_stage == 1:
            weights = [0.8, 0.0, 0.2, 0.0, 0.0]  # light obstacles (user provided)
        elif self.curr_stage == 2:
            weights = [0.7, 0.1, 0.1, 0.1, 0.0]  # normal obstacles (user provided)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if grid[y][x] == self.EMPTY:
                    grid[y][x] = random.choices(choices, weights)[0]

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_type = grid[y][x]
                world_pos = self._cell_to_world(x, y)
                if cell_type == self.EMPTY:
                    continue
                elif cell_type == self.L_WALL:
                    self._spawn_wall(world_pos, l_shape=True)
                elif cell_type == self.S_WALL:
                    self._spawn_wall(world_pos, l_shape=False)
                elif cell_type == self.MOVE_LEFT_RIGHT:
                    self._spawn_moving_block(world_pos, horizontal=True, cell_x=x, cell_y=y)
                elif cell_type == self.MOVE_UP_DOWN:
                    self._spawn_moving_block(world_pos, horizontal=False, cell_x=x, cell_y=y)
                elif cell_type == self.SPAWN:
                    self.spawn_pos = world_pos
                    try:
                        self.spawn_id = p.loadURDF("slab_v.urdf", [world_pos[0], world_pos[1], 0], useFixedBase=True)
                        if self.render:
                            p.changeVisualShape(self.spawn_id, -1, rgbaColor=[1, 0, 0, 1])
                    except Exception:
                        self.spawn_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=[world_pos[0], world_pos[1], 0])
                elif cell_type == self.TARGET:
                    self.target_pos = world_pos
                    try:
                        self.target_id = p.loadURDF("slab_v.urdf", [world_pos[0], world_pos[1], 0], useFixedBase=True)
                        if self.render:
                            p.changeVisualShape(self.target_id, -1, rgbaColor=[0, 1, 0, 1])
                    except Exception:
                        self.target_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=[world_pos[0], world_pos[1], 0])

        if self.spawn_pos is None:
            cx = (self.grid_size * self.cell_size) / 2.0
            cy = (self.grid_size * self.cell_size) / 2.0
            start = [cx, cy, self.hover_z]
        else:
            start = [self.spawn_pos[0], self.spawn_pos[1], self.hover_z]

        try:
            self.drone_id = p.loadURDF(self.drone_urdf_path, start, useFixedBase=False)
        except Exception:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], rgbaColor=[0.2, 0.6, 0.9, 1])
            self.drone_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=start)

        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        p.resetBasePositionAndOrientation(self.drone_id, [pos[0], pos[1], self.hover_z], orn)
        p.resetBaseVelocity(self.drone_id, linearVelocity=[0.0, 0.0, 0.0], angularVelocity=[0.0, 0.0, 0.0])



        self.prev_goal_dist = self._distance_to_goal()
        self.prev_pos = None
        self.stuck_counter = 0
        svector = np.array(self.spawn_pos) - np.array(self.target_pos)
        self.start_distance = np.linalg.norm(svector) if svector is not None else 2.5

        self.set_camera_top_down()

        obs = self._get_obs()
        return obs, {}

    def _compute_lidar_directions(self):
        angles = np.linspace(-math.pi, math.pi, self.lidar_num_rays, endpoint=False)
        dirs = []
        for a in angles:
            x = math.cos(a)
            y = math.sin(a)
            z = 0.0
            dirs.append((x, y, z))
        self.lidar_dirs_local = np.array(dirs, dtype=np.float32)

    def _distance_to_goal(self):
        if self.target_pos is None:
            return float("inf")
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        dx = pos[0] - self.target_pos[0]
        dy = pos[1] - self.target_pos[1]
        return math.sqrt(dx * dx + dy * dy)

    def _step_moving_blocks(self):
        for mb in self.moving_blocks:
            offset = mb["offset"] + mb["speed"] * mb["direction"] * self.timestep
            if abs(offset) > mb["range"]:
                mb["direction"] *= -1
                offset = mb["offset"] + mb["speed"] * mb["direction"] * self.timestep
            mb["offset"] = offset
            pos = mb["start_pos"].copy()
            if mb["horizontal"]:
                pos[0] += offset
            else:
                pos[2] += offset
            p.resetBasePositionAndOrientation(mb["id"], pos.tolist(), [0, 0, 0, 1])

    def _perform_lidar(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.drone_id)
        base_pos = np.array(base_pos, dtype=np.float32)
        euler = p.getEulerFromQuaternion(base_orn)
        yaw = euler[2]
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        ray_from_positions = []
        ray_to_positions = []

        origin = base_pos + np.array([0.0, 0.0, 0.1], dtype=np.float32)  # offset upward to avoid ground

        for d in self.lidar_dirs_local:
            lx, ly, lz = float(d[0]), float(d[1]), float(d[2])
            # Rotate directions by yaw
            wx = lx * cos_y - ly * sin_y
            wy = lx * sin_y + ly * cos_y
            wz = lz
            ray_from_positions.append(origin.tolist())
            ray_to_positions.append((origin + np.array([wx, wy, wz]) * self.lidar_max_range).tolist())

        results = p.rayTestBatch(ray_from_positions, ray_to_positions)
        dists = np.ones(self.lidar_num_rays, dtype=np.float32) * self.lidar_max_range

        if self.draw_lidar:
            p.removeAllUserDebugItems()

        for i, r in enumerate(results):
            hit_uid = r[0]
            hit_fraction = r[2]
            if hit_uid >= 0:
                dist = hit_fraction * self.lidar_max_range
                dists[i] = dist
                if self.draw_lidar:
                    hit_pos = [
                        ray_from_positions[i][0] + (ray_to_positions[i][0] - ray_from_positions[i][0]) * hit_fraction,
                        ray_from_positions[i][1] + (ray_to_positions[i][1] - ray_from_positions[i][1]) * hit_fraction,
                        ray_from_positions[i][2] + (ray_to_positions[i][2] - ray_from_positions[i][2]) * hit_fraction,
                    ]
                    p.addUserDebugLine(ray_from_positions[i], hit_pos, [1, 0, 0], lineWidth=1, lifeTime=self.timestep * 4)
            else:
                dists[i] = self.lidar_max_range
                if self.draw_lidar:
                    p.addUserDebugLine(ray_from_positions[i], ray_to_positions[i], [1, 0, 0], lineWidth=1, lifeTime=self.timestep * 4)

        # Add small gaussian noise to lidar during training for robustness
        if not self.render:
            dists = dists + np.random.normal(0.0, 0.01, size=dists.shape)
            dists = np.clip(dists, 0.0, self.lidar_max_range)

        return dists.astype(np.float32)

    def _get_obs(self):
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        lin_vel = np.array(lin_vel, dtype=np.float32)[:2]
        pos = np.array(pos, dtype=np.float32)[:2]
        lidar = self._perform_lidar()

        if self.target_pos is None:
            goal = np.array([0.0, 0.0], dtype=np.float32)
        else:
            goal = np.array([self.target_pos[0], self.target_pos[1]], dtype=np.float32)

        relative_goal = goal - pos  # Vector from drone to goal

        # include prev_action as part of observation
        prev_act = np.array(self.prev_action, dtype=np.float32)

        obs = np.concatenate([lin_vel, pos, lidar, relative_goal, prev_act], axis=0).astype(np.float32)
        return obs

    def _compute_forward_component(self, action_xy, relative_goal_xy, dist):
        """Return scalar projection of action onto goal direction (how much of the action is towards the goal)."""
        if dist < 1e-6:
            return 0.0
        goal_dir = np.array(relative_goal_xy, dtype=np.float32) / (dist + 1e-6)
        forward = float(np.dot(action_xy, goal_dir))
        return forward

    def compute_reward(self, distance, collision, action, progress, lidar):
        """Centralized reward computation for the single-drone environment.

        - Prioritizes reaching the goal and avoiding obstacles
        - Strong sparse reward for goal, strong negative for collision
        - Shaping: progress toward goal, proximity penalty to nearest obstacle, smoothness penalty
        """

        # Collision penalty (scale with speed so fast crashes are punished more)
        speed = float(np.linalg.norm(action))
        if collision:
            return -self.collision_base_penalty * (1.0 + speed )

        # Dense shaping terms
        reward = 0.0

        # Progress toward goal
        reward += self.progress_scale * progress

        # Small penalty for high speed to prefer safer, smoother moves
        reward -= self.speed_penalty * speed

        # Distance-based shaping (small penalty proportional to relative distance)
        r_dist = (distance / self.start_distance) * 0.1
        r_dist = min(r_dist, self.max_rdist)
        reward -= r_dist

        # Nearest-obstacle penalty (exponential, larger when very close)
        if lidar is not None and len(lidar) > 0:
            lidar_min = float(np.min(lidar))
            if lidar_min < self.obs_close_threshold:
                closeness = (self.obs_close_threshold - lidar_min) / max(self.obs_close_threshold, 1e-6)
                # quadratic soft penalty (0..1) scaled
                reward -= self.obs_penalty_weight * (closeness ** 2)

        # Smoothness penalty (change wrt previous action)
        act_change = float(np.linalg.norm(action - np.array(self.prev_action, dtype=np.float32)))
        reward -= self.smoothness_penalty_weight * act_change

        return reward

    def step(self, action):
        # Gymnasium step returns (obs, reward, terminated, truncated, info)
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -self.max_vel, self.max_vel)

        dist = self._distance_to_goal()

        # Smooth the action (exponential moving average)
        self.smoothed_action = self.action_smooth_alpha * action + (1.0 - self.action_smooth_alpha) * self.smoothed_action
        applied_action = self.smoothed_action.copy()

        # Set linear velocity (only x,y, keep hover z enforced separately)
        p.resetBaseVelocity(self.drone_id, linearVelocity=[float(applied_action[0]), float(applied_action[1]), 0.0])

        # Move moving blocks
        self._step_moving_blocks()

        # Step simulation
        p.stepSimulation()
        if self.render:
            time.sleep(self.timestep)

        obs = self._get_obs()
        self.elapsed_time += self.timestep

        ###################################
        #       Reward calculation 
        ###################################

        progress = self.prev_goal_dist - dist
        self.prev_goal_dist = dist

        # Extract lidar and relative goal from obs
        lidar = obs[4 : 4 + self.lidar_num_rays]
        relative_goal = obs[-4:-2]  # because prev_action appended at end

        # Collision detection (rough): check contact points
        contacts = p.getContactPoints()
        collision = False
        for c in contacts:
            if self.drone_id in (c[1], c[2]):  # c[1]=bodyA, c[2]=bodyB
                other_body = c[2] if c[1] == self.drone_id else c[1]
                if other_body != self.drone_id and other_body != self.floor_id:
                    collision = True
                    break

        # Compute forward component (how much of action points toward the goal)
        if dist > 0.0 and dist < float("inf"):
            goal_dir_xy = np.array(relative_goal, dtype=np.float32)
            goal_dist = float(np.linalg.norm(goal_dir_xy))
            if goal_dist > 1e-6:
                goal_unit = goal_dir_xy / goal_dist
            else:
                goal_unit = np.array([0.0, 0.0], dtype=np.float32)
            forward_comp = float(np.dot(np.array(applied_action, dtype=np.float32), goal_unit))
        else:
            forward_comp = 0.0

        # Base reward
        base_reward = self.compute_reward(dist, collision, applied_action, progress, lidar)

        # add explicit forward component (tunable)
        base_reward += self.forward_vel_weight * forward_comp

        reward = base_reward
        reason = ""
        
        done = False
        truncated = False
        info = {}

        # Too-far termination
        posrn = obs[2:4]
        if posrn[0] < -1 or posrn[0] > self.grid_size * self.cell_size + 1 or posrn[1] < -1 or posrn[1] > self.grid_size * self.cell_size + 1:
            done = True
            reason += " out_of_bounds"
            truncated = True
            reward -= 40.0 

        if collision:
            # keep the collision penalty in reward and mark done
            info["collision"] = True
            done = True
            reason += " collision"

        # Timeout
        if self.elapsed_time >= self.episode_timeout:
            truncated = True
            done = True
            info["timeout"] = True
            reason += " timeout"

        # Goal reached (within 0.1 meters)
        if dist < 0.5:
            reward += self.goal_reward/(self.counter + 20)
            self.counter +=1
            if dist < 0.2:
                reward += 120.0  
                done = True
                info["goal_reached"] = True
                reason += " goal_reached"
        else:
            info["goal_reached"] = False

        # Stuck penalty: if drone barely moves in position
        if self.prev_pos is not None:
            dist_moved = np.linalg.norm(np.array(obs[2:4]) - np.array(self.prev_pos))

            if dist_moved < self.min_movement_threshold:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        self.prev_pos = obs[2:4].copy()

        # Soft stuck penalty after warning steps
        if self.stuck_counter > self.stuck_warning_steps:
            reward -= self.stuck_penalty_scale * min(self.stuck_counter - self.stuck_warning_steps, self.max_stuck_penalty)
        # Terminate if stuck too long
        if self.stuck_counter > self.stuck_terminate_steps:
            reward -= self.max_stuck_penalty
            done = True
            info["stuck"] = True
            reason += " stuck"

        # Update prev_action stored in env for observations and smoothness computation
        self.prev_action = applied_action.copy()

        # Update curriculum stage success counters and possibly advance
        if done and self.curr_stage == 3:
            self.stage_episode_count += 1
            if info.get("goal_reached", False):
                self.stage_success_count += 1

            # check whether we should advance
            if self.stage_episode_count >= self.stage_min_episodes:
                success_rate = float(self.stage_success_count) / float(self.stage_episode_count)
                if success_rate >= 0.9 and self.curr_stage < 2:
                    self.curr_stage += 1
                    info["stage_level"] = self.curr_stage
                    # reset counters for new stage
                    self.stage_success_count = 0
                    self.stage_episode_count = 0

        # Add per-component logs for SAC training diagnostics
        info["distance_to_goal"] = dist

        info["Data"] = {

            "Distance to goal" : float(dist),
            "Reward" : float(reward),
        }

        info["Reason"] = reason

        if done:
            p.resetBaseVelocity(self.drone_id, linearVelocity=[0.0, 0.0, 0.0], angularVelocity=[0.0, 0.0, 0.0])
            self.smoothed_action = np.zeros(2, dtype=np.float32)
            self.prev_action = np.zeros(2, dtype=np.float32)
            self.counter = 0.0


        return obs, float(reward), done, truncated, info

    def set_camera_top_down(self):
        center_x = self.grid_size * 0.5
        center_y = self.grid_size * 0.5
        p.resetDebugVisualizerCamera(
            cameraDistance=self.grid_size * 1,
            cameraYaw=0,
            cameraPitch=-89.99,
            cameraTargetPosition=[center_x, center_y, 0]
        )

    def render(self):
        if self.render:
            pass  # PyBullet GUI renders automatically

    def close(self):
        p.disconnect()

def run():
    env = GridDroneEnv(render=True)
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Observation example:", obs)

    # Simple demonstration policy: proportional control towards goal with basic obstacle avoidance
    done = False
    while not done:
        # Extract from obs
        pos = obs[2:4]
        relative_goal = obs[-4:-2]
        lidar = obs[4:4 + env.lidar_num_rays]

        # Attraction to goal
        dist_to_goal = np.linalg.norm(relative_goal)
        if dist_to_goal > 0.05:
            attract_vel = (relative_goal / dist_to_goal) * env.max_vel * 0.5
        else:
            attract_vel = np.zeros(2)

        # Repulsion from obstacles (simple: push away from closest)
        min_dist_idx = np.argmin(lidar)
        min_dist = lidar[min_dist_idx]
        repel_vel = np.zeros(2)
        if min_dist < 1.0:
            # Direction away from obstacle (opposite of lidar ray direction)
            angle = -math.pi + (min_dist_idx / env.lidar_num_rays) * 2 * math.pi
            repel_dir = np.array([math.cos(angle), math.sin(angle)])
            repel_strength = -(1.0 - min_dist) * env.max_vel * 0.7
            repel_vel = repel_dir * repel_strength

        # Combine
        action = attract_vel + repel_vel
        action = np.clip(action, -env.max_vel, env.max_vel)

        obs, reward, done, truncated, info = env.step(action)
        print(f"Step: dist={info['distance_to_goal']:.2f}, reward={reward:.2f}, done={done}")

    env.close()

if __name__ == "__main__":

    run()

