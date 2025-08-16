DroneRL

A compact research/demo codebase that trains a 2D drone agent inside a PyBullet grid environment using Soft Actor-Critic (SAC) from Stable-Baselines3. The project contains a Gymnasium-compatible environment (GridDroneEnv), a headless trainer for scalable training (train.py), and a lightweight PyGame UI (trainer.py) for interactive runs and monitoring.

1. Overview

This repository is intended as a reproducible example and starting point for research or experimentation in continuous-control navigation tasks. The main goals are:

Provide a simple but realistic 2D simulated drone environment with static and moving obstacles.

Demonstrate training with SAC using parallelized environments (SubprocVecEnv + VecNormalize).

Offer a small UI for quick experiments and live visualization.

The environment is intentionally compact so it is easy to modify reward shaping, observation contents, or asset geometry.

2. What’s included

main.py — GridDroneEnv (Gymnasium env). Implements LiDAR, moving/static obstacles, spawn/target placement, reward logic, and termination conditions.

train.py — headless training script. Builds vectorized envs, configures SAC, includes a lightweight callback for live stats, saves model and VecNormalize stats, and writes TensorBoard logs to sac_drone_tb/.

trainer.py — simple PyGame-based UI for launching training, visual preview (PyBullet snapshot), progress logging, and TensorBoard launcher.

models/ — directory where trained artifacts are saved (drone_model.zip, drone_vec_normalize.pkl).

meshes/ and *.urdf — URDF and mesh files for drone and obstacles. Code falls back to primitives if URDFs fail to load.

requirements.txt — dependencies.

3. Quickstart

Install

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

If you plan to use GPU, install a torch wheel matching your CUDA toolchain before installing the rest.

Headless training

Run the headless trainer to train with multiple parallel workers:

python train.py

By default the script saves the trained model under models/drone_model.zip and VecNormalize stats to models/drone_vec_normalize.pkl.

UI trainer

Run the interactive frontend to preview and launch training with a few clicks:

python trainer.py

The UI allows choosing grid size, number of envs, lidar rays, stage (curriculum), and training timesteps.

Monitor with TensorBoard

tensorboard --logdir sac_drone_tb/ 

Open http://localhost:6006 to inspect reward, losses, and other logged scalars.

You can also use the UI to open tensorboard data. Just note down the training number and enter it in the field.



4. Environment (GridDroneEnv) — concise description

Action space: 2D continuous velocity command Box(-max_vel, max_vel, (2,)) interpreted as planar desired linear velocity (vx, vy).

Observation: concatenation of [vx, vy, px, py, lidar_distances (N), goal_dx, goal_dy, prev_action_x, prev_action_y].

Sensors: 2D horizontal LiDAR implemented via p.rayTestBatch with configurable number of rays and max range.

Obstacles: static L-shaped and square walls, and moving slabs (horizontal/vertical). Stage controls obstacle density.

Termination: goal reached, collision, timeout, out-of-bounds, or stuck (no meaningful position change for a number of steps).

Reset: spawns drone at spawn_pos, teleports to hover height, and zeros base velocity to avoid momentum carryover.

5. Curriculum stages

The environment supports three curriculum stages (configurable in the trainer):

Stage 0: No obstacles (start + target only) — used to bootstrap navigation skills.

Stage 1: Light obstacles (sparse walls/slabs) — intermediate difficulty.

Stage 2: Normal obstacles (default sampling weights) — full difficulty.

The trainer can be configured to advance stages automatically when the success rate in the current stage exceeds a threshold (e.g., 90%).

6. Reward design (summary)

Reward is intentionally mixed (dense + sparse) to speed up learning while ensuring the agent values reaching the goal and avoiding collisions:

Dense shaping: progress toward goal (previous distance − current distance) scaled by progress_scale.

Sparse: a large terminal reward when the agent docks within a small radius of the target.

Collision: significant negative penalty on contact, scaled by impact speed.

Proximity: a smooth penalty based on the nearest LiDAR reading if too close to obstacles.

Smoothness: small penalty for sudden action changes to encourage stable trajectories.

You can tune these constants in GridDroneEnv.__init__.

7. Training notes

Uses Stable-Baselines3 SAC with an MLP policy. Default network sizes and hyperparameters are chosen for stability but can be tuned in train.py.

VecNormalize is used to normalize observations and rewards during training. Save and reload the VecNormalize stats when doing evaluation to preserve normalization.

When running many parallel environments, prefer smaller train_freq and controlled gradient_steps to avoid overly aggressive updates.

8. Evaluation / inference

To evaluate a saved model:

Load VecNormalize (if used) and the model.

Run deterministic policy (model.predict(obs, deterministic=True)) and step environment. Use rendering to inspect behavior.

A simple evaluation script is included in the repository as an example in train.py/main.py.

9. Tips & common pitfalls

Momentum carryover: always reset base velocity (p.resetBaseVelocity) on environment reset or terminal events to avoid the drone flying off at episode start.

Reward clipping: prefer leaving rewards unclipped in the env and rely on VecNormalize to stabilize magnitudes.

Curriculum: advance stages only once the agent consistently reaches the goal in the easier stage. Monitor success rate with deterministic evals.

Thread vs process: the provided UI runs the trainer in a background thread and reads a module-level TRAINING_STATS. For robust separation, run training in a separate process and use a multiprocessing proxy for stats/logs.

10. Where to change things

main.py — change observation composition, LiDAR configuration, reward terms, or URDF paths.

train.py — change SAC hyperparameters, number of workers, schedule for saving/loading.

trainer.py — UI defaults and layout, or remove thread-kill logic if you prefer process-based control.

11. License

This repository is provided under the MIT License.
