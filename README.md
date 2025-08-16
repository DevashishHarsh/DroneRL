# DroneRL

A compact research/demo codebase that trains a 2D drone agent inside a PyBullet grid environment using Soft Actor-Critic (SAC) from Stable-Baselines3. The project contains a Gymnasium-compatible environment (GridDroneEnv), a headless trainer for scalable training (train.py), and a lightweight PyGame UI (trainer.py) for interactive runs and monitoring.

## Overview

This repository is intended as a reproducible example and starting point for research or experimentation in continuous-control navigation tasks. The main goals are:

Provide a simple but realistic 2D simulated drone environment with static and moving obstacles.

Demonstrate training with SAC using parallelized environments (SubprocVecEnv + VecNormalize).

Offer a small UI for quick experiments and live visualization.

The environment is intentionally compact so it is easy to modify reward shaping, observation contents, or asset geometry.

## What’s included

main.py — GridDroneEnv (Gymnasium env). Implements LiDAR, moving/static obstacles, spawn/target placement, reward logic, and termination conditions.

train.py — headless training script. Builds vectorized envs, configures SAC, includes a lightweight callback for live stats, saves model and VecNormalize stats, and writes TensorBoard logs to sac_drone_tb/.

trainer.py — simple PyGame-based UI for launching training, visual preview (PyBullet snapshot), progress logging, and TensorBoard launcher.

models/ — directory where trained artifacts are saved (drone_model.zip, drone_vec_normalize.pkl).

meshes/ and *.urdf — URDF and mesh files for drone and obstacles. Code falls back to primitives if URDFs fail to load.

requirements.txt — dependencies.

## Quickstart

### Install

Create a virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

If you plan to use GPU, install a torch wheel matching your CUDA toolchain before installing the rest.

### Headless training

Run the headless trainer to train with multiple parallel workers:
```
python train.py
```
By default the script saves the trained model under models/drone_model.zip and VecNormalize stats to models/drone_vec_normalize.pkl.

### UI trainer

Run the interactive frontend to preview and launch training with a few clicks:
```
python trainer.py
```
The UI allows choosing grid size, number of envs, lidar rays, stage (curriculum), and training timesteps.

### Monitor with TensorBoard
```
tensorboard --logdir sac_drone_tb/ 
```
> Open http://localhost:6006 to inspect reward, losses, and other logged scalars.

> You can also use the UI to open tensorboard data. Just note down the training number and enter it in the field.

## Curriculum stages

The environment supports three curriculum stages (configurable in the trainer):

- Easy : No obstacles (start + target only) — used to bootstrap navigation skills.

- Medium : Light obstacles (sparse walls/slabs) — intermediate difficulty.

- Hard: Normal obstacles (default sampling weights) — full difficulty.

- Auto: Configured to advance stages automatically when the success rate in the current stage exceeds a threshold (e.g., 90%).

## Training notes

Uses Stable-Baselines3 SAC with an MLP policy. Default network sizes and hyperparameters are chosen for stability but can be tuned in train.py.

VecNormalize is used to normalize observations and rewards during training. Save and reload the VecNormalize stats when doing evaluation to preserve normalization.

When running many parallel environments, prefer smaller train_freq and controlled gradient_steps to avoid overly aggressive updates.

## Evaluation / inference

To evaluate a saved model:
```
python evaluate.py
```
## Where to change things

- main.py — change observation composition, LiDAR configuration, reward terms, or URDF paths.

- train.py — change SAC hyperparameters, number of workers, schedule for saving/loading.

- trainer.py — UI defaults and layout, or remove thread-kill logic if you prefer process-based control.

### Side Note
- Floating blocks just depicts the moving slabs actually. 
- I have also added vertical slabs for the training even though they are not used at the moment. You can change the lidar configuration to actually use them and add them to your project!

## License

This repository is provided under the MIT License.
