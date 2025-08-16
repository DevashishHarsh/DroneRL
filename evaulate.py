"""
test_sac.py
Load a trained SAC model and run it in GridDroneEnv with GUI rendering.
"""

import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from main import GridDroneEnv


def main():
    MODEL_PATH = "models/drone_model.zip"  # Path to saved model
    VECNORM_PATH = "models/drone_vec_normalize.pkl"  # Path to saved VecNormalize

    # Create a single GUI environment
    def make_worker():
        env = GridDroneEnv(
            grid_size=5,
            render=True,  # Enable GUI for testing
            drone_urdf_path="drone.urdf",
            lidar_num_rays=50,
            lidar_max_range=8.0,
            max_vel=3,
            hover_z=1,
            timestep=1.0 / 60.0,
            episode_timeout=20,
            draw_lidar=False,
            action_smooth_alpha=0.2,
        )
        return env

    # Wrap in DummyVecEnv and load normalization
    env = DummyVecEnv([make_worker])
    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = False   # important — no stat updates
    env.norm_reward = False  # optional: keep rewards unnormalized for viewing

    print("[test] Loading trained SAC model...")
    model = SAC.load(MODEL_PATH, env=env, device="cuda")

    print("[test] Running test episodes...")
    num_episodes = 5
    for ep in range(num_episodes):
        obs= env.reset()
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            done = terminated 
            ep_reward += reward
            steps += 1
            time.sleep(1.0 / 30.0)  # Slow down for human viewing
            print(f"[test] Episode {ep}, Step {steps}, Reward: {ep_reward} and Distance to Goal {info[0]['distance_to_goal']}")

    env.close()
    print("[test] Done.")


if __name__ == "__main__":
    main()

