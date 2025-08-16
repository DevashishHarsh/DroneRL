
import time
from multiprocessing import freeze_support

# Module-level handle for same-process readers (will be set to TrainingStats instance)
TRAINING_STATS = None


def train(envs, totalsteps, lidar_rays, refresh_rate, stage, stats_proxy=None, log_queue=None):
    """
    Run SAC training.
    Parameters:
      envs (int): number of parallel envs
      totalsteps (int or float): total timesteps to train
      lidar_rays (int)
      refresh_rate (int)
      stage (int)
      stats_proxy: optional multiprocessing.Manager().Namespace for cross-process stats
      log_queue: optional multiprocessing.Queue for cross-process log messages
    """
    # local imports (keeps module import cheap if not used)
    import threading
    import numpy as np
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from main import GridDroneEnv

    global TRAINING_STATS

    # ---------------- FILES / CONFIG ----------------
    MODEL_PATH = "models/drone_model.zip"
    VECNORM_PATH = "models/drone_vec_normalize.pkl"

    RENDER = False
    N_ENVS = envs
    TOTAL_TIMESTEPS = totalsteps
    LIDAR_NUM_RAYS = lidar_rays
    LIDAR_RANGE = 8.0
    HOVER_Z = 0.5
    MAX_VEL = 3
    EPISODE_TIMEOUT = 20
    TIMESTEP = 1.0 / refresh_rate
    ALPHA = 0.2
    PRINT_EVERY = 1
    STAGE = stage

    def _maybe_log(s):
        # print to stdout for convenience, and try to send to UI via log_queue
        try:
            print(s, flush=True)
        except Exception:
            pass
        if log_queue is not None:
            try:
                # non-blocking push
                log_queue.put_nowait(str(s))
            except Exception:
                # if queue is Full or closed, ignore
                pass

    def make_worker(render_env: bool):
        def _init():
            # worker-local pybullet disconnects to avoid leftover connections
            try:
                import pybullet as p
                for cid in range(10):
                    try:
                        p.disconnect(cid)
                    except Exception:
                        pass
            except Exception:
                pass

            env = GridDroneEnv(
                grid_size=5,
                render=render_env,
                drone_urdf_path="drone.urdf",
                lidar_num_rays=LIDAR_NUM_RAYS,
                lidar_max_range=LIDAR_RANGE,
                max_vel=MAX_VEL,
                hover_z=HOVER_Z,
                timestep=TIMESTEP,
                episode_timeout=EPISODE_TIMEOUT,
                draw_lidar=False,
                action_smooth_alpha=ALPHA,
                stage=STAGE
            )
            env = Monitor(env)
            return env
        return _init

    if RENDER:
        _maybe_log("[trainer] Running in GUI mode with a single environment (RENDER=True).")
        vec_env = DummyVecEnv([make_worker(True)])
    else:
        _maybe_log("[trainer] Running in headless mode with SubprocVecEnv (RENDER=False).")
        env_fns = [make_worker(False) for _ in range(N_ENVS)]
        vec_env = SubprocVecEnv(env_fns)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # device selection - try cuda then fallback to cpu if unavailable
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _maybe_log(f"[trainer] Using device: {device}")

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        tensorboard_log="./sac_drone_tb",
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        train_freq=32,
        gradient_steps=64,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )

    # ---------------- Shared stats container ----------------
    class TrainingStats:
        def __init__(self):
            self.lock = threading.Lock()
            self.goals = 0
            self.collision = 0
            self.stage = STAGE
            self.num_steps = 0
            self.last_data = None
            self.last_reason = None
            self.distances = []

        def snapshot(self):
            with self.lock:
                return {
                    "goals": int(self.goals),
                    "collision": int(self.collision),
                    "stage": int(self.stage),
                    "num_steps": int(self.num_steps),
                    "last_data": self.last_data,
                    "last_reason": self.last_reason,
                    "distances_len": len(self.distances),
                }

    # Callback that updates the in-process stats, writes to stats_proxy and pushes logs to log_queue.
    class InfoPrintCallback(BaseCallback):
        def __init__(self, stats_obj, print_every=PRINT_EVERY, stats_proxy_local=None, log_queue_local=None):
            super().__init__(verbose=0)
            self.print_every = print_every
            self.stats = stats_obj
            self.stats_proxy = stats_proxy_local
            self.log_queue = log_queue_local

        def _maybe_push_log(self, s):
            # local print + queue push
            _maybe_log(s)

        def _on_step(self) -> bool:
            # called frequently; keep it lightweight
            do_log = False
            with self.stats.lock:
                self.stats.num_steps += 1
                if (self.stats.num_steps % self.print_every) == 0:
                    do_log = True
                current_steps = self.stats.num_steps

            # gather infos if available
            infos = self.locals.get("infos", None)
            if infos and len(infos) > 0:
                for info in infos:
                    if not info:
                        continue
                    with self.stats.lock:
                        if "distance_to_goal" in info:
                            try:
                                self.stats.distances.append(info["distance_to_goal"])
                            except Exception:
                                pass
                        if info.get("goal_reached", False):
                            self.stats.goals += 1
                        if info.get("collision", False):
                            self.stats.collision += 1
                        if "stage_level" in info:
                            self.stats.stage = info["stage_level"]
                        # copy data/reason
                        self.stats.last_data = info.get("Data")
                        self.stats.last_reason = info.get("Reason")

            # propagate minimal fields to stats_proxy so UI can poll across processes
            if self.stats_proxy is not None:
                try:
                    # update fields on proxy (wrapped in try to avoid crashing training)
                    self.stats_proxy.num_steps = int(current_steps)
                    self.stats_proxy.goals = int(self.stats.goals)
                    self.stats_proxy.collision = int(self.stats.collision)
                    self.stats_proxy.stage = int(self.stats.stage)
                    self.stats_proxy.last_data = self.stats.last_data
                    self.stats_proxy.last_reason = self.stats.last_reason
                    self.stats_proxy.distances_len = len(self.stats.distances)
                except Exception:
                    pass

                # cooperative stop check - if UI set request_stop True, return False to stop learning
                try:
                    if getattr(self.stats_proxy, "request_stop", False):
                        self._maybe_push_log("[trainer callback] request_stop detected via stats_proxy -> stopping training gracefully")
                        return False
                except Exception:
                    # if proxy access fails, ignore
                    pass

            # occasionally push human-readable logs via queue
            if do_log:
                try:
                    s = f"[trainer LIVE] steps={current_steps} goals={self.stats.goals} collisions={self.stats.collision} stage={self.stats.stage} last_data={self.stats.last_data}"
                    self._maybe_push_log(s)
                except Exception:
                    pass

            # continue training
            return True

    # Create and expose stats object locally
    stats = TrainingStats()
    TRAINING_STATS = stats

    callback = InfoPrintCallback(stats_obj=stats, print_every=PRINT_EVERY, stats_proxy_local=stats_proxy, log_queue_local=log_queue)

    try:
        _maybe_log("[trainer] Starting SAC training. Close GUI or set stats_proxy.request_stop to stop.")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    except KeyboardInterrupt:
        _maybe_log("[trainer] Interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        # push exception to UI/logs and re-raise to surface the error
        _maybe_log(f"[trainer] Exception during training: {e}")
        raise
    finally:
        _maybe_log("[trainer] Cleaning up trainer (closing envs and saving model)...")

        try:
            model.save(MODEL_PATH)
            _maybe_log(f"[trainer] Saved model to {MODEL_PATH}")
        except Exception as e:
            _maybe_log(f"[trainer] Failed saving model: {e}")

        try:
            # save VecNormalize stats if used
            if isinstance(vec_env, VecNormalize):
                vec_env.save(VECNORM_PATH)
                _maybe_log(f"[trainer] Saved VecNormalize stats to {VECNORM_PATH}")
        except Exception as e:
            _maybe_log(f"[trainer] Failed to save VecNormalize: {e}")

        try:
            vec_env.close()
        except Exception:
            pass

    return stats


if __name__ == "__main__":
    freeze_support()
    # Example call for local debugging (no cross-process proxy/queue)
    train(envs=2, totalsteps=200000, lidar_rays=36, refresh_rate=30, stage=1)
