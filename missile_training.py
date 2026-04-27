from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from missile_chase_target_RL import MissileEnv
import numpy as np
import pandas as pd
import os

# OPTIONAL: wandb (safe import)
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    use_wandb = True
except:
    use_wandb = False


# ===============================
# CREATE LOG DIRECTORY
# ===============================
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints_v4", exist_ok=True)


# ===============================
# CUSTOM CALLBACK FOR LOGGING
# ===============================
class TrainingLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.log_data = []

    def _on_step(self) -> bool:
     if len(self.model.ep_info_buffer) > 0:
        ep_info = self.model.ep_info_buffer[-1]

        log_entry = {
            "timesteps": self.num_timesteps,
            "ep_rew_mean": ep_info.get("r", 0),
            "ep_len_mean": ep_info.get("l", 0),
        }

        self.log_data.append(log_entry)

        # 🔥 SAVE LIVE (IMPORTANT)
        import pandas as pd
        pd.DataFrame(self.log_data).to_csv("logs/training_log.csv", index=False)

     return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.log_data)
        df.to_csv("logs/training_log.csv", index=False)
        print("Training log saved to logs/training_log.csv")


# ===============================
# INIT WANDB (OPTIONAL)
# ===============================
if use_wandb:
    wandb.init(
        project="missile_guidance",
        name="v4_perturbations",
        config={
            "total_timesteps": 600_000,
            "ent_coef": 0.01,
            "learning_rate": 3e-4
        }
    )


# ===============================
# CREATE ENVIRONMENT
# ===============================
env = MissileEnv()


# ===============================
# CREATE MODEL
# ===============================
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    ent_coef=0.01
)


# ===============================
# CALLBACKS
# ===============================
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints_v4/",
    name_prefix="missile_ppo_v4"
)

logger_callback = TrainingLogger()

callbacks = [checkpoint_callback, logger_callback]

if use_wandb:
    callbacks.append(WandbCallback())


# ===============================
# TRAIN MODEL
# ===============================
model.learn(
    total_timesteps=600_000,
    callback=callbacks,
    reset_num_timesteps=False
)


# ===============================
# SAVE MODEL
# ===============================
model.save("missile_ppo_v4")

if use_wandb:
    wandb.finish()

print("Training completed successfully")