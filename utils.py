import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from missile_chase_target_RL import MissileEnv


# ===============================
# RUN SIMULATION
# ===============================
def run_simulation(model_path, steps=500):
    env = MissileEnv()
    model = PPO.load(model_path)

    obs, _ = env.reset()

    missile_positions = []
    target_positions = []

    for _ in range(steps):
        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 🔥 SAFE POSITION EXTRACTION (NO ERRORS)
        # Assumption: obs = [mx, my, tx, ty, ...]
        missile_pos = obs[0:2]
        target_pos = obs[2:4]

        missile_positions.append(missile_pos)
        target_positions.append(target_pos)

        if done:
            break

    return np.array(missile_positions), np.array(target_positions)


# ===============================
# PLOT TRAJECTORY (2D)
# ===============================
def plot_trajectory(missile, target):
    fig, ax = plt.subplots()

    ax.plot(missile[:, 0], missile[:, 1], color='red', label="Missile")
    ax.plot(target[:, 0], target[:, 1], color='blue', label="Target")

    ax.scatter(missile[0, 0], missile[0, 1], color='green', label="Missile Start")
    ax.scatter(target[0, 0], target[0, 1], color='orange', label="Target Start")

    ax.set_title("Missile vs Target Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    ax.legend()
    ax.grid(True)

    return fig