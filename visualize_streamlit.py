import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from missile_chase_target_RL import MissileEnv


# ===============================
# RUN SIMULATION (SAFE)
# ===============================
def run_simulation_3d(model_path, steps=500):
    env = MissileEnv()
    model = PPO.load(model_path)

    obs, _ = env.reset()

    missile_positions = []
    target_positions = []

    for step in range(steps):
        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        # ✅ SAFE EXTRACTION
        try:
            missile_pos = np.array(obs[0:2], dtype=np.float64)
            target_pos = np.array(obs[2:4], dtype=np.float64)
        except:
            continue

        # ✅ HANDLE INVALID VALUES
        if (
            missile_pos is None
            or target_pos is None
            or np.any(np.isnan(missile_pos))
            or np.any(np.isnan(target_pos))
        ):
            continue

        # ✅ STORE WITH TIME (Z axis)
        missile_positions.append([
            float(missile_pos[0]),
            float(missile_pos[1]),
            float(step)
        ])

        target_positions.append([
            float(target_pos[0]),
            float(target_pos[1]),
            float(step)
        ])

        if terminated or truncated:
            break

    # Convert safely
    missile_positions = np.array(missile_positions)
    target_positions = np.array(target_positions)

    return missile_positions, target_positions


# ===============================
# PLOT 3D TRAJECTORY
# ===============================
def plot_3d(missile, target):

    # 🔥 FINAL SAFETY CLEANING
    if len(missile) == 0 or len(target) == 0:
        return None

    mask = ~np.isnan(missile).any(axis=1)
    missile = missile[mask]
    target = target[mask]

    if len(missile) == 0 or len(target) == 0:
        return None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Trajectories
    ax.plot(
        missile[:, 0],
        missile[:, 1],
        missile[:, 2],
        label="Missile",
        color='red'
    )

    ax.plot(
        target[:, 0],
        target[:, 1],
        target[:, 2],
        label="Target",
        color='blue'
    )

    # Start points
    ax.scatter(
        missile[0, 0],
        missile[0, 1],
        missile[0, 2],
        color='green',
        label="Missile Start"
    )

    ax.scatter(
        target[0, 0],
        target[0, 1],
        target[0, 2],
        color='orange',
        label="Target Start"
    )

    ax.set_title("3D Missile vs Target Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time")

    ax.legend()

    return fig


# ===============================
# MAIN STREAMLIT FUNCTION
# ===============================
def run_3d_visualization(model_path):

    st.info("Running 3D Simulation...")

    missile, target = run_simulation_3d(model_path)

    if missile is None or len(missile) == 0:
        st.error("No valid simulation data. Try retraining the model.")
        return

    st.success("Simulation Complete")

    fig = plot_3d(missile, target)

    if fig is None:
        st.error("Visualization failed due to invalid data.")
        return

    st.pyplot(fig)

    # ===============================
    # METRICS
    # ===============================
    final_distance = np.linalg.norm(missile[-1][:2] - target[-1][:2])

    st.subheader("Performance Metrics")

    col1, col2 = st.columns(2)
    col1.metric("Final Distance", f"{final_distance:.2f}")
    col2.metric("Total Steps", len(missile))

    if final_distance < 50:
        st.success("🎯 Target Intercepted")
    else:
        st.warning("❌ Target Not Intercepted")