import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from missile_chase_target_RL import MissileEnv


# ===============================
# RUN STEP-BY-STEP SIMULATION
# ===============================
def run_live_simulation(model_path, steps=200, delay=0.05):
    env = MissileEnv()
    model = PPO.load(model_path)

    obs, _ = env.reset()

    missile_positions = []
    target_positions = []

    placeholder = st.empty()
    metric_placeholder = st.empty()

    for step in range(steps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        missile = obs[0:2]
        target = obs[2:4]

        missile_positions.append(missile)
        target_positions.append(target)

        # Plot update
        fig, ax = plt.subplots()
        m = np.array(missile_positions)
        t = np.array(target_positions)

        ax.plot(m[:, 0], m[:, 1], color='red')
        ax.plot(t[:, 0], t[:, 1], color='blue')

        ax.scatter(m[-1, 0], m[-1, 1], color='red')
        ax.scatter(t[-1, 0], t[-1, 1], color='blue')

        ax.set_title(f"Step {step}")
        ax.grid(True)

        placeholder.pyplot(fig)

        # Live metric
        dist = np.linalg.norm(m[-1] - t[-1])
        metric_placeholder.metric("Live Distance", f"{dist:.2f}")

        time.sleep(delay)

        if terminated or truncated:
            break


# ===============================
# MODEL COMPARISON
# ===============================
def compare_models(model1, model2):
    st.subheader("Model Comparison")

    env = MissileEnv()

    obs1, _ = env.reset()
    obs2, _ = env.reset()

    model_a = PPO.load(model1)
    model_b = PPO.load(model2)

    dist_a = []
    dist_b = []

    for _ in range(200):
        action1, _ = model_a.predict(obs1)
        action2, _ = model_b.predict(obs2)

        obs1, _, t1, tr1, _ = env.step(action1)
        obs2, _, t2, tr2, _ = env.step(action2)

        m1 = obs1[0:2]
        t1p = obs1[2:4]

        m2 = obs2[0:2]
        t2p = obs2[2:4]

        dist_a.append(np.linalg.norm(m1 - t1p))
        dist_b.append(np.linalg.norm(m2 - t2p))

    fig, ax = plt.subplots()
    ax.plot(dist_a, label="Model 1")
    ax.plot(dist_b, label="Model 2")
    ax.set_title("Distance Comparison")
    ax.legend()

    st.pyplot(fig)


# ===============================
# MAIN UI FUNCTION
# ===============================
def run_advanced_dashboard():
    st.header("Advanced RL Dashboard")

    tab1, tab2 = st.tabs(["Live Simulation", "Model Comparison"])

    # -------------------------------
    # LIVE SIMULATION
    # -------------------------------
    with tab1:
        model_path = st.text_input("Model Path", "missile_ppo_v4.zip")
        steps = st.slider("Steps", 50, 500, 200)

        if st.button("Start Live Simulation"):
            run_live_simulation(model_path, steps)

    # -------------------------------
    # MODEL COMPARISON
    # -------------------------------
    with tab2:
        model1 = st.text_input("Model 1", "missile_ppo_v4.zip")
        model2 = st.text_input("Model 2", "missile_ppo_v4.zip")

        if st.button("Compare Models"):
            compare_models(model1, model2)