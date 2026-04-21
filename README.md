# RL_Mini_Project_D16AD-B_60
# 🚀 Autonomous Missile Guidance using Reinforcement Learning

## 📌 Overview
This project focuses on developing an intelligent missile guidance system using Deep Reinforcement Learning (DRL). The goal is to train an agent that can intercept a moving target in a simulated environment by learning optimal control strategies.

Unlike traditional guidance methods, this approach enables adaptive decision-making in dynamic and unpredictable scenarios.

---

## 🧠 Key Features
- Reinforcement Learning-based missile control
- Continuous action space for realistic movement
- PPO (Proximal Policy Optimization) algorithm
- Simulated missile–target engagement environment
- Visualization of interception trajectory
- Training monitoring using Weights & Biases (wandb)

---

## ⚙️ Technologies Used
- Python
- Stable-Baselines3 (PPO)
- OpenAI Gym (Environment)
- PyTorch
- Matplotlib
- Weights & Biases (wandb)

---

## 📂 Project Structure
missile_guidance_RL/
│── missile_training.py # Training script
│── missile_chase_target_RL.py # Environment (Missile + Target logic)
│── visualize.py # Visualization of trained model
│── checkpoints_v4/ # Saved models (optional)
│── wandb/ # Training logs

---

## 📊 Output
- Training logs displayed in terminal
- Reward and performance graphs (wandb)
- Missile vs target trajectory visualization

---

## 🎯 Reinforcement Learning Setup

### State Space
- Relative distance
- Velocity components
- Line-of-sight angle
- Missile heading and speed

### Action Space
- Continuous control (acceleration / turn rate)

### Reward Function
- Positive reward for interception
- Negative reward for distance
- Penalty for excessive control effort
- Failure penalty if target not intercepted

---

## 📈 Algorithm Used
- Proximal Policy Optimization (PPO)

---

## 📌 Applications
- Autonomous missile guidance systems
- UAV interception strategies
- Defense simulations
- Robotics path planning

---

## ⚠️ Note
This project is for academic and simulation purposes only.

---

## 👨‍💻 Author
Kushal Yadav
