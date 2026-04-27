import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os


# ===============================
# LOAD DATA
# ===============================
def load_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        return None


# ===============================
# PLOT FUNCTION
# ===============================
def plot_graph(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig


# ===============================
# MAIN FUNCTION
# ===============================
def show_training_plots(file_path):

    df = load_data(file_path)

    st.subheader("Training Performance Graphs")

    # --------------------------------
    # CASE 1: CSV EXISTS
    # --------------------------------
    if df is not None:

        st.success("Training log loaded")

        # Show raw data
        st.subheader("Raw Training Data")
        st.dataframe(df.head())

        # Try common column names
        columns = df.columns

        # Reward
        if "ep_rew_mean" in columns:
            fig = plot_graph(df.index, df["ep_rew_mean"],
                             "Reward vs Time", "Steps", "Reward")
            st.pyplot(fig)

        # Episode Length
        if "ep_len_mean" in columns:
            fig = plot_graph(df.index, df["ep_len_mean"],
                             "Episode Length vs Time", "Steps", "Length")
            st.pyplot(fig)

        # Loss
        if "loss" in columns:
            fig = plot_graph(df.index, df["loss"],
                             "Loss vs Time", "Steps", "Loss")
            st.pyplot(fig)

        # KL Divergence
        if "approx_kl" in columns:
            fig = plot_graph(df.index, df["approx_kl"],
                             "KL Divergence vs Time", "Steps", "KL")
            st.pyplot(fig)

    # --------------------------------
    # CASE 2: NO CSV → DUMMY DATA
    # --------------------------------
    else:
        st.warning("No training log found → showing sample graphs")

        import numpy as np

        steps = np.arange(100)

        reward = np.log(steps + 1) * 50 - 200
        length = steps * 5
        loss = np.exp(-steps / 20) * 100
        kl = np.random.random(100) * 0.01

        st.pyplot(plot_graph(steps, reward, "Reward vs Time", "Steps", "Reward"))
        st.pyplot(plot_graph(steps, length, "Episode Length", "Steps", "Length"))
        st.pyplot(plot_graph(steps, loss, "Loss Curve", "Steps", "Loss"))
        st.pyplot(plot_graph(steps, kl, "KL Divergence", "Steps", "KL"))