import streamlit as st

# Page config
st.set_page_config(
    page_title="Missile RL System",
    layout="wide"
)

# Title
st.title("🚀 Autonomous Missile Guidance System (RL)")

# Sidebar Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Simulation",
        "Training Analysis",
        "3D Visualization",
        "Advanced Dashboard"   # ✅ NEW PAGE ADDED
    ]
)

# ===============================
# PAGE 1 — SIMULATION
# ===============================
if page == "Simulation":
    from utils import run_simulation, plot_trajectory

    st.header("Simulation Dashboard")

    model_path = st.text_input(
        "Model Path",
        "missile_ppo_v4.zip"
    )

    steps = st.slider(
        "Simulation Steps",
        100,
        2000,
        500
    )

    if st.button("Run Simulation"):
        with st.spinner("Running..."):
            missile, target = run_simulation(model_path, steps)

        st.success("Simulation Completed")

        fig = plot_trajectory(missile, target)
        st.pyplot(fig)

        # Metrics
        st.subheader("Performance Metrics")

        final_distance = ((missile[-1] - target[-1])**2).sum()**0.5

        col1, col2 = st.columns(2)
        col1.metric("Final Distance", f"{final_distance:.2f}")
        col2.metric("Steps", len(missile))

        if final_distance < 50:
            st.success("🎯 Target Intercepted")
        else:
            st.warning("❌ Target Not Intercepted")


# ===============================
# PAGE 2 — TRAINING ANALYSIS
# ===============================
elif page == "Training Analysis":
    from training_viz import show_training_plots

    st.header("Training Results Analysis")

    log_file = st.text_input(
        "Training Log File (CSV)",
        "logs/training_log.csv"
    )

    if st.button("Load Training Results"):
        show_training_plots(log_file)


# ===============================
# PAGE 3 — 3D VISUALIZATION
# ===============================
elif page == "3D Visualization":
    from visualize_streamlit import run_3d_visualization

    st.header("3D Missile-Target Visualization")

    model_path = st.text_input(
        "Model Path",
        "missile_ppo_v4.zip"
    )

    if st.button("Run 3D Visualization"):
        run_3d_visualization(model_path)


# ===============================
# PAGE 4 — ADVANCED DASHBOARD
# ===============================
elif page == "Advanced Dashboard":
    from advanced_dashboard import run_advanced_dashboard

    st.header("Advanced RL Dashboard")

    run_advanced_dashboard()