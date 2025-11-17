import os
import sys

# Make sure project root is on sys.path so "import safetraj" works
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from safetraj import SafeTrajEvaluator, SafeTrajConfig
from safetraj import plot_trajectory, plot_feature_importance

# ----- page setup -----
st.set_page_config(page_title="SafeTraj-X", layout="wide")

st.title("SafeTraj-X – Safety & OOD Explorer")

st.write(
    "This demo shows how simple motion commands map to predicted trajectories, "
    "OOD scores, and a combined risk estimate."
)

# single evaluator instance (fixed seed => stable behaviour)
config = SafeTrajConfig(seed=42)
evaluator = SafeTrajEvaluator(config=config)

# ----- sidebar inputs -----
st.sidebar.header("Input parameters")

mode = st.sidebar.radio(
    "Input mode",
    ["Typical range", "Custom (no limits)"],
)

if mode == "Typical range":
    # inputs restricted to the same range used for training data
    orientation = st.sidebar.slider("Orientation (rad)", -3.14, 3.14, 0.0, 0.1)
    v_lin = st.sidebar.slider("Linear velocity (m/s)", -1.05, 2.88, 0.5, 0.05)
    v_rot = st.sidebar.slider("Rotational velocity (rad/s)", -1.99, 1.99, 0.2, 0.05)
else:
    # fully free numeric inputs (can go outside the training range)
    orientation = st.sidebar.number_input("Orientation (rad)", value=0.0)
    v_lin = st.sidebar.number_input("Linear velocity (m/s)", value=0.5)
    v_rot = st.sidebar.number_input("Rotational velocity (rad/s)", value=0.2)

run = st.sidebar.button("Evaluate")

# ----- main logic -----
if run:
    x = [orientation, v_lin, v_rot]
    res = evaluator.evaluate(x, return_traj=True)

    st.subheader("Input command")
    st.write(
        {
            "orientation": orientation,
            "v_lin": v_lin,
            "v_rot": v_rot,
            "mode": mode,
        }
    )

    st.subheader("Risk & OOD summary")
    risk_label = res["risk_label"]

    # simple color hint
    if "high-risk" in risk_label:
        st.error(f"Risk label: {risk_label}")
    elif "borderline" in risk_label:
        st.warning(f"Risk label: {risk_label}")
    else:
        st.success(f"Risk label: {risk_label}")

    st.write(f"Mahalanobis score: `{res['mahalanobis_score']:.3f}`")
    st.write(f"Isolation Forest score: `{res['isolation_forest_score']:.3f}`")
    st.write(f"Combined risk score: `{res['risk_score']:.3f}`")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted trajectory (x–y)")
        fig_traj = plot_trajectory(res["trajectory"])
        st.pyplot(fig_traj)

    with col2:
        st.subheader("Feature importance")
        fig_imp = plot_feature_importance(res["feature_importance"])
        st.pyplot(fig_imp)

    st.subheader("Raw result (debug)")
    st.json(res)
else:
    st.info("Choose parameters in the sidebar and click **Evaluate**.")
