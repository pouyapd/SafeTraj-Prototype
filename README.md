# SafeTraj-X

SafeTraj-X is a compact tool for **trajectory prediction**, **input-based OOD detection**, and **risk estimation** for mobile robots and smart wheelchairs.  
It converts a simple motion command `[orientation, v_lin, v_rot]` into:

- a predicted kinematic trajectory  
- two OOD scores (Mahalanobis + Isolation Forest)  
- a combined risk score  
- a human-readable risk label  
- simple feature-importance  
- optional real-time visualization via Streamlit  

This structure makes SafeTraj-X suitable for safety research, evaluation, and explainability.

---

## 🔍 Overview

The goal of SafeTraj-X is to provide a **clean, modular, and reproducible** framework for studying how different motion commands behave under simple kinematic prediction and how “unusual” or risky those commands might be.

Instead of scattering experiments across notebooks or scripts, the project organizes everything into a small Python package with:

- a single high-level evaluator (`SafeTrajEvaluator`)
- optional visualization tools
- optional dashboard for interactive exploration

This makes analysis faster, clearer, and easier to share with supervisors or interviewers.

---

## ✨ Features

- **Trajectory Prediction**  
  Predicts a sequence `[x(t), y(t), θ(t)]` using a lightweight kinematic model.

- **Two OOD Detectors**  
  - Mahalanobis distance (Empirical Covariance)  
  - Isolation Forest (scikit-learn)

- **Risk Estimation**  
  Combines both OOD scores into a single `risk_score` and a label:
  - low-risk  
  - borderline  
  - high-risk  

- **Simple Explainability**  
  Heuristic feature-importance for the three input values.

- **Streamlit Dashboard**  
  Sliders → real-time trajectory → real-time risk updates.

---

## 📁 Project Structure


SafeTraj-X/
  safetraj/:               # Core package
    __init__.py
    config.py              # Configuration dataclass
    data.py                # Synthetic training data generation
    predictor.py           # Kinematic trajectory predictor
    ood.py                 # Mahalanobis + Isolation Forest OOD detectors
    xai.py                 # Simple feature-importance
    evaluator.py           # High-level risk evaluator (main API)
    plotting.py            # Matplotlib plotting utilities

  dashboard/:
    app.py                 # Streamlit interactive dashboard

  examples/:
    demo_basic.py          # CLI demonstration script

  tests/:
    test_evaluator.py      # Minimal unit test (sanity check)


   - ## 🛠 Installation
     
  pip install -r requirements.txt
---

  ## 📜 License
  MIT License

    


  requirements.txt
  README.md
