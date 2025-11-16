# SafeTraj-X

SafeTraj-X is a compact tool for **trajectory prediction**, **input-based OOD detection**, and **risk estimation** for mobile robots and smart wheelchairs.

It takes a simple motion command:

[orientation, v_lin, v_rot]

yaml
Copy code

and produces:

- predicted kinematic trajectory  
- two OOD scores: Mahalanobis and Isolation Forest  
- a combined risk score  
- a human-readable risk label  
- simple feature-importance  
- optional real-time visualization via Streamlit  

---

## 🔍 Overview

SafeTraj-X provides a clean and modular framework for studying how motion commands behave under a lightweight kinematic model, and how unusual or risky those commands may be.

The project is packaged as a small Python module with:

- a single high-level evaluator (`SafeTrajEvaluator`)
- optional plotting tools
- optional Streamlit dashboard
- examples and tests

This makes the tool suitable for research, safety validation, explainability, and sharing results.

---

## ✨ Features

### **Trajectory Prediction**
Predicts a sequence `[x(t), y(t), θ(t)]` using a lightweight kinematic model.

### **OOD Detection**
- Mahalanobis distance using empirical covariance  
- Isolation Forest anomaly score  

### **Risk Estimation**
Both OOD results are normalized to `[0, 1]` and combined to produce:

- `risk_score`  
- `risk_label`:
  - low-risk  
  - borderline  
  - high-risk  

### **Simple Explainability**
Lightweight feature-importance for the three motion inputs.

### **Streamlit Dashboard**
Interactive sliders → real-time trajectory → real-time risk.

---

## 📦 Installation

```bash
git clone https://github.com/pouyapd/SafeTraj-X.git
cd SafeTraj-X
pip install -r requirements.txt
pip install -e .
To launch the dashboard:

bash
Copy code
streamlit run dashboard/app.py
🚀 Quick Example
python
Copy code
from safetraj import SafeTrajEvaluator

# Motion command: [orientation, linear_velocity, angular_velocity]
cmd = [0.5, 1.2, -0.3]

evaluator = SafeTrajEvaluator()
result = evaluator.evaluate(cmd)

print("Risk:", result["risk_label"], result["risk_score"])
print("Trajectory shape:", result["trajectory"].shape)
🧠 Risk Score Details
Mahalanobis score measures distance from training distribution.

Isolation Forest assigns an anomaly score.

Both are normalized to [0, 1].

Combined via weighted average (default 0.5 / 0.5).

Thresholds:

< 0.33 → low-risk

< 0.66 → borderline

≥ 0.66 → high-risk

📁 Project Structure
yaml
Copy code
SafeTraj-X/
│
├── safetraj/
│   ├── __init__.py
│   ├── config.py            # Configuration dataclass
│   ├── data.py              # Training data generation
│   ├── predictor.py         # Kinematic trajectory predictor
│   ├── ood.py               # OOD detectors (Mahalanobis + IF)
│   ├── xai.py               # Feature-importance
│   ├── evaluator.py         # Main high-level evaluator
│   └── plotting.py          # Matplotlib trajectory plotter
│
├── dashboard/
│   └── app.py               # Streamlit interactive dashboard
│
├── examples/
│   └── demo_basic.py        # CLI demo
│
├── tests/
│   └── test_evaluator.py    # Minimal sanity test
│
└── README.md
