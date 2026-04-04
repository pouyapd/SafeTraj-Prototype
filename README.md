# SafeTraj-Prototype

**Trajectory Behaviour Analysis Toolkit for Neural Motion Predictors**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?logo=streamlit)](https://safetraj-prototype-auhavyacsp7udguqdsezhm.streamlit.app/)

---

**Author:** Pouya Bathaei Pourmand  
**Affiliation:** MSc in Computer Engineering (AI) — University of Genoa, Italy  
**Project:** [REXASI-PRO](https://rexasi-pro.spindoxlabs.com/) — Reliable & Explainable AI for Smart Mobility (EU Horizon Europe)

---

## Overview

SafeTraj is a lightweight and modular Python toolkit for analysing the behaviour of pretrained neural trajectory prediction models in autonomous mobility systems (smart wheelchairs, mobile robots).

It was developed alongside my MSc thesis research at the University of Genoa as a personal open-source tool, inspired by trajectory analysis work conducted within the EU Horizon Europe REXASI-PRO project.

The focus is on **behaviour analysis, not model training** — the neural models analysed were pretrained by project partners. SafeTraj only performs trajectory-level evaluation and does not include proprietary model weights.

---

## Why This Project?

Learning-based motion predictors can behave very differently depending on input conditions such as initial orientation and velocity commands. Understanding *when and how* these models fail is essential for improving reliability in safety-critical systems.

SafeTraj provides a transparent environment to:
- Analyse predicted trajectories and identify unstable behaviours
- Estimate trajectory-level risk scores
- Visualise model behaviour across different input conditions
- Generate interpretable explanations of failure patterns

---

## Features

### Trajectory Analysis
Evaluates predicted kinematic trajectories `[x(t), y(t), θ(t)]` using:
- Distance to goal
- Trajectory deviation and curvature
- Statistical thresholds derived from training data

### Risk Estimation
Computes per-trajectory:
- Risk score (continuous)
- Risk category: Low / Medium / High

### Explainability
Feature-importance analysis showing which input commands (orientation, velocity) most affect trajectory risk — supporting interpretable model comparison.

### Interactive Dashboard
Streamlit dashboard for real-time exploration:
- Interactive input controls (φ, v, ω)
- Live trajectory visualisation
- Live risk estimation and category display

**Low-risk command:**

![Dashboard Low Risk](https://raw.githubusercontent.com/pouyapd/SafeTraj-Prototype/main/assets/dashboard_low_risk.png)

**High-risk command:**

![Dashboard High Risk](https://raw.githubusercontent.com/pouyapd/SafeTraj-Prototype/main/assets/dashboard_high_risk.png)

### TrajSafe-LLM Module
A lightweight reporting module under `trajsafe_llm/` that produces rule-based safety labels and optional natural-language explanations via a local LLM (Ollama).

- CLI-friendly safety report generation
- Optional LLM explanation as a post-analysis layer
- Designed for interpretability and debugging, not as a replacement for formal safety validation

---

## Project Structure

```
SafeTraj-Prototype/
├── api/                # FastAPI REST endpoint
│   └── main.py
├── safetraj/           # Core analysis and risk estimation modules
├── dashboard/          # Streamlit interactive dashboard
├── trajsafe_llm/       # LLM-based safety reporting module
├── examples/           # Example scripts and usage demos
├── tests/              # Unit tests
├── assets/             # Dashboard screenshots
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/pouyapd/SafeTraj-Prototype.git
cd SafeTraj-Prototype
pip install -r requirements.txt
```

### Run the Dashboard
```bash
streamlit run dashboard/app.py
```

### Run the API
```bash
uvicorn api.main:app --reload
```

### Run the CLI Safety Report
```bash
python -m trajsafe_llm.report --input examples/sample_trajectory.csv
```

---

## API

A REST API is available for programmatic access to trajectory risk scoring.

### Interactive docs
Once the API is running, visit:
```
http://127.0.0.1:8000/docs
```

### Example request
```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"orientation": 0.0, "v_lin": 0.5, "v_rot": 0.2}'
```

### Example response
```json
{
  "risk_label": "in-distribution & low-risk",
  "risk_score": 0.6523,
  "mahalanobis_score": 1.1579,
  "isolation_forest_score": -0.1063,
  "feature_importance": {
    "orientation": 0.0,
    "v_lin": 0.714,
    "v_rot": 0.286
  },
  "trajectory_length": 20
}
```

---

## Related Projects

- [SafeTraj-Experiments](https://github.com/pouyapd/SafeTraj-Experiments) — Full thesis experimental results: input sensitivity analysis, goal difficulty maps, and model comparison across DNN-LNA architectures
- [SafeNav-RL](https://github.com/pouyapd/SafeNav-RL) — RL-based navigation agent extending this analysis work toward safety-constrained policy learning

---

*Developed as part of MSc thesis research at the University of Genoa, inspired by work conducted within the EU Horizon Europe REXASI-PRO project.*
