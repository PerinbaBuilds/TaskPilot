# 🌱 TaskPilot — Sustainable Cloud Job Scheduler

<p align="center">
  <img src="assets/logo.svg" width="100" alt="TaskPilot Logo"/>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/></a>
  <a href="https://en.wikipedia.org/wiki/Reinforcement_learning"><img src="https://img.shields.io/badge/Reinforcement%20Learning-FF6B35?style=for-the-badge"/></a>
  <a href="https://groq.com/"><img src="https://img.shields.io/badge/Groq%20LLM-XAI-8B5CF6?style=for-the-badge"/></a>
  <a href="https://taskpilot-krt8.onrender.com"><img src="https://img.shields.io/badge/Live-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white"/></a>
</p>

<p align="center">
  <b>Route cloud workloads intelligently — minimising carbon emissions using real-time energy data, RL agents, and LLM-powered explainability.</b>
</p>

<p align="center">
  🌐 <a href="https://taskpilot-krt8.onrender.com"><strong>Live Demo → taskpilot-krt8.onrender.com</strong></a>
</p>

---

## 🚀 What Is TaskPilot?

Most cloud schedulers optimise for speed and cost alone. TaskPilot adds a third dimension: **sustainability**.

It routes each compute job to the server that best balances:

| Metric | Weight (green) | Weight (balanced) | Weight (performance) |
|--------|:--------------:|:-----------------:|:--------------------:|
| ⚡ Energy efficiency | 40% | 25% | 10% |
| 💰 Cost | 25% | 25% | 10% |
| 🚀 Throughput | 10% | 25% | 40% |
| ⏱ Latency | 25% | 25% | 40% |

On top of static scoring, **3 RL agents** (one per priority tier) continuously learn from system state and carbon intensity data — blending their policy 25% into every scheduling decision.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────┐
│                 Browser Dashboard                 │
│         HTML · CSS · JS · Chart.js 4.4           │
│    (session-isolated — each user owns their data) │
└────────────────────┬─────────────────────────────┘
                     │ HTTP / same-origin
┌────────────────────▼─────────────────────────────┐
│              FastAPI  (api.py)                    │
│                                                   │
│  ┌──────────────────┐   ┌──────────────────────┐ │
│  │  3 × RL Agents   │   │  Groq LLM (XAI)      │ │
│  │  TD-error update │   │  llama-3.3-70b       │ │
│  └────────┬─────────┘   └──────────────────────┘ │
│           │                                       │
│  ┌────────▼──────────────────────────────────┐   │
│  │       Tier-Pool Scoring Engine            │   │
│  │  green → bottom third (efficient servers) │   │
│  │  balanced → middle third                  │   │
│  │  performance → top third (powerful)       │   │
│  └────────────────────────────────────────────┘  │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  Datasets                                  │  │
│  │  • Steel Industry Energy Dataset  (UCI)    │  │
│  │  • 9-server synthetic cluster              │  │
│  │  • 15-min workload traces                  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

---

## ✨ Features

- **RL-powered routing** — agents trained at startup on real energy data, blended into every decision
- **Tier-pool isolation** — green jobs only compete on efficient servers; performance jobs on powerful ones
- **Explainable AI** — Groq LLM explains every scheduling decision in plain English
- **Live carbon signal** — real-time carbon intensity from Steel Industry Dataset drives recommendations
- **Sustainability score** — composite metric tracking green ratio, CO₂ efficiency, and reward
- **Session isolation** — each browser session has its own independent queue and history
- **Bulk CSV upload** — submit 100s of jobs at once from a CSV file

---

## 🗂 Project Structure

```
TaskPilot/
├── api.py                    ← FastAPI entry point (scoring, RL, XAI, serving HTML)
├── rl/                       ← Reinforcement learning
│   ├── agents.py             ← Linear function approximation agent
│   ├── rl_env.py             ← CloudEnv: state/reward simulation
│   └── train_agents.py       ← Standalone training script
├── core/                     ← Shared utilities
│   ├── config.py             ← Priority weight definitions
│   ├── data_loader.py        ← Dataset loading & normalisation
│   ├── energy_model.py       ← Server power model (P_idle → P_peak)
│   ├── job_queue.py          ← In-memory job queue
│   └── llm_manager.py        ← Fallback explanation generator
├── frontend/
│   └── templates/
│       └── dashboard.html    ← Single-page dashboard (Chart.js, dark theme)
├── data/
│   ├── steel_industry_data.csv   ← UCI energy dataset
│   ├── dataset_rl/               ← Server specs + workload traces
│   └── demo_jobs.csv             ← Sample jobs for testing ↓
├── Procfile                  ← Railway deployment
├── render.yaml               ← Render deployment
└── requirements.txt
```

---

## 🧪 Try It — Demo Jobs

Upload **`data/demo_jobs.csv`** from the dashboard to test with a pre-built mix of 10 jobs:

```csv
cpu,memory,priority,latency
52,16,performance,high
14,71,balanced,high
30,46,green,high
23,23,green,medium
72,75,balanced,high
11,59,balanced,low
79,62,balanced,low
77,98,green,medium
75,71,performance,medium
95,64,green,medium
```

Hit **▶ Run Scheduler** to see RL-powered routing, server selection, sustainability score and XAI explanations live.

---

## 🤖 RL Agent Design

```python
# State vector fed to each agent
state = [green_score, system_load, job_size, energy_cost]

# Reward function
reward = w_green × green − w_cost × cost − w_perf × load

# TD-error weight update (linear function approximation)
weights += lr × (reward − dot(weights, state)) × state

# Final score blending
final_score = 0.75 × static_score + 0.25 × RL_score
```

Three agents train for **300 episodes each at startup** — one per priority tier (green / balanced / performance).

---

## 📊 Sustainability Score

```
score = (avg_reward    × 0.30
       + green_ratio   × 0.25
       + carbon_score  × 0.25   ← (1 − avg_carbon_factor)
       + co2_efficiency× 0.20) × 100
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| ML / RL | NumPy (no heavy ML framework) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Data | Pandas, OpenPyXL |
| Frontend | Vanilla JS + Chart.js 4.4 |
| Hosting | Render (free) / Railway |

---

## 🏃 Run Locally

```bash
git clone https://github.com/PerinbaBuilds/TaskPilot.git
cd TaskPilot
pip install -r requirements.txt

# Windows
set GROQ_API_KEY=your_key_here
# Mac / Linux
export GROQ_API_KEY=your_key_here

uvicorn api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```
