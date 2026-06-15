# GreenMind — Sustainable Cloud Job Scheduler

> Intelligent, RL-powered cloud job scheduler that routes workloads to the most energy-efficient servers based on real-time carbon intensity data.

🌐 **Live Demo:** [taskpilot-krt8.onrender.com](https://taskpilot-krt8.onrender.com)

---

## What It Does

GreenMind schedules compute jobs across a pool of servers using **reinforcement learning** and **real-time carbon intensity signals**. Instead of routing purely by performance, it balances:

- ⚡ Energy efficiency (kWh consumed)
- 🌿 Carbon footprint (CO₂ emissions)
- 💰 Cost per job
- 🚀 Throughput / latency requirements

Each job is assigned a **priority tier** (green / balanced / performance), and only servers in that tier compete for the job — enforcing sustainable routing by design.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Browser (Dashboard)             │
│         HTML + CSS + JS + Chart.js           │
└──────────────────┬──────────────────────────┘
                   │ HTTP (same-origin)
┌──────────────────▼──────────────────────────┐
│            FastAPI Backend (api.py)          │
│                                              │
│  ┌─────────────┐  ┌────────────────────┐    │
│  │  RL Agents  │  │  Groq LLM (XAI)    │    │
│  │  (3 agents) │  │  llama-3.3-70b     │    │
│  └──────┬──────┘  └────────────────────┘    │
│         │                                    │
│  ┌──────▼──────────────────────────────┐    │
│  │  Tier-Pool Scoring Engine           │    │
│  │  green → bottom third servers       │    │
│  │  balanced → middle third            │    │
│  │  performance → top third            │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │  Datasets                           │    │
│  │  • Steel Industry Energy (UCI)      │    │
│  │  • Synthetic server pool (9 nodes)  │    │
│  │  • Workload traces (task_15min)     │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS + [Chart.js 4.4](https://www.chartjs.org/) |
| **Reinforcement Learning** | NumPy — linear function approximation, TD-error updates |
| **LLM Explanations (XAI)** | [Groq API](https://groq.com/) — `llama-3.3-70b-versatile` |
| **Data** | Pandas, OpenPyXL — Steel Industry Energy Dataset (UCI) |
| **Deployment** | Railway / Render |
| **Language** | Python 3.10+ |

---

## RL Agent Design

Three agents are trained at startup (300 episodes each), one per priority tier:

```
State vector:  [green_score, system_load, job_size, energy_cost]
Reward:        w_green × green − w_cost × cost − w_perf × load
Update rule:   weights += lr × (reward − prediction) × state   (TD-error)
```

At scheduling time, scores are blended:
```
final_score = 0.75 × static_score + 0.25 × RL_score
```

---

## Sustainability Score Formula

```
score = (avg_reward × 0.30
       + green_job_ratio × 0.25
       + (1 − avg_carbon) × 0.25
       + co2_efficiency × 0.20) × 100
```

---

## Project Structure

```
TaskPilot/
├── api.py                  # FastAPI entry point — scoring, RL, XAI
├── rl/                     # Reinforcement learning package
│   ├── agents.py           # Linear function approximation agent
│   ├── rl_env.py           # CloudEnv — state/reward simulation
│   └── train_agents.py     # Standalone training script
├── core/                   # Shared utilities
│   ├── config.py           # Priority weight configs
│   ├── data_loader.py      # Dataset loading + normalisation
│   ├── energy_model.py     # Power consumption model
│   └── job_queue.py        # In-memory job queue
├── frontend/               # Web layer
│   ├── flask_app.py        # (legacy) Flask server
│   └── templates/
│       └── dashboard.html  # Single-page dashboard
├── data/                   # Datasets
│   ├── steel_industry_data.csv
│   ├── dataset_rl/         # Server specs + workload traces
│   └── demo_jobs.csv       # Sample jobs for testing
├── Procfile                # Railway deployment
├── render.yaml             # Render deployment
└── requirements.txt
```

---

## Running Locally

```bash
git clone https://github.com/PerinbaBuilds/TaskPilot.git
cd TaskPilot
pip install -r requirements.txt

# Set your Groq API key (optional — fallback explanation used if missing)
set GROQ_API_KEY=your_key_here   # Windows
export GROQ_API_KEY=your_key_here  # Mac/Linux

uvicorn api:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

---

## Demo

Upload `data/demo_jobs.csv` from the dashboard to see a pre-built set of 10 jobs across all three priority tiers, then hit **Run Scheduler** to see RL-powered routing in action.

```csv
cpu,memory,priority,latency
52,16,performance,high
14,71,balanced,high
30,46,green,high
23,23,green,medium
72,75,balanced,high
11,59,balanced,low
...
```

---

## Dashboard Features

| Tab | Features |
|-----|---------|
| **Overview** | KPI cards, reward timeline, server distribution doughnut, carbon intensity chart |
| **Jobs** | Pending queue, completed history, CSV bulk upload |
| **Analytics** | Per-server reward, energy vs CO₂, priority breakdown |
| **Explainable AI** | LLM explanation per job, radar chart, score breakdown |
| **Sustainability** | Animated gauge, CO₂ avoided estimate, carbon recommendation |

---

## Contributors

| Name | Role |
|------|------|
| [PerinbaBuilds](https://github.com/PerinbaBuilds) | Creator & Lead Developer |

---

## License

MIT
