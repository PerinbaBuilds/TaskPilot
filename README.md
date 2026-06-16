# TaskPilot — Sustainable Cloud Job Scheduler

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

## What Is TaskPilot?

Most cloud schedulers optimise for speed and cost alone. TaskPilot adds a third dimension: **sustainability**.

It routes each compute job to the server that best balances performance, cost, and carbon footprint — using a live energy signal and RL agents that continuously learn from scheduling outcomes.

| Metric | Green tier | Balanced tier | Performance tier |
|--------|:----------:|:-------------:|:----------------:|
| Energy efficiency | 40% | 25% | 10% |
| Cost | 25% | 25% | 10% |
| Throughput | 10% | 25% | 40% |
| Latency | 25% | 25% | 40% |

---

## Architecture

```
┌───────────────────────────────────────────────────┐
│                  Browser Dashboard                 │
│          HTML · CSS · Chart.js 4.4 · JS           │
│     (session-isolated — every user is independent) │
└─────────────────────┬─────────────────────────────┘
                      │ HTTP / same-origin
┌─────────────────────▼─────────────────────────────┐
│               FastAPI  (api.py)                    │
│                                                    │
│  ┌───────────────────┐  ┌──────────────────────┐  │
│  │  3 × RL TierAgent │  │  Groq LLM  (XAI)     │  │
│  │  TD(0) Q-learning │  │  llama-3.3-70b       │  │
│  └────────┬──────────┘  └──────────────────────┘  │
│           │                                        │
│  ┌────────▼───────────────────────────────────┐   │
│  │         Tier-Pool Scoring Engine           │   │
│  │  green       → Servers 1-3 (efficient)     │   │
│  │  balanced    → Servers 4-6 (mid-tier)      │   │
│  │  performance → Servers 7-9 (powerful)      │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  ┌────────────────────────────────────────────┐   │
│  │  Datasets                                  │   │
│  │  • Steel Industry Energy Dataset  (UCI)    │   │
│  │  • 9-server synthetic cluster              │   │
│  │  • 15-minute workload traces               │   │
│  └────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
```

---

## Features

- **RL-powered routing** — 3 TierAgents (one per priority) trained at startup on real energy data; epsilon-greedy exploration with online TD(0) updates after every job
- **Tier-pool isolation** — green jobs only compete on efficient servers; performance jobs on the powerful tier
- **Explainable AI** — Groq LLM (llama-3.3-70b-versatile) explains every scheduling decision in plain English; cached by `(server, priority, carbon_band)` to minimise API calls
- **Live carbon signal** — carbon intensity blended from the Steel Industry Dataset (UCI) drives real-time recommendations
- **Sustainability score** — composite KPI tracking green ratio, CO2 efficiency, and agent reward quality
- **Session isolation** — every browser session has its own independent queue and history via UUID header
- **Bulk CSV upload** — submit 100s of jobs at once from a CSV file
- **Reward timeline chart** — 3-dataset smoothed view: raw, 3-job rolling average, 10-job trend
- **Energy intensity distribution** — 5 descriptive quintile bands (Very Low → Very High) with green-to-red colour gradient

---

## Project Structure

```
TaskPilot/
├── api.py                         ← FastAPI entry point (scoring, RL, XAI, HTML serving)
├── rl/
│   ├── agents.py                  ← TierAgent: linear Q-learning, TD(0) updates
│   ├── rl_env.py                  ← CloudEnv: state / reward simulation
│   └── train_agents.py            ← Standalone offline training script
├── core/
│   ├── config.py                  ← Priority weight definitions
│   ├── data_loader.py             ← Dataset loading & normalisation
│   ├── energy_model.py            ← Server power model (P_idle → P_peak)
│   ├── job_queue.py               ← In-memory job queue
│   └── llm_manager.py             ← Natural-language fallback explanation generator
├── frontend/
│   └── templates/
│       └── dashboard.html         ← Single-page dashboard (Chart.js, dark theme)
├── data/
│   ├── steel_industry_data.csv    ← UCI Steel Industry Energy Dataset
│   ├── dataset_rl/                ← Server specs (Server_L.xlsx) + task traces
│   ├── sample_100_jobs.csv        ← 100-job benchmark (priority/cpu/mem/kwh/co2)
│   └── demo_jobs.csv              ← 10-job quick demo
├── docker-compose.yml             ← Local Docker setup
├── Procfile                       ← Heroku / Railway process file
├── render.yaml                    ← Render deployment config
├── requirements.txt
├── Software_Requirements_Specification.md
└── Software_Design_Document.md
```

---

## Run Locally

```bash
git clone https://github.com/PerinbaBuilds/TaskPilot.git
cd TaskPilot
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here   # Windows: set GROQ_API_KEY=...

uvicorn api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Run with Docker

```bash
docker compose up --build
# Open http://localhost:8000
```

---

## RL Agent Design

```
State:   [server0_load_norm, server1_load_norm, server2_load_norm,
          carbon_factor, energy_price, job_cpu, job_mem]   (7 features)

Action:  server index within the tier pool  (0 / 1 / 2)

Reward:  quality_score - overload_penalty
         quality  = weighted sum of perf / cost / co2 / lat scores
         overload = max(0, dc_load - avg_pool_load) / 5

Update:  TD(0)  ->  W[action] += lr * (reward + gamma * max_Q_next - Q_now) * features
```

Three TierAgents pre-train for **500 simulated episodes at startup** with varied loads and strong overload penalties, then keep learning online after every scheduled job.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| ML / RL | NumPy (linear function approximation, no heavy framework) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Data | Pandas, OpenPyXL |
| Frontend | Vanilla JS + Chart.js 4.4 |
| Hosting | Render (free tier) |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Optional | Enables LLM-generated XAI explanations. Falls back to rule-based natural language if absent. |
| `PORT` | Set by host | Uvicorn listen port (default 8000 locally). |
