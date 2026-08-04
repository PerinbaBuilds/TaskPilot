# TaskPilot

**A sustainable cloud job scheduler that routes each workload to the server with the lowest carbon + cost + latency cost — and uses reinforcement learning to keep getting better at it.**

> Live demo: [taskpilot-krt8.onrender.com](https://taskpilot-krt8.onrender.com)
> *(free-tier host — first request may take ~50s to wake)*

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Groq_LLM-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq"/>
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" alt="Chart.js"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render"/>
</p>

---

## Why this exists

Most schedulers optimise for speed and price and stop there — the carbon footprint of *where* a job runs is invisible. I wanted to see whether a scheduler could treat sustainability as a first-class objective alongside performance, and whether a lightweight RL agent could learn a better placement policy than fixed rules while staying explainable enough to trust. TaskPilot is that experiment, wrapped in a dashboard you can actually watch make decisions.

---

## Features

- **Carbon-aware routing** — every job is placed using a live grid-carbon signal, not just CPU/price.
- **RL placement agents** — one Q-learning agent per priority tier learns from each decision and improves online.
- **Tier pools** — green jobs only compete on efficient servers, performance jobs on powerful ones; sustainability is structural, not optional.
- **Explainable decisions** — each placement comes with a plain-English "why", optionally upgraded by an LLM on demand.
- **Live analytics** — reward trend, server distribution, energy/CO₂ per server, and carbon intensity over time.
- **Bulk scheduling** — upload a CSV of 100+ jobs and watch them route in real time.

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | FastAPI + Uvicorn | Async, tiny footprint, serves both the API and the single-page dashboard from one process |
| RL / ML | NumPy | Linear Q-learning needs no heavy framework — fast cold-start and fully inspectable weights |
| LLM (XAI) | Groq — `llama-3.3-70b-versatile` | Fastest hosted inference for on-demand explanations; strictly optional with a rule-based fallback |
| Data | Pandas + OpenPyXL | The energy/workload/server datasets are tabular; Pandas is the path of least resistance |
| Frontend | Vanilla JS + Chart.js | No build step — the dashboard is one HTML file, trivial to deploy and edit |
| Config | python-dotenv | Zero-friction local secrets via `.env` |

*Only non-obvious choices are justified — the rest are the boring, correct defaults.*

---

## Architecture

```
Browser (dashboard.html — Vanilla JS + Chart.js)
        │  POST /run  { jobs:[...] }   ·   X-Session-ID header
        ▼
┌─────────────────────── FastAPI (api.py) ───────────────────────┐
│                                                                 │
│  normalize_job ──► compute_state ──► compute_scores            │
│       │                 │                  │                    │
│       │            (carbon, price,     (perf/cost/co2/lat       │
│       │             load signal)        per server, per tier)   │
│       ▼                                    ▼                    │
│  Session store                      TierAgent (RL)             │
│  (per X-Session-ID,                  picks server in pool,      │
│   in-memory)                         TD(0) learns from reward   │
│                                            │                    │
│                                            ▼                    │
│                                    scheduled_jobs + metrics     │
└─────────────────────────────────────────────────────────────────┘
        │
        └──► lazy POST /explain  ──►  Groq LLM (or rule-based fallback)

Datasets (loaded once at startup):
  Steel Industry Energy (UCI) → carbon + energy signal
  Server_L.xlsx               → 9-server cluster specs
  task_15min traces + price   → load + energy price
```

- **`normalize_job`** — the one interesting defensive decision: the browser can send messy payloads (string CPU values, a `mem`/`memory` alias, unknown priorities). Every job is coerced to safe, clamped types *before* scheduling, so `/run` can never crash on user input. The tradeoff: the server silently corrects bad data instead of rejecting it — fine for a scheduler, wrong for, say, a payments API.
- **RL vs. `/run` latency** — the agent runs inline (it's just a matrix-vector product), but LLM explanations are deferred to `/explain` so a batch of 100 jobs never blocks on network calls.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full breakdown.

---

## Getting Started

```bash
git clone https://github.com/PerinbaBuilds/TaskPilot.git
cd TaskPilot
cp .env.example .env        # optional — fill in GROQ_API_KEY for LLM explanations
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Or with Docker:

```bash
docker compose up --build   # open http://localhost:8000
```

**Requirements:** Python 3.11+ (Docker image pins 3.11). No database — all state is in-memory per session. A Groq API key is optional; without it, explanations fall back to a rule-based generator.

---

## Usage

1. Open the dashboard and click **New Job**, or upload a CSV from the queue panel.
   A CSV needs `priority,latency,cpu,memory` (columns `estimated_kwh,co2_kg` are optional):

   ```csv
   priority,latency,cpu,memory
   green,low,15,39
   performance,high,80,70
   balanced,medium,45,55
   ```

2. Hit **Run Scheduler**. Each job is routed to a server, and the Overview / Analytics / Explainable-AI tabs fill in live.

Prefer the API directly? `/run` is self-contained — send the jobs in the body:

```bash
curl -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: demo' \
  -d '[{"priority":"green","latency":"low","cpu":15,"memory":39}]'
```

---

## Known Limitations / What I'd Do Differently

- **State is in-memory only.** A server restart wipes sessions and resets the RL weights (they retrain in ~500 episodes on boot). Fine for a demo; a real deployment would persist agent weights and use Redis for sessions.
- **The "cluster" is simulated.** Nine servers from a spec sheet, not real infrastructure — placement decisions aren't executed against anything.
- **RL is linear function approximation.** It learns useful load-balancing behaviour, but a deep policy (or contextual bandit) would likely do better on the reward signal; I chose linear for fast cold-start and interpretability.
- **No automated test suite yet.** Endpoints were verified manually and with ad-hoc scripts; the scheduling loop and `normalize_job` deserve real unit tests.
- **Carbon signal is a proxy.** It's derived from the UCI Steel Industry dataset blended with a time-of-day curve, not a live grid API like WattTime/Electricity Maps — which is what I'd wire in for a production version.

---

## License

MIT
