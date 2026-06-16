# Software Design Document

**Project:** TaskPilot — Sustainable Cloud Job Scheduler  
**Version:** 1.0  
**Date:** 2026-06-16  
**Author:** PerinbaBuilds

---

## 1. Introduction

### 1.1 Purpose

This document describes the internal architecture, component design, data flows, and key design decisions for TaskPilot.

### 1.2 Scope

Covers: backend API, RL agent design, scoring engine, XAI pipeline, session isolation model, and frontend dashboard.

---

## 2. System Architecture

TaskPilot is a single-process Python web application. There is no separate frontend build step — the dashboard is a single HTML file served by FastAPI and rendered entirely in the browser.

```
Browser
  └── dashboard.html (Vanilla JS + Chart.js 4.4)
        │
        │ HTTP  (X-Session-ID header on every request)
        ▼
FastAPI  api.py
  ├── Session store (_sessions dict, keyed by UUID)
  ├── Scoring engine (compute_scores)
  ├── RL agents (_rl_agents dict, 3 × TierAgent)
  ├── XAI pipeline (generate_explanation + _xai_cache)
  └── Dataset layer (STEEL, TASKS, PRICE, SERVERS DataFrames)
```

### 2.1 Process Model

The application runs as a single Uvicorn worker. All state (sessions, RL weights, dataset pointers) lives in process memory. A restart resets everything including RL agent weights (agents re-train from scratch in ~500 episodes on startup).

---

## 3. Component Design

### 3.1 Session Store

```python
_sessions: dict[str, SessionDict]

SessionDict = {
    "job_queue":      list[dict],         # pending jobs
    "job_id_counter": int,                # monotonic ID
    "submitted_ids":  set[int],
    "server_loads":   dict[str, int],     # dc_name -> jobs assigned
    "rr_index":       dict[str, int],     # fallback round-robin index (unused by RL)
}
```

Sessions are created lazily on first access. The session UUID comes from the browser's `localStorage` and is sent as `X-Session-ID` on every request. There is no expiry — sessions live for the process lifetime.

### 3.2 Dataset Layer

Four DataFrames loaded at startup:

| Variable | File | Rows | Key columns |
|----------|------|------|-------------|
| `STEEL` | `steel_industry_data.csv` | 35,040 | `Usage_kWh`, `CO2(tCO2)`, `NSM` |
| `TASKS` | `dataset_rl/task_15min_L.csv` | varies | `plan_cpu_i`, `plan_mem_int` |
| `PRICE` | `dataset_rl/price.csv` | varies | `price_1` |
| `SERVERS` | `dataset_rl/Server_L.xlsx` | 9 | `ID`, `CPU`, `GPU`, `cpu_rate`, `gpu_rate` |

**Carbon factor derivation:**  
Raw `CO2(tCO2)` values are ~50% zero (making min-max normalisation produce a flat line). The carbon factor is instead computed as:

```python
carbon_factor = 0.5 * rank_percentile(CO2) + 0.5 * normalise(NSM)
```

`NSM` (seconds from midnight) provides a time-of-day component that mirrors how real grid carbon intensity rises and falls through the day.

**Dataset pointers** (`DATA_PTR`, `STEEL_PTR`) advance by 1 per scheduled job, cycling with modulo. This simulates a streaming time series.

### 3.3 Scoring Engine (`compute_scores`)

For every job, the scoring engine evaluates all 9 servers but restricts competition to the 3 servers in the job's tier pool:

```
Tier pools (servers sorted by CPU ascending):
  green       → indices 0-2   (least powerful = most energy-efficient)
  balanced    → indices 3-5
  performance → indices 6-8   (most powerful)
```

Per-server metrics computed:
- `perf_score` — normalised CPU+GPU throughput
- `cost_score` — inverse of energy cost (ekWh × energy_price)
- `co2_score`  — inverse of CO2 output (ekWh × (0.35 + carbon_factor))
- `lat_score`  — inverse of latency proxy (1 / cpu_rate)

All four scores are min-max normalised across the 9-server set, then the weighted quality score is computed per priority:

```python
PRIORITY_WEIGHTS = {
    "green":       (0.10, 0.25, 0.40, 0.25),  # (perf, cost, co2, lat)
    "balanced":    (0.25, 0.25, 0.25, 0.25),
    "performance": (0.40, 0.10, 0.10, 0.40),
}
```

### 3.4 RL Agent (`rl/agents.py` — `TierAgent`)

**Algorithm:** Multi-action linear Q-learning with TD(0) updates.

```
W: ndarray shape (n_servers=3, N_FEATURES=7)

Q(s, a) = W[a] · features(s)

On each scheduled job:
  action = argmax Q(s, a)   [with ε-greedy exploration]
  reward = quality_score(chosen) - overload_penalty(chosen)
  TD     = reward + γ · max_a Q(s_next, a) - Q(s, action)
  W[action] += lr · TD · features(s)
```

**Feature vector:**
```
[server0_load / max_load,   # normalised load for each pool server
 server1_load / max_load,
 server2_load / max_load,
 1.0 - carbon_factor,       # high = clean energy (agent should prefer acting now)
 energy_price,
 job_cpu / 100,
 job_memory / 100]
```

**Exploration:** `ε = 0.15 × max(0.1, 1 - steps / 2000)` — decays from 15% to 1.5% over 2000 jobs.

**Pre-training:** At startup, `_init_rl_agents()` runs 500 simulated episodes per agent with randomised server loads and a strong overload penalty (0.8×). This gives the agents a reasonable initial policy before any real jobs arrive.

**Reward function:**
```python
quality  = bd["perf_score"]*w[0] + bd["cost_score"]*w[1] + bd["co2_score"]*w[2] + bd["lat_score"]*w[3]
overload = max(0.0, server_loads[chosen] - avg_pool_load) / 5.0
reward   = quality - overload
```

The `reward` field returned to the client is `quality_score` (0–1, always non-negative) to keep the dashboard display stable. The internal RL reward (which can be negative due to overload penalties) is used only for agent training.

### 3.5 XAI Pipeline

```
generate_explanation(job, state, chosen_dc, scores, breakdown, metrics, rl_q)
  │
  ├── cache key = (server, priority, carbon_band)
  │   carbon_band = "low" / "med" / "high"  (max ~9 unique keys for 100 jobs)
  │
  ├── cache hit  → return cached string
  │
  ├── Groq available → call llama-3.3-70b-versatile with structured prompt
  │     prompt includes: tier pool, server scores, weights, carbon conditions
  │
  └── Groq unavailable → deterministic template
        identifies dominant metric, carbon description, rival servers
        returns natural-language paragraph
```

Caching by `(server, priority, carbon_band)` caps Groq API calls at ~9 for any batch size (3 servers × 3 priorities × 3 carbon bands).

---

## 4. Data Flow — Single Job Scheduling

```
Browser: POST /submit_batch [{job}]
  └─> api: append to session["job_queue"], assign job_id

Browser: POST /run
  └─> api: for each job in queue:
        1. compute_state()       → state (load, carbon_factor, energy_price, ...)
        2. compute_scores()      → scores{}, breakdown{}
        3. TierAgent.act(feat)   → action index  [ε-greedy]
        4. chosen_dc = pool[action]
        5. _rl_reward()          → rl_reward (training signal)
        6. TierAgent.learn(feat, action, rl_reward, next_feat)
        7. session["server_loads"][chosen_dc] += 1
        8. generate_explanation() → explanation string
        9. append result to response list

Browser: receives results list
  └─> renderOverview(), renderHistory(), renderAnalytics(), renderXAITab()
```

---

## 5. Frontend Design

The dashboard is a single HTML file (`frontend/templates/dashboard.html`) with no build step. All logic is in plain JavaScript.

### 5.1 Tab Structure

| Tab | ID | Key content |
|-----|----|-------------|
| Overview | `tab-overview` | Sustainability score ring, KPI cards, reward timeline, server donut |
| History | `tab-history` | Scrollable table of all jobs in session |
| Analytics | `tab-analytics` | 4 charts + full history table |
| XAI | `tab-xai` | Job selector, explanation text, score breakdown bars |

### 5.2 Charts (Chart.js 4.4)

| Chart | Type | Description |
|-------|------|-------------|
| Reward Timeline | Line | 3 datasets: raw (faint), 3-job rolling avg (green), 10-job trend (blue dashed) |
| Server Distribution | Doughnut | Jobs per server, colour-coded by tier |
| Avg Reward per Server | Bar | Mean quality score per server |
| Energy & CO2 by Server | Bar (grouped) | Total kWh and CO2 per server |
| Priority Routing | Bar (grouped) | % of each priority routed to each server |
| Energy Intensity Dist. | Bar | 5 quintile bands (Very Low → Very High), green-to-red |
| Carbon Intensity Over Time | Line | carbon_factor per job, indexed by job number |

### 5.3 Session Management

```javascript
// On page load
let sessionId = localStorage.getItem('gm_session_id');
if (!sessionId) { sessionId = crypto.randomUUID(); localStorage.setItem(...); }

// On every fetch
headers: { 'X-Session-ID': sessionId }
```

### 5.4 Progress Overlay

A full-screen overlay (`#schedOverlay`, `z-index: 99999`) shows step-by-step progress during scheduling:
- Step 1: Parsing CSV / validating jobs
- Step 2: Submitting batch to server
- Step 3: Running RL scheduler
- Step 4: Fetching explanations
- Step 5: Rendering results

---

## 6. Deployment

### 6.1 Render (production)

```yaml
# render.yaml
services:
  - type: web
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GROQ_API_KEY
        sync: false
```

Branch `main` is auto-deployed on push. Development happens on branch `fresh`; `fresh` is force-pushed to `main` to trigger a deploy.

### 6.2 Docker (local)

```bash
docker compose up --build
```

See `docker-compose.yml` for configuration.

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single-process, in-memory state | Free-tier deployment has no database; sessions reset on restart which is acceptable for a demo |
| Linear function approximation for RL | No heavy ML framework dependency; fast startup; interpretable weights |
| Tier-pool isolation | Prevents green jobs from being routed to power-hungry servers regardless of agent policy |
| Quality score (not RL reward) displayed to user | RL reward can go negative due to overload penalty; quality score is always 0–1 and more meaningful to users |
| XAI cache by `(server, priority, carbon_band)` | Caps Groq API calls at ~9 per batch regardless of batch size |
| Carbon factor blended with NSM | Raw CO2 column is ~50% zero; time-of-day signal adds continuous variation that mirrors real grid patterns |
| No frontend build step | Reduces deployment complexity; single HTML file is easy to edit and redeploy |
