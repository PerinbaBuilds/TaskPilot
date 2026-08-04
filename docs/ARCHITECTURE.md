# TaskPilot — Architecture

A detailed walkthrough of how TaskPilot is put together: the request lifecycle, each
component's responsibility, the RL and scoring internals, the session model, the
datasets, deployment, and the known failure modes.

> For a one-screen summary, see the diagram in the [README](../README.md).
> For requirements and design rationale, see
> [`Software_Requirements_Specification.md`](Software_Requirements_Specification.md) and
> [`Software_Design_Document.md`](Software_Design_Document.md).

---

## 1. System overview

TaskPilot is a **single-process** web application. One FastAPI app (`api.py`) both
serves the dashboard HTML and exposes the JSON API. There is no separate frontend
build, no database, and no background worker — everything runs in one Uvicorn process.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser  ·  frontend/templates/dashboard.html                        │
│  Vanilla JS + Chart.js.  Holds the job queue + history in localStorage │
│  keyed by a per-browser UUID (X-Session-ID).                           │
└───────────────┬──────────────────────────────────────────────────────┘
                │  HTTP (same origin)
                ▼
┌──────────────────────────────── api.py ───────────────────────────────┐
│                                                                        │
│  Startup:  load datasets  ·  normalise columns  ·  pre-train 3 agents  │
│                                                                        │
│  Per request:                                                          │
│    _get_session(sid) ──► in-memory session dict                        │
│    normalize_job     ──► safe, clamped job fields                      │
│    compute_state     ──► carbon / price / load snapshot (from data)    │
│    compute_scores    ──► per-server perf/cost/co2/lat, per tier pool    │
│    TierAgent.act     ──► pick a server within the tier pool (ε-greedy)  │
│    TierAgent.learn   ──► TD(0) weight update from the reward            │
│    generate_explanation (rule-based inline; LLM on demand via /explain) │
│                                                                        │
└───────────────┬────────────────────────────────────────────────────────┘
                │ reads (once, at startup)
                ▼
┌────────────────────────────── Datasets ───────────────────────────────┐
│  data/steel_industry_data.csv     UCI energy dataset → carbon + energy │
│  data/dataset_rl/Server_L.xlsx    9-server cluster specs               │
│  data/dataset_rl/task_15min_L.csv 15-min workload traces (load)        │
│  data/dataset_rl/price.csv        energy price series                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository layout

```
TaskPilot/
├── api.py                     # FastAPI app: scoring, RL orchestration, XAI, HTML serving
├── rl/
│   ├── agents.py              # TierAgent — linear Q-learning, TD(0) updates
│   ├── rl_env.py              # CloudEnv — state/reward simulation for offline training
│   └── train_agents.py        # standalone offline training script
├── core/
│   ├── config.py              # priority weight definitions
│   ├── data_loader.py         # dataset loading & normalisation helpers
│   ├── energy_model.py        # server power model (P_idle → P_peak)
│   ├── job_queue.py           # in-memory queue helper
│   └── llm_manager.py         # rule-based explanation fallback
├── frontend/templates/
│   └── dashboard.html         # single-page dashboard (Chart.js, dark theme)
├── data/                      # datasets + sample job CSVs
├── scripts/                   # standalone CLI tools (dataset check, job generator)
├── docs/                      # this file + SRS + SDD
├── Dockerfile · docker-compose.yml · Procfile · render.yaml
└── requirements.txt · .env.example
```

---

## 3. Request lifecycle — scheduling a batch

The core path is `POST /run`. It is **self-contained**: the client sends the whole job
queue in the request body, so a server restart between calls can never lose the batch.

```
Browser: POST /run  body=[ {priority, latency, cpu, memory}, ... ]
                    header: X-Session-ID: <uuid>
   │
   ▼
1. _get_session(sid)              → isolated in-memory session dict
2. if body jobs present:          → they REPLACE the session queue (source of truth)
       normalize_job(job)         → coerce cpu/memory to clamped floats,
                                     map 'mem'→'memory', validate priority/latency,
                                     assign a monotonic job_id
   for each job in the queue:
3.   compute_state()              → {load, carbon_factor, energy_price, ...}
4.   compute_scores(state, prio)  → scores{}, metrics{}, breakdown{}  (per server)
5.   pool = servers where in_pool → the 3 servers for this priority tier
6.   TierAgent.act(features)      → server index within pool  (ε-greedy)
7.   reward = quality − overload  → _rl_reward()
8.   TierAgent.learn(f, a, r)     → TD(0) weight update
9.   server_loads[chosen] += 1    → track load for subsequent jobs in the batch
10.  explanation = rule-based     → instant, no network
11.  append result to response
   │
   ▼
Response: { scheduled_jobs:[...], server_loads:{...} }
   │
   ▼
Browser: renders Overview / Analytics / History; XAI tab lazily calls /explain per job
```

Steps 3–10 are pure Python/NumPy and run in well under a millisecond per job, so a
100-job batch completes in a fraction of a second.

---

## 4. Components

### 4.1 Session store
`_sessions: dict[str, dict]`, keyed by the `X-Session-ID` header (a UUID the browser
generates once and keeps in `localStorage`). Each session holds:

```python
{
  "job_queue":      [ ... ],           # pending jobs
  "job_id_counter": int,               # monotonic id source
  "submitted_ids":  set(),
  "server_loads":   { "Server 1": 0, ... },   # jobs placed this session
  "rr_index":       { "green": 0, ... },       # legacy round-robin cursor (unused by RL)
}
```

Sessions are created lazily and live for the process lifetime — there is no expiry and
no persistence. Two browsers never see each other's data.

### 4.2 `normalize_job(job)`
The defensive boundary. The browser can send `cpu` as the string `"52"`, use `mem`
instead of `memory`, send an unknown `priority`, or omit fields entirely. This function
coerces every field to a safe value (clamped floats for cpu/memory, a valid lowercase
priority defaulting to `balanced`, a latency string) so no downstream arithmetic can
throw. It runs on every job both at enqueue time and again as each job is popped.

### 4.3 `compute_state()`
Advances a pointer through the datasets and returns the current environment snapshot:

| field | source | meaning |
|-------|--------|---------|
| `load` | task trace (cpu+mem util) | current system load, 0–1 |
| `energy_price` | price series | normalised grid price, 0–1 |
| `carbon_factor` | steel dataset | grid carbon intensity, 0–1 (0 = clean) |
| `energy_factor` | steel dataset | normalised energy usage |

`carbon_factor` is derived as `0.5 · rank_pct(CO₂) + 0.5 · normalise(NSM)` — the raw
CO₂ column is ~50 % zeros, so a rank-percentile blended with an NSM (time-of-day) curve
gives a signal that varies continuously the way real grid carbon does.

### 4.4 `compute_scores(state, priority, server_loads)`
For all 9 servers, computes four normalised (0–1) sub-scores — `perf_score`,
`cost_score`, `co2_score`, `lat_score` — from the server specs, the energy model, and
the current state. Only the 3 servers in the job's **tier pool** are marked `in_pool`
and compete; the rest are excluded so green work never lands on a power-hungry box.

`final` per server is the weighted sum using the tier's profile from `core/config.py`:

| priority | perf | cost | CO₂ | lat |
|----------|:----:|:----:|:---:|:---:|
| green | 0.10 | 0.25 | 0.40 | 0.25 |
| balanced | 0.25 | 0.25 | 0.25 | 0.25 |
| performance | 0.40 | 0.10 | 0.10 | 0.40 |

### 4.5 RL agents — `rl/agents.py` (`TierAgent`)
Three agents, one per priority. Multi-action **linear Q-learning** with TD(0):

- **State (7 features):** the 3 pool servers' normalised loads, `1 − carbon_factor`,
  `energy_price`, `job_cpu`, `job_mem`.
- **Action:** which of the 3 pool servers to place the job on.
- **Q(s,a):** `W[a] · features`, with `W` a `(3, 7)` weight matrix.
- **Policy:** ε-greedy, ε decaying from 0.15 → ~0.015 over the first 2000 steps.
- **Reward:** `quality_score − overload_penalty`, where
  `overload = max(0, chosen_load − avg_pool_load) / 5`. This pushes the agent to spread
  load across the pool instead of hammering the single highest-scoring server.
- **Update:** `W[a] += lr · (reward + γ·max Q(s') − Q(s,a)) · features`.

At startup, `_init_rl_agents()` pre-trains each agent for ~500 simulated episodes with
randomised loads so the policy is reasonable before the first real job. It keeps
learning online during real scheduling.

> **Display vs. training reward.** The value shown in the UI (`reward`) is the chosen
> server's quality score (always 0–1). The internal RL reward can go negative from the
> overload penalty — that's the training signal, kept separate so the dashboard stays
> readable.

### 4.6 Explainability (XAI)
Two tiers, decoupled from scheduling for speed:

1. **Inline, rule-based** — during `/run`, every job gets an instant natural-language
   explanation (tier, dominant metric, carbon conditions, runner-up). No network.
2. **On-demand LLM** — when a job is opened in the Explainable-AI tab, the browser calls
   `POST /explain`, which asks Groq (`llama-3.3-70b-versatile`) for a richer paragraph,
   cached by `(server, priority, carbon_band)`. If no `GROQ_API_KEY` is set or the call
   fails, it returns the rule-based text. The LLM client is constructed defensively so a
   missing/invalid key can never crash the app.

---

## 5. API surface

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve the dashboard HTML |
| GET/HEAD | `/health` | Liveness + build marker (`run_accepts_body`) |
| GET | `/session` | Issue/echo a session id |
| POST | `/submit` | Enqueue a single job |
| POST | `/submit_batch` | Enqueue a list of jobs |
| POST | `/run` | Schedule the queue (jobs may be sent in the body) |
| POST | `/explain` | LLM explanation for one already-scheduled job |
| POST | `/reset` | Clear the session's state |
| GET | `/info` · `/tiers` | Server list + tier-pool assignments |
| GET | `/sustainability` | Current carbon factor + a routing recommendation |

---

## 6. Deployment

- **Local:** `uvicorn api:app` (see the README quick-start). `.env` is loaded via
  `python-dotenv` if present.
- **Docker:** `docker compose up --build` — the image pins Python 3.11 and installs from
  `requirements.txt`. Compose mounts the source and runs with `--reload` for dev.
- **Render / Railway / Heroku:** `render.yaml` and `Procfile` define a web service
  running `uvicorn api:app --host 0.0.0.0 --port $PORT`. `GROQ_API_KEY` is an optional,
  unsynced env var.

Dependencies are pinned to lower bounds (`>=`) so builds resolve reliably; the code
tolerates the range it declares.

---

## 7. Failure modes & how they're handled

| Failure | Handling |
|---------|----------|
| Free-tier instance sleeps/restarts between calls | `/run` is self-contained — jobs travel in the request body, so the queue can't be lost |
| Cold-start 502/503 while the host wakes | Frontend `apiPost()` retries with backoff and shows a "waking server" notice |
| Malformed job payload (string cpu, alias fields, bad priority) | `normalize_job()` coerces/clamps every field before scheduling |
| Missing/invalid `GROQ_API_KEY` or SDK quirk | LLM client init is wrapped; app boots and falls back to rule-based text |
| Slow LLM call | Not on the `/run` path at all; `/explain` uses a short timeout + no retries |

---

## 8. Extension points

- **Persist RL weights** (e.g. save/load `W` to disk or object storage) so learning
  survives restarts.
- **Real carbon API** — swap the steel-dataset proxy for WattTime / Electricity Maps.
- **Deeper policy** — replace linear Q-learning with a small neural policy or a
  contextual bandit if the reward signal justifies it.
- **Durable sessions** — move `_sessions` to Redis to scale beyond one process.
- **Execute placements** — connect the scheduler to a real orchestrator instead of the
  simulated 9-server cluster.
