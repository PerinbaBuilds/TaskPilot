# Software Requirements Specification

**Project:** TaskPilot — Sustainable Cloud Job Scheduler  
**Version:** 1.0  
**Date:** 2026-06-16  
**Author:** PerinbaBuilds

---

## 1. Introduction

### 1.1 Purpose

This document defines the functional and non-functional requirements for TaskPilot, a reinforcement learning-powered cloud job scheduler that minimises carbon emissions while maintaining performance and cost efficiency.

### 1.2 Scope

TaskPilot is a web application that:

- Accepts compute job submissions (manual or CSV bulk upload)
- Routes each job to an optimal server using RL-based scoring
- Displays scheduling results, sustainability metrics, and AI-generated explanations
- Provides analytics charts covering reward trends, energy usage, server distribution, and carbon intensity

### 1.3 Definitions

| Term | Definition |
|------|-----------|
| Job | A compute task described by CPU requirement, memory, priority tier, and latency class |
| Priority tier | One of: `green`, `balanced`, `performance` — determines which server pool is eligible |
| TierAgent | An RL agent (one per priority tier) that selects a server within the tier's pool |
| Carbon factor | Normalised grid carbon intensity (0 = clean, 1 = dirty), derived from the Steel Industry Dataset |
| Sustainability score | Composite KPI (0–100) combining green ratio, CO2 efficiency, and reward quality |
| Session | Browser-isolated context identified by a UUID in `localStorage`; passed as `X-Session-ID` header |

---

## 2. Overall Description

### 2.1 Product Perspective

TaskPilot is a standalone web application. It does not integrate with a live cloud provider; it simulates a 9-server cluster using real workload and energy datasets. It is deployable on any platform supporting Python + Uvicorn (Render, Railway, Heroku, Docker).

### 2.2 User Classes

| Class | Description |
|-------|-------------|
| End user | Schedules jobs, reviews results, reads XAI explanations via the browser dashboard |
| Developer | Deploys and configures the backend; may extend RL agents or scoring weights |

### 2.3 Constraints

- Python 3.10 or higher
- Free-tier deployment (Render): single process, no persistent storage between restarts
- Groq API key is optional; XAI degrades gracefully to rule-based explanations without it
- No database — all session state is in-memory

---

## 3. Functional Requirements

### 3.1 Job Submission

| ID | Requirement |
|----|-------------|
| FR-01 | The system shall accept a single job with fields: `priority`, `latency`, `cpu`, `memory` |
| FR-02 | The system shall accept a batch of jobs as a JSON array via `POST /submit_batch` |
| FR-03 | The system shall accept CSV file uploads and parse them client-side before submission |
| FR-04 | The system shall assign a unique integer job ID to each submitted job within the session |
| FR-05 | CSV files shall support columns: `priority`, `latency`, `cpu`, `memory` (required); `estimated_kwh`, `co2_kg` (optional) |

### 3.2 Scheduling

| ID | Requirement |
|----|-------------|
| FR-10 | The system shall assign each job to a server within the tier pool matching the job's priority |
| FR-11 | Server selection shall be driven by a TierAgent using epsilon-greedy action selection |
| FR-12 | The agent shall build a 7-feature state vector: `[s0_load, s1_load, s2_load, carbon_factor, energy_price, job_cpu, job_mem]` |
| FR-13 | The agent shall perform a TD(0) weight update after every scheduled job |
| FR-14 | The system shall track per-server load counts within each session |
| FR-15 | The system shall return per-job results including: `chosen_dc`, `reward`, `power_kwh`, `co2`, `cost`, `state`, `scores`, `breakdown`, `explanation` |

### 3.3 Scoring

| ID | Requirement |
|----|-------------|
| FR-20 | The system shall compute normalised scores for throughput, cost, CO2, and latency for every server in the tier pool |
| FR-21 | Scores shall be normalised to [0, 1] across the servers in the active pool |
| FR-22 | The system shall apply priority-specific weights to produce a quality score |
| FR-23 | The reward field returned to the client shall be the quality score (0–1), not the internal RL reward signal |

### 3.4 Explainability (XAI)

| ID | Requirement |
|----|-------------|
| FR-30 | The system shall generate a plain-English explanation for every scheduled job |
| FR-31 | Explanations shall be cached by `(server, priority, carbon_band)` key to limit Groq API calls |
| FR-32 | If the Groq API is unavailable or not configured, the system shall fall back to a deterministic natural-language template |
| FR-33 | The explanation shall reference the priority tier, dominant scoring metric, grid carbon conditions, and competing servers |

### 3.5 Sustainability Metrics

| ID | Requirement |
|----|-------------|
| FR-40 | The system shall compute a session-level sustainability score (0–100) after every run |
| FR-41 | The score shall weight: avg_reward (30%), green_ratio (25%), carbon_score (25%), co2_efficiency (20%) |
| FR-42 | The system shall expose a `/sustainability` endpoint returning the current carbon factor and a scheduling recommendation |

### 3.6 Dashboard

| ID | Requirement |
|----|-------------|
| FR-50 | The dashboard shall have four tabs: Overview, History, Analytics, XAI |
| FR-51 | The Overview tab shall display sustainability score, KPI cards, a reward timeline chart, and server distribution donut |
| FR-52 | The History tab shall show a paginated table of all scheduled jobs in the current session |
| FR-53 | The Analytics tab shall show: avg reward per server, energy & CO2 by server, priority routing chart, energy intensity distribution |
| FR-54 | The XAI tab shall list all scheduled jobs and display the full explanation for the selected job |
| FR-55 | Clicking the TaskPilot logo shall navigate to the Overview tab |

### 3.7 Session Isolation

| ID | Requirement |
|----|-------------|
| FR-60 | Each browser session shall be assigned a UUID stored in `localStorage` |
| FR-61 | All API calls shall include the UUID as the `X-Session-ID` request header |
| FR-62 | Session data (queue, history, server loads) shall be fully isolated between sessions |
| FR-63 | The `/reset` endpoint shall clear the current session's state |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement |
|----|-------------|
| NFR-01 | The scheduler shall process a 100-job batch in under 30 seconds on Render free tier |
| NFR-02 | XAI cache shall ensure no more than ~9 Groq API calls are made per 100-job batch |
| NFR-03 | RL agent startup pre-training (500 episodes) shall complete before the first request is served |

### 4.2 Reliability

| ID | Requirement |
|----|-------------|
| NFR-10 | The system shall function fully without a Groq API key (XAI degrades gracefully) |
| NFR-11 | Missing or malformed CSV columns shall be handled without crashing the scheduler |
| NFR-12 | The application shall restart cleanly after any unhandled exception at the process level |

### 4.3 Security

| ID | Requirement |
|----|-------------|
| NFR-20 | The Groq API key shall only be read from the `GROQ_API_KEY` environment variable; it shall never be hardcoded |
| NFR-21 | The application shall not expose internal session data across different session IDs |

### 4.4 Maintainability

| ID | Requirement |
|----|-------------|
| NFR-30 | Priority weights shall be defined in a single location (`core/config.py`) |
| NFR-31 | RL agent logic shall be fully contained in `rl/agents.py`, independent of the API layer |

---

## 5. External Interfaces

### 5.1 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the dashboard HTML |
| POST | `/submit_batch` | Enqueue a list of jobs |
| POST | `/run` | Schedule all queued jobs and return results |
| POST | `/reset` | Clear session state |
| GET | `/servers` | Return server tier assignments |
| GET | `/sustainability` | Return current carbon factor and recommendation |

### 5.2 External Services

| Service | Usage | Fallback |
|---------|-------|----------|
| Groq API (`llama-3.3-70b-versatile`) | Generate XAI explanations | Deterministic natural-language template |

### 5.3 Datasets

| File | Source | Usage |
|------|--------|-------|
| `steel_industry_data.csv` | UCI ML Repository | Carbon factor and energy price signal |
| `dataset_rl/Server_L.xlsx` | Synthetic | Server specifications (CPU, GPU, power) |
| `dataset_rl/task_15min_L.csv` | Synthetic | Workload traces (cpu/mem utilisation) |
