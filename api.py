from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
from groq import Groq
import os
import random
import sys
import uuid
import numpy as np
from rl.agents import Agent
from core.config import get_weights as _cfg_weights

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# PATHS — resolved relative to this script so the server can be
# started from any working directory
# ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def _load(rel_path: str, loader):
    full = os.path.join(BASE, rel_path)
    if not os.path.exists(full):
        print(f"\n❌  Missing dataset: {full}")
        print(f"    Place the file at that path and restart the server.\n")
        sys.exit(1)
    return loader(full)

# ─────────────────────────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────────────────────────
TASKS   = _load("data/dataset_rl/task_15min_L.csv", pd.read_csv)
PRICE   = _load("data/dataset_rl/price.csv",        pd.read_csv)
SERVERS = _load("data/dataset_rl/Server_L.xlsx",    pd.read_excel)
STEEL   = _load("data/steel_industry_data.csv",     pd.read_csv)

DATA_PTR  = 0
STEEL_PTR = 0

# ─────────────────────────────────────────────────────────────────
# SESSION STORE — each visitor gets isolated queue/history
# ─────────────────────────────────────────────────────────────────
_sessions: dict = {}

def _get_session(sid: str) -> dict:
    if sid not in _sessions:
        _sessions[sid] = {"job_queue": [], "job_id_counter": 0, "submitted_ids": set()}
    return _sessions[sid]

# ─────────────────────────────────────────────────────────────────
# NORMALISE STEEL COLUMNS ONCE AT STARTUP
# ─────────────────────────────────────────────────────────────────
_ukwh_min = STEEL["Usage_kWh"].min()
_ukwh_rng = STEEL["Usage_kWh"].max() - _ukwh_min + 1e-9
_co2_min  = STEEL["CO2(tCO2)"].min()
_co2_rng  = STEEL["CO2(tCO2)"].max() - _co2_min  + 1e-9

STEEL["norm_energy"] = (STEEL["Usage_kWh"] - _ukwh_min) / _ukwh_rng
STEEL["norm_co2"]    = (STEEL["CO2(tCO2)"] - _co2_min)  / _co2_rng

# ─────────────────────────────────────────────────────────────────
# SORT SERVERS BY CPU (ascending): index 0 = weakest, -1 = strongest
# ─────────────────────────────────────────────────────────────────
SERVERS   = SERVERS.sort_values("CPU").reset_index(drop=True)
n_servers = len(SERVERS)

# capacity_tier: 0.0 = smallest server, 1.0 = largest server
SERVERS["capacity_tier"] = SERVERS.index / max(n_servers - 1, 1)

# ─────────────────────────────────────────────────────────────────
# HARD TIER POOLS — each priority competes ONLY within its pool
#
# green       → bottom third  (most energy-efficient / lowest CO2)
# balanced    → middle third  (moderate capability)
# performance → top third     (most powerful / fastest)
# ─────────────────────────────────────────────────────────────────
def get_tier_pool(priority: str) -> pd.DataFrame:
    third = max(n_servers // 3, 1)
    if priority == "green":
        pool = SERVERS.iloc[:third]
    elif priority == "performance":
        pool = SERVERS.iloc[n_servers - third:]
    else:  # balanced
        start = third
        end   = n_servers - third
        pool  = SERVERS.iloc[start:end] if end > start else SERVERS.iloc[third: third + 1]
    return pool if len(pool) > 0 else SERVERS


# ─────────────────────────────────────────────────────────────────
# SCORING WEIGHTS (perf, cost, co2, lat)
# ─────────────────────────────────────────────────────────────────
PRIORITY_WEIGHTS = {
    #                  perf   cost   co2    lat
    "green":          (0.10,  0.25,  0.40,  0.25),
    "balanced":       (0.25,  0.25,  0.25,  0.25),
    "performance":    (0.40,  0.10,  0.10,  0.40),
}

# ─────────────────────────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────────
GROQ_KEY    = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_KEY)


# ─────────────────────────────────────────────────────────────────
# RL AGENT INTEGRATION
# ─────────────────────────────────────────────────────────────────
def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))

_rl_agents: dict = {}

def _init_rl_agents():
    try:
        from rl.rl_env import CloudEnv
        env = CloudEnv(os.path.join(BASE, "data", "steel_industry_data.csv"))
        for priority in ["green", "balanced", "performance"]:
            agent = Agent(priority, lr=0.05)
            wts = _cfg_weights(priority)
            for _ in range(300):
                state = env.reset()
                # reward = green_benefit - cost_penalty - load_penalty
                reward = float(wts[0]*state[0] - wts[2]*state[3] - wts[1]*state[1])
                agent.learn(state, reward)
            _rl_agents[priority] = agent
        print(f"RL agents trained for {list(_rl_agents.keys())}")
    except Exception as e:
        print(f"RL agent training skipped: {e}")

_init_rl_agents()


# ─────────────────────────────────────────────────────────────────
# SESSION INIT
# ─────────────────────────────────────────────────────────────────
@app.get("/session")
def new_session():
    sid = str(uuid.uuid4())
    _get_session(sid)
    return {"session_id": sid}


# ─────────────────────────────────────────────────────────────────
# JOB SUBMISSION
# ─────────────────────────────────────────────────────────────────
@app.post("/submit")
def submit_job(job: dict, request: Request):
    sid = request.headers.get("X-Session-ID", "default")
    s = _get_session(sid)
    s["job_id_counter"] += 1
    new_id = s["job_id_counter"]
    job["job_id"] = new_id
    s["submitted_ids"].add(new_id)
    s["job_queue"].append(job)
    return {"job_id": new_id}


# ─────────────────────────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────────────────────────
@app.post("/reset")
def reset_state(request: Request):
    global DATA_PTR, STEEL_PTR
    sid = request.headers.get("X-Session-ID", "default")
    s = _get_session(sid)
    s["job_queue"]      = []
    s["submitted_ids"]  = set()
    s["job_id_counter"] = 0
    DATA_PTR = 0
    STEEL_PTR = 0
    return {"status": "reset ok"}


# ─────────────────────────────────────────────────────────────────
# BUILD SYSTEM STATE FROM DATASETS
# ─────────────────────────────────────────────────────────────────
def compute_state() -> dict:
    global DATA_PTR, STEEL_PTR
    task  = TASKS.iloc[DATA_PTR  % len(TASKS)]
    price = PRICE.iloc[DATA_PTR  % len(PRICE)]
    steel = STEEL.iloc[STEEL_PTR % len(STEEL)]
    DATA_PTR  += 1
    STEEL_PTR += 1
    cpu_util = float(task.get("plan_cpu_i",   50)) / 100.0
    mem_util = float(task.get("plan_mem_int", 50)) / 100.0
    load     = min(max((cpu_util + mem_util) / 2.0, 0.0), 1.0)
    return {
        "load":             round(load, 4),
        "energy_price":     round(float(price.get("price_1", 50)) / 100.0, 4),
        "carbon_factor":    round(float(steel["norm_co2"]),    4),
        "energy_factor":    round(float(steel["norm_energy"]), 4),
        "steel_energy_kwh": round(float(steel["Usage_kWh"]),   4),
        "steel_co2":        round(float(steel["CO2(tCO2)"]),   6),
        "load_type":        str(steel["Load_Type"]),
        "week_status":      str(steel["WeekStatus"]),
        "day_of_week":      str(steel["Day_of_week"]),
    }


# ─────────────────────────────────────────────────────────────────
# POWER MODEL
# ─────────────────────────────────────────────────────────────────
def server_power(row, load: float) -> float:
    return row["P_idle"] + (row["P_peak"] - row["P_idle"]) * load


# ─────────────────────────────────────────────────────────────────
# MIN-MAX NORMALISE within a list
# ─────────────────────────────────────────────────────────────────
def minmax(values: list) -> list:
    mn, mx = min(values), max(values)
    rng = mx - mn + 1e-9
    return [(v - mn) / rng for v in values]


# ─────────────────────────────────────────────────────────────────
# COMPUTE SCORES
# ─────────────────────────────────────────────────────────────────
def compute_scores(state: dict, priority: str) -> tuple[dict, dict, dict]:
    w        = PRIORITY_WEIGHTS[priority]
    pool_df  = get_tier_pool(priority)
    pool_pos = set(pool_df.index.tolist())

    server_ids = []
    raw_perfs, raw_costs, raw_co2s, raw_lats, raw_kwhs = [], [], [], [], []

    for pos, row in SERVERS.iterrows():
        dc   = f"Server {int(row['ID'])}"
        perf = (row["CPU"] * row["cpu_rate"] + row["GPU"] * row["gpu_rate"]) / 1000.0
        pwr  = server_power(row, state["load"])
        ekwh = pwr / 1000.0
        cost = ekwh * state["energy_price"]
        co2  = ekwh * (0.35 + state["carbon_factor"])
        lat  = 1.0 / (row["cpu_rate"] + 1e-9)
        server_ids.append(dc)
        raw_perfs.append(perf)
        raw_costs.append(cost)
        raw_co2s.append(co2)
        raw_lats.append(lat)
        raw_kwhs.append(ekwh)

    pool_positions = [i for i, pos in enumerate(SERVERS.index) if pos in pool_pos]

    def pool_vals(raw):
        return [raw[i] for i in pool_positions]

    nm_perf = minmax(pool_vals(raw_perfs))
    nm_cost = [1.0 - v for v in minmax(pool_vals(raw_costs))]
    nm_co2  = [1.0 - v for v in minmax(pool_vals(raw_co2s))]
    nm_lat  = [1.0 - v for v in minmax(pool_vals(raw_lats))]

    pool_norm = {}
    for rank, srv_i in enumerate(pool_positions):
        pool_norm[server_ids[srv_i]] = (
            nm_perf[rank], nm_cost[rank], nm_co2[rank], nm_lat[rank]
        )

    scores, metrics, score_breakdown = {}, {}, {}

    for i, dc in enumerate(server_ids):
        in_pool = (SERVERS.index[i] in pool_pos)

        if in_pool:
            ps, cs, gs, ls = pool_norm[dc]
            base  = ps * w[0] + cs * w[1] + gs * w[2] + ls * w[3]
            noise = random.uniform(-0.03, 0.03)
            score = round(min(max(base + noise, 0.0), 1.0), 4)
        else:
            ps = cs = gs = ls = 0.0
            score = 0.0

        scores[dc]  = score
        metrics[dc] = {
            "energy_kwh":    round(raw_kwhs[i],  6),
            "cost":          round(raw_costs[i], 6),
            "co2":           round(raw_co2s[i],  6),
            "capacity_tier": round(SERVERS.iloc[i]["capacity_tier"], 4),
        }
        score_breakdown[dc] = {
            "perf_score":  round(ps, 4),
            "cost_score":  round(cs, 4),
            "co2_score":   round(gs, 4),
            "lat_score":   round(ls, 4),
            "capacity_score": round(SERVERS.iloc[i]["capacity_tier"], 4),
            "final":       round(score, 4),
            "in_pool":     in_pool,
        }

    # ─────────────────────────────────────────────────────────────
    # RL AGENT BLENDING — apply after main scoring loop
    # Blend: 75% static, 25% RL-informed
    # ─────────────────────────────────────────────────────────────
    if _rl_agents and priority in _rl_agents:
        state_vec = np.array([
            1.0 - state['carbon_factor'],
            state['load'],
            0.5,
            state['energy_price'],
        ], dtype=np.float32)
        rl_raw = _rl_agents[priority].act(state_vec)
        rl_score = _sigmoid(rl_raw)
        for i, dc in enumerate(server_ids):
            in_pool = score_breakdown[dc]['in_pool']
            if in_pool:
                score = round(min(max(0.75 * scores[dc] + 0.25 * rl_score, 0.0), 1.0), 4)
                scores[dc] = score
                score_breakdown[dc]['final'] = score
                score_breakdown[dc]['rl_score'] = round(rl_score, 4)
            else:
                score_breakdown[dc]['rl_score'] = 0.0
    else:
        for dc in server_ids:
            score_breakdown[dc]['rl_score'] = 0.0

    return scores, metrics, score_breakdown


# ─────────────────────────────────────────────────────────────────
# XAI – LLM EXPLANATION VIA GROQ
# ─────────────────────────────────────────────────────────────────
def generate_explanation(job, state, chosen_dc, scores, breakdown, metrics) -> str:
    priority   = job["priority"]
    w          = PRIORITY_WEIGHTS[priority]
    pool_names = [
        f"Server {int(row['ID'])}"
        for _, row in get_tier_pool(priority).iterrows()
    ]
    cm = metrics[chosen_dc]

    bd_text = "\n".join(
        f"  {dc}: perf={b['perf_score']:.3f} cost={b['cost_score']:.3f} "
        f"co2={b['co2_score']:.3f} lat={b['lat_score']:.3f} "
        f"in_pool={b['in_pool']} final={b['final']:.3f}"
        for dc, b in sorted(breakdown.items(), key=lambda x: x[1]["final"], reverse=True)
    )

    prompt = f"""You are an XAI assistant for a sustainable cloud job scheduler.

The scheduler uses TIER-POOL routing: each priority maps to a server tier.
  green       → small/efficient servers  (lowest CO2 + cost)
  balanced    → mid-tier servers         (moderate capability)
  performance → large/powerful servers   (highest throughput)

Eligible pool for this job: {pool_names}
Only pool servers compete; others are excluded for sustainability reasons.

Job: priority={priority}, latency={job.get('latency','N/A')}
State: load={state['load']:.3f}, energy_price={state['energy_price']:.4f}, carbon={state['carbon_factor']:.4f}
Weights: perf×{w[0]}, cost×{w[1]}, co2×{w[2]}, lat×{w[3]}

Scores:
{bd_text}

Chosen: {chosen_dc} | energy={cm['energy_kwh']:.4f}kWh cost={cm['cost']:.6f} co2={cm['co2']:.6f} tier={cm['capacity_tier']:.2f}

In 3-4 sentences explain why {chosen_dc} won. Mention tier routing logic,
the dominant metric, and current system conditions. Do not greet or start with I."""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
            timeout=10,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        bd = breakdown[chosen_dc]
        ru = sorted(
            [(k, v["final"]) for k, v in breakdown.items()
             if k != chosen_dc and v["in_pool"]],
            key=lambda x: x[1], reverse=True,
        )
        ru_str = f", runner-up: {ru[0][0]} @ {ru[0][1]:.3f}" if ru else ""
        return (
            f"{chosen_dc} selected from pool {pool_names} for '{priority}' job"
            f"(score: {scores[chosen_dc]:.3f}{ru_str}). "
            f"CO2={bd['co2_score']:.3f}, cost={bd['cost_score']:.3f}, "
            f"perf={bd['perf_score']:.3f}, lat={bd['lat_score']:.3f}. "
            f"[Groq error: {str(e)[:80]}]"
        )


# ─────────────────────────────────────────────────────────────────
# SCHEDULER ENDPOINT
# ─────────────────────────────────────────────────────────────────
@app.post("/run")
def run_scheduler(request: Request):
    sid = request.headers.get("X-Session-ID", "default")
    s = _get_session(sid)
    scheduled_jobs = []
    while s["job_queue"]:
        job   = s["job_queue"].pop(0)
        state = compute_state()
        scores, metrics, breakdown = compute_scores(state, job["priority"])

        pool_scores = {dc: sc for dc, sc in scores.items() if breakdown[dc]["in_pool"]}
        chosen_dc   = max(pool_scores, key=pool_scores.get)
        explanation = generate_explanation(job, state, chosen_dc, scores, breakdown, metrics)

        scheduled_jobs.append({
            "job_id":          job["job_id"],
            "chosen_dc":       chosen_dc,
            "scores":          scores,
            "score_breakdown": breakdown,
            "reward":          scores[chosen_dc],
            "priority":        job["priority"],
            "latency":         job.get("latency", "N/A"),
            "state":           state,
            "power_kwh":       metrics[chosen_dc]["energy_kwh"],
            "cost":            metrics[chosen_dc]["cost"],
            "co2":             metrics[chosen_dc]["co2"],
            "all_metrics":     metrics,
            "explanation":     explanation,
        })
    return {"scheduled_jobs": scheduled_jobs}


# ─────────────────────────────────────────────────────────────────
# INFO — exposes n_servers + pool assignments to frontend
# ─────────────────────────────────────────────────────────────────
@app.get("/info")
def get_info():
    pools = {
        p: [f"Server {int(row['ID'])}" for _, row in get_tier_pool(p).iterrows()]
        for p in ["green", "balanced", "performance"]
    }
    return {
        "n_servers":  n_servers,
        "server_ids": [f"Server {int(row['ID'])}" for _, row in SERVERS.iterrows()],
        "tier_pools": pools,
    }


# ─────────────────────────────────────────────────────────────────
# DEBUG: view tier pool assignments
# ─────────────────────────────────────────────────────────────────
@app.get("/tiers")
def show_tiers():
    green_pool = [f"Server {int(r['ID'])}" for _, r in get_tier_pool("green").iterrows()]
    bal_pool   = [f"Server {int(r['ID'])}" for _, r in get_tier_pool("balanced").iterrows()]
    perf_pool  = [f"Server {int(r['ID'])}" for _, r in get_tier_pool("performance").iterrows()]
    result = {}
    for _, row in SERVERS.iterrows():
        dc = f"Server {int(row['ID'])}"
        result[dc] = {
            "capacity_tier":        round(row["capacity_tier"], 4),
            "in_green_pool":        dc in green_pool,
            "in_balanced_pool":     dc in bal_pool,
            "in_performance_pool":  dc in perf_pool,
        }
    return {"server_tiers": result, "total_servers": n_servers}


# ─────────────────────────────────────────────────────────────────
# SUSTAINABILITY HINT
# ─────────────────────────────────────────────────────────────────
@app.get("/sustainability")
def get_sustainability_hint():
    """Returns current carbon factor and scheduling recommendation."""
    try:
        steel = STEEL.iloc[STEEL_PTR % len(STEEL)]
        cf = float(steel["norm_co2"])
        if cf < 0.33:
            rec = "performance"
            msg = "Low carbon intensity — ideal for any workload"
        elif cf < 0.66:
            rec = "balanced"
            msg = "Moderate carbon intensity — prefer balanced jobs"
        else:
            rec = "green"
            msg = "High carbon intensity — prioritize green jobs"
        return {"carbon_factor": round(cf, 4), "recommendation": rec, "message": msg}
    except Exception:
        return {"carbon_factor": 0.5, "recommendation": "balanced", "message": "Unable to read carbon data"}


# ─────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────
_DASHBOARD = os.path.join(BASE, "frontend", "templates", "dashboard.html")

@app.get("/")
def root():
    return HTMLResponse(open(_DASHBOARD, encoding="utf-8").read())

@app.get("/health")
def health():
    return {"message": "GreenMind API — tier-pool routing active"}
