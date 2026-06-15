import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
from io import StringIO

st.set_page_config(
    page_title="GreenMind AI Scheduler",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://127.0.0.1:8000"

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS — green-branded theme
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

/* ── Header banner ── */
.gm-header {
    background: linear-gradient(135deg, #0d4f2e 0%, #1a7a4a 50%, #0d4f2e 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    border: 1px solid #2ea043;
    display: flex;
    align-items: center;
    gap: 16px;
}
.gm-header h1 {
    color: #ffffff;
    margin: 0;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.gm-header p {
    color: #7ee787;
    margin: 4px 0 0 0;
    font-size: 0.85rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #2ea043; }
.metric-card .label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
}
.metric-card .unit {
    font-size: 0.75rem;
    color: #7ee787;
    margin-top: 2px;
}

/* ── Sustainability ring ── */
.sustain-ring {
    background: radial-gradient(circle, #0d4f2e 0%, #0f1117 70%);
    border: 2px solid #2ea043;
    border-radius: 50%;
    width: 120px;
    height: 120px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: auto;
}
.sustain-ring .score { font-size: 2.2rem; font-weight: 800; color: #2ea043; }
.sustain-ring .lbl   { font-size: 0.65rem; color: #7ee787; margin-top: -4px; }

/* ── Pool badge ── */
.pool-green   { background:#0d4f2e; color:#7ee787; border:1px solid #2ea043; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
.pool-balanced{ background:#2d1f00; color:#f0b030; border:1px solid #d29922; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
.pool-perf    { background:#3d0000; color:#ff7b72; border:1px solid #f85149; border-radius:6px; padding:2px 8px; font-size:0.75rem; }

/* ── Server tile ── */
.server-tile {
    border-radius: 8px;
    padding: 10px 12px;
    margin: 4px 0;
    font-size: 0.8rem;
    font-weight: 600;
    text-align: center;
}
.tile-green   { background:#0d3b22; border:1px solid #2ea043; color:#7ee787; }
.tile-balanced{ background:#2a1f00; border:1px solid #d29922; color:#f0b030; }
.tile-perf    { background:#3a0f0f; border:1px solid #f85149; color:#ff7b72; }

/* ── Carbon alert ── */
.carbon-low  { background:#0d3b22; border:1px solid #2ea043; border-radius:8px; padding:10px 14px; color:#7ee787; font-size:0.85rem; }
.carbon-med  { background:#2a1f00; border:1px solid #d29922; border-radius:8px; padding:10px 14px; color:#f0b030; font-size:0.85rem; }
.carbon-high { background:#3a0f0f; border:1px solid #f85149; border-radius:8px; padding:10px 14px; color:#ff7b72; font-size:0.85rem; }

/* ── Job result card ── */
.job-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.job-card.chosen { border-color: #2ea043; background: #0d2b1a; }

/* ── Tab styling ── */
button[data-baseweb="tab"] {
    font-size: 0.85rem !important;
}

/* ── Sidebar ── */
.sidebar-section {
    background: #0f1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
}

/* ── Template button row ── */
.tmpl-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
for key, default in [
    ("queue", []),
    ("history", []),
    ("last_run", None),
    ("csv_upload_key", 0),
    ("csv_processed", False),
    ("compare_a", None),
    ("compare_b", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────
# FETCH SERVER INFO
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_server_info():
    try:
        r = requests.get(f"{API_BASE}/info", timeout=5)
        r.raise_for_status()
        d = r.json()
        return d["n_servers"], d["server_ids"], d.get("tier_pools", {})
    except Exception:
        return 0, [], {}

@st.cache_data(ttl=30)
def fetch_latest_state():
    """Fetch one state snapshot from the API for carbon-aware suggestions."""
    try:
        r = requests.get(f"{API_BASE}/state_preview", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

n_servers, all_server_ids, tier_pools = fetch_server_info()

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
PRIORITY_EMOJI = {"green": "🌿", "balanced": "⚖️", "performance": "⚡"}
PRIORITY_COLOR = {"green": "#2ea043", "balanced": "#d29922", "performance": "#f85149"}

def sustainability_score(jobs: list) -> float:
    """0-100 composite: lower CO2/cost + higher reward = better."""
    if not jobs:
        return 0.0
    avg_reward = sum(j["reward"] for j in jobs) / len(jobs)
    avg_co2    = sum(j["co2"]    for j in jobs) / len(jobs)
    co2_score  = max(0.0, 1.0 - avg_co2 * 1000)
    return round((avg_reward * 0.6 + co2_score * 0.4) * 100, 1)

def carbon_label(cf: float):
    if cf < 0.33:
        return "🟢 Low carbon intensity — great time for any workload", "carbon-low"
    elif cf < 0.66:
        return "🟡 Moderate carbon intensity — prefer green or balanced jobs", "carbon-med"
    else:
        return "🔴 High carbon intensity — defer performance jobs if possible", "carbon-high"

def recommend_priority(cf: float) -> str:
    if cf < 0.33:
        return "performance"
    elif cf < 0.66:
        return "balanced"
    return "green"

def make_radar(breakdown: dict, chosen: str):
    metrics = ["perf_score", "cost_score", "co2_score", "lat_score"]
    labels  = ["Performance", "Cost\nEfficiency", "CO₂\nScore", "Latency"]
    N       = len(metrics)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")

    colors = {"chosen": "#2ea043", "other": "#30363d"}
    for dc, bd in breakdown.items():
        if not bd.get("in_pool"):
            continue
        vals = [bd[m] for m in metrics] + [bd[metrics[0]]]
        color = "#2ea043" if dc == chosen else "#58a6ff"
        lw    = 2.5 if dc == chosen else 1.2
        alpha = 0.25 if dc == chosen else 0.05
        ax.plot(angles, vals, color=color, linewidth=lw, label=dc)
        ax.fill(angles, vals, color=color, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#8b949e", fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="#30363d", fontsize=7)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#8b949e")
    ax.spines["polar"].set_color("#21262d")
    ax.grid(color="#21262d", linewidth=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=8, framealpha=0, labelcolor="#e6edf3")
    ax.set_title("Score Radar", color="#e6edf3", pad=20, fontsize=10)
    return fig

def export_csv(jobs: list) -> str:
    rows = []
    for j in jobs:
        rows.append({
            "job_id":    j["job_id"],
            "chosen_dc": j["chosen_dc"],
            "priority":  j["priority"],
            "latency":   j["latency"],
            "reward":    j["reward"],
            "power_kwh": j["power_kwh"],
            "cost":      j["cost"],
            "co2":       j["co2"],
            "load":      j["state"]["load"],
            "carbon_factor": j["state"]["carbon_factor"],
            "load_type": j["state"]["load_type"],
            "day":       j["state"]["day_of_week"],
        })
    return pd.DataFrame(rows).to_csv(index=False)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 GreenMind")

    # ── Carbon-aware suggestion ──────────────────────────────────
    if st.session_state.history:
        last_cf = st.session_state.history[-1]["state"]["carbon_factor"]
        msg, cls = carbon_label(last_cf)
        rec = recommend_priority(last_cf)
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)
        st.caption(f"Suggested priority: **{rec}**")
        st.markdown("")

    # ── Server tier map ─────────────────────────────────────────
    if tier_pools:
        with st.expander("🗂 Server Tier Map", expanded=False):
            tile_cls = {"green": "tile-green", "balanced": "tile-balanced", "performance": "tile-perf"}
            for p, servers in tier_pools.items():
                st.markdown(f"**{PRIORITY_EMOJI.get(p,'')} {p.capitalize()}**")
                cols = st.columns(len(servers))
                for i, s in enumerate(servers):
                    cols[i].markdown(
                        f'<div class="server-tile {tile_cls[p]}">{s}</div>',
                        unsafe_allow_html=True,
                    )
        st.markdown("")

    # ── Add Single Job ───────────────────────────────────────────
    st.markdown("### Add Job")

    # Quick templates
    st.caption("Quick templates:")
    t1, t2, t3 = st.columns(3)
    if t1.button("🌿 Green", use_container_width=True):
        st.session_state["_tmpl"] = ("green", "low")
    if t2.button("⚖️ Balanced", use_container_width=True):
        st.session_state["_tmpl"] = ("balanced", "medium")
    if t3.button("⚡ Perf", use_container_width=True):
        st.session_state["_tmpl"] = ("performance", "high")

    tmpl = st.session_state.pop("_tmpl", None)
    default_priority = tmpl[0] if tmpl else "balanced"
    default_latency  = tmpl[1] if tmpl else "medium"

    priority_opts = ["balanced", "green", "performance"]
    latency_opts  = ["low", "medium", "high"]
    priority = st.selectbox("Priority", priority_opts,
                            index=priority_opts.index(default_priority))
    latency  = st.selectbox("Latency",  latency_opts,
                            index=latency_opts.index(default_latency))
    col1, col2 = st.columns(2)
    cpu = col1.slider("CPU %",    0, 100, 50)
    mem = col2.slider("Memory %", 0, 100, 50)

    if tier_pools.get(priority):
        pool_servers = tier_pools[priority]
        cls = {"green": "pool-green", "balanced": "pool-balanced", "performance": "pool-perf"}[priority]
        st.markdown(
            f'Routed to: ' +
            " ".join(f'<span class="{cls}">{s}</span>' for s in pool_servers),
            unsafe_allow_html=True,
        )

    if st.button("➕ Add Job", type="primary", use_container_width=True):
        try:
            r = requests.post(
                f"{API_BASE}/submit",
                json={"priority": priority, "latency": latency},
                timeout=10,
            )
            r.raise_for_status()
            resp = r.json()
            if resp.get("status") == "already_submitted":
                st.warning("Job already in queue")
            else:
                st.session_state.queue.append({
                    "job_id": resp["job_id"], "priority": priority,
                    "latency": latency, "cpu": cpu, "mem": mem,
                })
                st.success(f"✅ Job {resp['job_id']} added")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.markdown("---")

    # ── Bulk CSV Upload ──────────────────────────────────────────
    st.markdown("### Bulk Upload")
    uploaded = st.file_uploader(
        "Upload jobs.csv",
        type="csv",
        key=f"csv_uploader_{st.session_state.csv_upload_key}",
        help="Columns: priority, latency (optional: cpu, mem)",
    )
    if uploaded and not st.session_state.csv_processed:
        st.session_state.csv_processed = True
        try:
            df_up    = pd.read_csv(uploaded)
            added, skipped = 0, 0
            progress = st.progress(0)
            for idx, row in df_up.iterrows():
                payload = {
                    "priority": str(row.get("priority", "balanced")),
                    "latency":  str(row.get("latency",  "medium")),
                }
                r = requests.post(f"{API_BASE}/submit", json=payload, timeout=10)
                if r.status_code == 200:
                    resp = r.json()
                    if resp.get("status") == "already_submitted":
                        skipped += 1
                    else:
                        st.session_state.queue.append({
                            "job_id": resp["job_id"], "priority": payload["priority"],
                            "latency": payload["latency"],
                            "cpu": row.get("cpu"), "mem": row.get("mem"),
                        })
                        added += 1
                progress.progress((idx + 1) / len(df_up))
            msg = f"✅ Added {added} jobs"
            if skipped:
                msg += f" ({skipped} skipped)"
            st.success(msg)
            st.session_state.csv_upload_key += 1
        except Exception as e:
            st.error(f"Upload failed: {e}")
            st.session_state.csv_processed = False
    elif not uploaded:
        st.session_state.csv_processed = False

    st.markdown("---")

    # ── Controls ─────────────────────────────────────────────────
    queue_len = len(st.session_state.queue)
    st.markdown(f"**Queue:** {queue_len} job{'s' if queue_len != 1 else ''} pending")

    if st.button("▶ Run Scheduler", type="primary", use_container_width=True,
                 disabled=(queue_len == 0)):
        with st.spinner(f"Scheduling {queue_len} job{'s' if queue_len != 1 else ''}…"):
            try:
                r = requests.post(f"{API_BASE}/run", timeout=300)
                r.raise_for_status()
                scheduled = r.json().get("scheduled_jobs", [])
                if scheduled:
                    st.session_state.history.extend(scheduled)
                    st.session_state.last_run = scheduled
                    st.session_state.queue    = []
                    st.success(f"✅ Processed {len(scheduled)} jobs")
                else:
                    st.info("No jobs processed")
            except Exception as e:
                st.error(f"Scheduler failed: {e}")

    if st.session_state.last_run:
        csv_data = export_csv(st.session_state.last_run)
        st.download_button(
            "⬇ Export Last Run (CSV)",
            data=csv_data,
            file_name="greenmind_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.button("🗑 Clear History", type="secondary", use_container_width=True):
        try:
            requests.post(f"{API_BASE}/reset", timeout=10)
        except Exception:
            pass
        st.session_state.history         = []
        st.session_state.last_run        = None
        st.session_state.queue           = []
        st.session_state.csv_processed   = False
        st.session_state.csv_upload_key += 1
        st.success("Cleared")

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gm-header">
  <div style="font-size:2.5rem">🌱</div>
  <div>
    <h1>GreenMind AI Scheduler</h1>
    <p>Sustainable cloud job scheduling · Tier-pool routing · Explainable AI · Reinforcement Learning</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Global KPI strip ─────────────────────────────────────────────
if st.session_state.history:
    h = st.session_state.history
    ss = sustainability_score(h)
    total_jobs = len(h)
    avg_reward = sum(j["reward"] for j in h) / total_jobs
    total_co2  = sum(j["co2"]    for j in h)
    total_kwh  = sum(j["power_kwh"] for j in h)
    green_jobs = sum(1 for j in h if j["priority"] == "green")
    green_pct  = green_jobs / total_jobs * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, value, unit in [
        (k1, "Sustainability",  f"{ss}",          "/ 100"),
        (k2, "Jobs Scheduled",  f"{total_jobs}",  "total"),
        (k3, "Avg Reward",      f"{avg_reward:.3f}", "score"),
        (k4, "Total Energy",    f"{total_kwh:.3f}", "kWh"),
        (k5, "Green Jobs",      f"{green_pct:.0f}%", f"{green_jobs} jobs"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="unit">{unit}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("")

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab_queue, tab_results, tab_xai, tab_compare, tab_analytics, tab_sustain = st.tabs([
    "📋 Queue",
    "📊 Results",
    "🧠 Explainable AI",
    "🔍 Compare",
    "📈 Analytics",
    "🌍 Sustainability",
])

# ═════════════════════════════════════════════════════════════════
# QUEUE TAB
# ═════════════════════════════════════════════════════════════════
with tab_queue:
    st.subheader("Pending Jobs")
    if st.session_state.queue:
        dfq = pd.DataFrame(st.session_state.queue)
        st.dataframe(dfq, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Priority Mix**")
            st.bar_chart(dfq["priority"].value_counts())
        with col2:
            st.markdown("**Latency Mix**")
            st.bar_chart(dfq["latency"].value_counts())
        with col3:
            if "cpu" in dfq.columns and dfq["cpu"].notna().any():
                st.markdown("**CPU Demand Distribution**")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.set_facecolor("#0f1117"); fig.patch.set_facecolor("#0f1117")
                ax.hist(dfq["cpu"].dropna(), bins=10, color="#2ea043", edgecolor="#0f1117")
                ax.set_xlabel("CPU %", color="#8b949e")
                ax.set_ylabel("Count",  color="#8b949e")
                ax.tick_params(colors="#8b949e")
                for spine in ax.spines.values(): spine.set_color("#21262d")
                st.pyplot(fig); plt.close(fig)
    else:
        st.markdown("""
        <div style="text-align:center; padding:60px; color:#8b949e;">
          <div style="font-size:3rem">📭</div>
          <div style="margin-top:10px">Queue is empty — add jobs via the sidebar or upload a CSV</div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# RESULTS TAB
# ═════════════════════════════════════════════════════════════════
with tab_results:
    if st.session_state.last_run:
        last = st.session_state.last_run
        total_cost = sum(j["cost"]      for j in last)
        total_co2  = sum(j["co2"]       for j in last)
        total_kwh  = sum(j["power_kwh"] for j in last)
        avg_reward = sum(j["reward"]    for j in last) / len(last)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Jobs Processed", len(last))
        r2.metric("Avg Reward",     f"{avg_reward:.3f}")
        r3.metric("Total Energy",   f"{total_kwh:.4f} kWh")
        r4.metric("Total CO₂",      f"{total_co2:.6f} tCO2")
        st.markdown("---")

        # ── Server distribution pie ──────────────────────────────
        dc_counts = {}
        for j in last:
            dc_counts[j["chosen_dc"]] = dc_counts.get(j["chosen_dc"], 0) + 1

        if len(dc_counts) > 1:
            col_pie, col_list = st.columns([1, 2])
            with col_pie:
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor("#0f1117")
                ax.set_facecolor("#0f1117")
                wedge_colors = plt.cm.Set2(np.linspace(0, 1, len(dc_counts)))
                ax.pie(
                    dc_counts.values(), labels=dc_counts.keys(),
                    colors=wedge_colors, autopct="%1.0f%%",
                    textprops={"color": "#e6edf3", "fontsize": 9},
                    wedgeprops={"linewidth": 1, "edgecolor": "#0f1117"},
                )
                ax.set_title("Job Distribution", color="#e6edf3", fontsize=10)
                st.pyplot(fig); plt.close(fig)
            with col_list:
                st.markdown("**Job Results**")
                for job in last:
                    p = job["priority"]
                    em = PRIORITY_EMOJI.get(p, "•")
                    clr = PRIORITY_COLOR.get(p, "#8b949e")
                    st.markdown(f"""
                    <div class="job-card {'chosen' if True else ''}">
                      <span style="color:{clr}">{em} <b>Job {job['job_id']}</b></span>
                      &nbsp;→&nbsp; <b>{job['chosen_dc']}</b>
                      &nbsp;|&nbsp; <span style="color:#8b949e">reward: {job['reward']:.3f}
                      &nbsp;·&nbsp; {job['power_kwh']:.4f} kWh
                      &nbsp;·&nbsp; CO₂ {job['co2']:.5f}</span>
                    </div>""", unsafe_allow_html=True)
        st.markdown("---")

        # ── Detailed expandable per-job cards ────────────────────
        st.subheader("Detailed Results")
        for job in last:
            p  = job["priority"]
            em = PRIORITY_EMOJI.get(p, "•")
            with st.expander(
                f"{em} Job {job['job_id']} → **{job['chosen_dc']}** "
                f"| {p} | reward {job['reward']:.3f}"
            ):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown("**📡 System State at Scheduling Time**")
                    state = job["state"]
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Load",          f"{state['load']:.3f}")
                    s2.metric("Energy Price",  f"{state['energy_price']:.4f}")
                    s3.metric("Carbon Factor", f"{state['carbon_factor']:.4f}")
                    s4.metric("Energy Factor", f"{state['energy_factor']:.4f}")
                    st.caption(
                        f"🕐 {state['load_type']} · {state['day_of_week']} "
                        f"({state['week_status']})"
                    )
                with col_b:
                    st.markdown("**🔋 Chosen Server Metrics**")
                    st.metric("Energy", f"{job['power_kwh']:.4f} kWh")
                    st.metric("Cost",   f"{job['cost']:.6f}")
                    st.metric("CO₂",   f"{job['co2']:.6f} tCO2")

                # Server score grid
                st.markdown("**🏆 Score Comparison**")
                in_pool  = {dc: sc for dc, sc in job["scores"].items()
                            if job["score_breakdown"][dc].get("in_pool")}
                out_pool = {dc: sc for dc, sc in job["scores"].items()
                            if not job["score_breakdown"][dc].get("in_pool")}

                if in_pool:
                    st.caption("✅ Eligible pool (competed):")
                    sc_cols = st.columns(len(in_pool))
                    for i, (dc, sc) in enumerate(
                        sorted(in_pool.items(), key=lambda x: x[1], reverse=True)
                    ):
                        with sc_cols[i]:
                            if dc == job["chosen_dc"]:
                                st.success(f"🏆 {dc}\n\n**{sc:.3f}**")
                            else:
                                st.metric(dc, f"{sc:.3f}")

                if out_pool:
                    st.caption("⛔ Excluded by tier routing:")
                    oc = st.columns(len(out_pool))
                    for i, (dc, _) in enumerate(out_pool.items()):
                        oc[i].metric(dc, "—", delta="out of tier",
                                     delta_color="off")
    else:
        st.markdown("""
        <div style="text-align:center; padding:60px; color:#8b949e;">
          <div style="font-size:3rem">📊</div>
          <div style="margin-top:10px">Run the scheduler to see results</div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# EXPLAINABLE AI TAB
# ═════════════════════════════════════════════════════════════════
with tab_xai:
    st.subheader("🧠 Why Was This Server Chosen?")

    if st.session_state.last_run:
        job_opts = {
            f"Job {j['job_id']} → {j['chosen_dc']} ({j['priority']})": j
            for j in st.session_state.last_run
        }
        sel = st.selectbox("Select a job:", list(job_opts.keys()))
        job = job_opts[sel]
        p   = job["priority"]

        # Header row
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            pcls = {"green": "pool-green", "balanced": "pool-balanced",
                    "performance": "pool-perf"}[p]
            st.markdown(
                f'### {PRIORITY_EMOJI.get(p,"")} Job {job["job_id"]} '
                f'<span class="{pcls}">{p}</span>',
                unsafe_allow_html=True,
            )
            if tier_pools.get(p):
                st.caption(
                    f"Eligible pool: {', '.join(tier_pools[p])} "
                    f"— others excluded by tier routing"
                )
        with hcol2:
            st.metric("Final Reward", f"{job['reward']:.3f}")

        # LLM explanation
        st.markdown("#### 💬 LLM Explanation")
        st.info(job["explanation"])
        st.markdown("---")

        if "score_breakdown" in job:
            breakdown = job["score_breakdown"]
            metrics_order = ["perf_score", "cost_score", "co2_score", "lat_score"]
            metric_labels = ["Performance", "Cost Efficiency", "CO₂ Score", "Latency"]

            in_pool_bd  = {dc: bd for dc, bd in breakdown.items() if bd.get("in_pool")}
            out_pool_bd = {dc: bd for dc, bd in breakdown.items() if not bd.get("in_pool")}

            # ── Radar + Bar side-by-side ─────────────────────────
            chart_left, chart_right = st.columns([1, 2])

            with chart_left:
                st.markdown("#### 🕸 Score Radar")
                if in_pool_bd:
                    fig_r = make_radar(breakdown, job["chosen_dc"])
                    st.pyplot(fig_r); plt.close(fig_r)

            with chart_right:
                st.markdown("#### 📊 Score Breakdown Bar Chart")
                if in_pool_bd:
                    n_srv     = len(in_pool_bd)
                    bar_width = 0.18
                    x         = np.arange(len(metrics_order))
                    colors    = cm.Set2(np.linspace(0, 1, n_srv))

                    fig, ax = plt.subplots(figsize=(9, 4))
                    fig.patch.set_facecolor("#0f1117")
                    ax.set_facecolor("#0f1117")

                    for i, (dc, bd) in enumerate(in_pool_bd.items()):
                        vals = [bd[m] for m in metrics_order]
                        bars = ax.bar(
                            x + i * bar_width, vals, bar_width,
                            label=dc, color=colors[i],
                            edgecolor="#0f1117", linewidth=0.5,
                        )
                        if dc == job["chosen_dc"]:
                            for bar in bars:
                                bar.set_edgecolor("#2ea043")
                                bar.set_linewidth(2)

                    ax.set_xticks(x + bar_width * (n_srv - 1) / 2)
                    ax.set_xticklabels(metric_labels, fontsize=9, color="#8b949e")
                    ax.set_ylabel("Score (0–1)", color="#8b949e")
                    ax.set_ylim(0, 1.15)
                    ax.tick_params(colors="#8b949e")
                    for spine in ax.spines.values(): spine.set_color("#21262d")
                    ax.axhline(y=1.0, color="#21262d", linestyle="--", linewidth=0.8)
                    ax.legend(loc="upper right", fontsize=8, framealpha=0,
                              labelcolor="#e6edf3")
                    ax.set_title(
                        f"In-Pool Breakdown — Job {job['job_id']} (chosen: {job['chosen_dc']})",
                        color="#e6edf3", fontsize=10,
                    )
                    st.pyplot(fig); plt.close(fig)

            # ── Full breakdown table ─────────────────────────────
            st.markdown("#### 📋 Full Breakdown Table")
            rows = []
            for dc, bd in breakdown.items():
                row = {
                    "Server":   dc,
                    "In Pool":  "✅" if bd.get("in_pool") else "⛔",
                    "Selected": "🏆" if dc == job["chosen_dc"] else "",
                }
                row.update({lbl: f"{bd[m]:.3f}" for lbl, m in zip(metric_labels, metrics_order)})
                row["Final"] = f"{bd['final']:.3f}"
                rows.append(row)
            df_bd = pd.DataFrame(rows).set_index("Server")
            st.dataframe(df_bd, use_container_width=True)

            if out_pool_bd:
                st.caption(f"⛔ Tier-excluded servers: {', '.join(out_pool_bd.keys())}")

        # ── Weight bar ───────────────────────────────────────────
        st.markdown("#### ⚖️ Priority Weights Applied")
        weight_map = {
            "green":       {"Performance": 0.10, "Cost": 0.25, "CO₂": 0.40, "Latency": 0.25},
            "balanced":    {"Performance": 0.25, "Cost": 0.25, "CO₂": 0.25, "Latency": 0.25},
            "performance": {"Performance": 0.40, "Cost": 0.10, "CO₂": 0.10, "Latency": 0.40},
        }
        w = weight_map.get(p, {})
        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        fig2.patch.set_facecolor("#0f1117")
        ax2.set_facecolor("#0f1117")
        bar_colors = ["#2ea043", "#58a6ff", "#e74c3c", "#f39c12"]
        bars = ax2.barh(list(w.keys()), list(w.values()), color=bar_colors, height=0.5)
        ax2.set_xlim(0, 0.55)
        ax2.set_xlabel("Weight", color="#8b949e")
        ax2.set_title(f"Weights for '{p}' priority", color="#e6edf3", fontsize=10)
        ax2.tick_params(colors="#8b949e")
        for spine in ax2.spines.values(): spine.set_color("#21262d")
        for i, v in enumerate(w.values()):
            ax2.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, color="#e6edf3")
        st.pyplot(fig2); plt.close(fig2)

        # ── Tier routing explanation ─────────────────────────────
        st.markdown("#### 🎯 Tier Pool Routing Logic")
        st.info(
            "**Why tier pools?** Without pools the strongest server wins every metric "
            "after min-max normalisation — perf, latency, and often cost. Tier pools "
            "enforce sustainability: 🌿 green jobs run on efficient small servers "
            "(lowest CO₂ + cost), ⚡ performance jobs get powerful large servers, and "
            "⚖️ balanced jobs use mid-tier hardware. Within each pool the best server "
            "wins on the metrics that matter for that priority."
        )
    else:
        st.info("Run the scheduler to see XAI explanations")

# ═════════════════════════════════════════════════════════════════
# COMPARE TAB  (new)
# ═════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("🔍 Side-by-Side Job Comparison")

    all_jobs = st.session_state.history
    if len(all_jobs) < 2:
        st.info("You need at least 2 scheduled jobs to compare. Run the scheduler first.")
    else:
        job_labels = [
            f"Job {j['job_id']} → {j['chosen_dc']} ({j['priority']})"
            for j in all_jobs
        ]
        ca, cb = st.columns(2)
        with ca:
            sel_a = st.selectbox("Job A", job_labels, index=0, key="cmp_a")
        with cb:
            sel_b = st.selectbox("Job B", job_labels, index=1, key="cmp_b")

        job_a = all_jobs[job_labels.index(sel_a)]
        job_b = all_jobs[job_labels.index(sel_b)]

        def cmp_metric(label, va, vb, lower_is_better=False):
            better = (va < vb) if lower_is_better else (va > vb)
            ca_str = f"🟢 {va}" if better  else f"{va}"
            cb_str = f"🟢 {vb}" if not better else f"{vb}"
            return {"Metric": label, "Job A": ca_str, "Job B": cb_str}

        st.markdown("---")
        col_a, col_b = st.columns(2)

        for col, job, label in [(col_a, job_a, "A"), (col_b, job_b, "B")]:
            p  = job["priority"]
            em = PRIORITY_EMOJI.get(p, "•")
            clr = PRIORITY_COLOR.get(p, "#8b949e")
            with col:
                st.markdown(f"#### Job {label}: Job {job['job_id']}")
                st.markdown(
                    f'<span style="color:{clr}">{em} {p}</span> → **{job["chosen_dc"]}**',
                    unsafe_allow_html=True,
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Reward",    f"{job['reward']:.3f}")
                m2.metric("Energy",    f"{job['power_kwh']:.4f} kWh")
                m3.metric("CO₂",       f"{job['co2']:.5f}")

                state = job["state"]
                st.caption(
                    f"Load {state['load']:.3f} · Carbon {state['carbon_factor']:.4f} "
                    f"· {state['load_type']} · {state['day_of_week']}"
                )

                # Radar
                if "score_breakdown" in job:
                    fig_r = make_radar(job["score_breakdown"], job["chosen_dc"])
                    st.pyplot(fig_r); plt.close(fig_r)

        # ── Delta summary table ──────────────────────────────────
        st.markdown("---")
        st.markdown("#### Δ Metric Differences (A − B)")
        delta_rows = []
        for label, ka, kb, lower in [
            ("Reward",      "reward",    "reward",    False),
            ("Energy (kWh)","power_kwh", "power_kwh", True),
            ("Cost",        "cost",      "cost",       True),
            ("CO₂",         "co2",       "co2",        True),
        ]:
            va, vb = job_a[ka], job_b[kb]
            diff   = va - vb
            better = (diff < 0) if lower else (diff > 0)
            delta_rows.append({
                "Metric":   label,
                "Job A":    f"{va:.5f}",
                "Job B":    f"{vb:.5f}",
                "Δ (A−B)":  f"{'▲' if diff > 0 else '▼'} {abs(diff):.5f}",
                "Winner":   "A" if (better) else "B",
            })
        st.dataframe(pd.DataFrame(delta_rows).set_index("Metric"), use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# ANALYTICS TAB
# ═════════════════════════════════════════════════════════════════
with tab_analytics:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        # ── Filter bar ───────────────────────────────────────────
        with st.expander("🔽 Filter Results", expanded=False):
            f1, f2 = st.columns(2)
            prio_filter = f1.multiselect(
                "Priority", ["green", "balanced", "performance"],
                default=["green", "balanced", "performance"],
            )
            lat_filter = f2.multiselect(
                "Latency", ["low", "medium", "high"],
                default=["low", "medium", "high"],
            )
            df = df[df["priority"].isin(prio_filter) & df["latency"].isin(lat_filter)]

        if df.empty:
            st.warning("No jobs match the selected filters.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Jobs",  len(df))
            k2.metric("Avg Reward",  f"{df['reward'].mean():.3f}")
            k3.metric("Total kWh",   f"{df['power_kwh'].sum():.4f}")
            k4.metric("Total CO₂",   f"{df['co2'].sum():.6f} t")

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Server Usage Distribution")
                usage = df["chosen_dc"].value_counts().sort_index()
                st.bar_chart(usage)
                used   = set(usage.index)
                unused = set(all_server_ids) - used if all_server_ids else set()
                if unused:
                    st.warning(f"⚠️ No jobs sent to: {', '.join(sorted(unused))}")
                else:
                    st.success(f"✅ All {len(all_server_ids)} servers received jobs")

            with col2:
                st.subheader("Average Reward per Server")
                st.bar_chart(df.groupby("chosen_dc")["reward"].mean().sort_index())

            # ── Reward over time ─────────────────────────────────
            st.subheader("Reward Over Time")
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
            ax.plot(df.index, df["reward"], marker="o", markersize=4,
                    linewidth=1.5, color="#2ea043")
            ax.fill_between(df.index, df["reward"], alpha=0.15, color="#2ea043")

            # Rolling average
            if len(df) >= 5:
                roll = df["reward"].rolling(5).mean()
                ax.plot(df.index, roll, linewidth=1, color="#58a6ff",
                        linestyle="--", label="5-job rolling avg")
                ax.legend(fontsize=8, framealpha=0, labelcolor="#e6edf3")

            ax.set_xlabel("Job Index", color="#8b949e")
            ax.set_ylabel("Reward",    color="#8b949e")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values(): spine.set_color("#21262d")
            st.pyplot(fig); plt.close(fig)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Priority → Server Distribution (%)")
                pivot_p = pd.crosstab(df["priority"], df["chosen_dc"], normalize="index") * 100
                st.bar_chart(pivot_p)
            with col4:
                st.subheader("Latency → Server Distribution (%)")
                pivot_l = pd.crosstab(df["latency"], df["chosen_dc"], normalize="index") * 100
                st.bar_chart(pivot_l)

            col5, col6 = st.columns(2)
            with col5:
                st.subheader("Carbon Intensity Over Time")
                carbon = [j["state"]["carbon_factor"] for j in st.session_state.history
                          if j["priority"] in prio_filter and j["latency"] in lat_filter]
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
                ax.plot(carbon, marker="o", markersize=3, linewidth=1, color="#e74c3c")
                ax.axhline(0.33, color="#2ea043", linewidth=0.8, linestyle="--", label="Low threshold")
                ax.axhline(0.66, color="#d29922", linewidth=0.8, linestyle="--", label="High threshold")
                ax.set_ylabel("CO₂ Intensity", color="#8b949e")
                ax.tick_params(colors="#8b949e")
                for spine in ax.spines.values(): spine.set_color("#21262d")
                ax.legend(fontsize=7, framealpha=0, labelcolor="#e6edf3")
                st.pyplot(fig); plt.close(fig)
            with col6:
                st.subheader("Energy Intensity Distribution")
                energy = [j["state"]["energy_factor"] for j in st.session_state.history
                          if j["priority"] in prio_filter and j["latency"] in lat_filter]
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
                ax.hist(energy, bins=20, color="#3498db", edgecolor="#0f1117")
                ax.set_xlabel("Energy Intensity", color="#8b949e")
                ax.set_ylabel("Frequency",        color="#8b949e")
                ax.tick_params(colors="#8b949e")
                for spine in ax.spines.values(): spine.set_color("#21262d")
                st.pyplot(fig); plt.close(fig)

            st.subheader("Energy & CO₂ by Server")
            st.bar_chart(df.groupby("chosen_dc")[["power_kwh", "co2"]].sum())

            # ── Full history table ───────────────────────────────
            st.subheader("Full Scheduling History")
            hist_display = df[[
                "job_id", "chosen_dc", "priority", "latency",
                "reward", "power_kwh", "cost", "co2",
            ]]
            st.dataframe(hist_display, use_container_width=True, hide_index=True)

            # Export filtered
            st.download_button(
                "⬇ Export Filtered Results (CSV)",
                data=export_csv(
                    [j for j in st.session_state.history
                     if j["priority"] in prio_filter and j["latency"] in lat_filter]
                ),
                file_name="greenmind_filtered.csv",
                mime="text/csv",
            )
    else:
        st.info("No analytics yet — run scheduler after adding jobs")

# ═════════════════════════════════════════════════════════════════
# SUSTAINABILITY TAB  (new)
# ═════════════════════════════════════════════════════════════════
with tab_sustain:
    st.subheader("🌍 Sustainability Dashboard")

    if not st.session_state.history:
        st.info("Run the scheduler to see sustainability metrics")
    else:
        h = st.session_state.history

        # ── Sustainability Score gauge ───────────────────────────
        ss = sustainability_score(h)
        green_jobs = [j for j in h if j["priority"] == "green"]
        total_co2  = sum(j["co2"] for j in h)
        total_kwh  = sum(j["power_kwh"] for j in h)

        # Estimate CO₂ avoided vs "all performance" baseline
        baseline_co2 = sum(
            j["co2"] * (1 + (1 - j["state"]["carbon_factor"])) for j in h
        )
        co2_avoided = max(0, baseline_co2 - total_co2)

        g1, g2, g3, g4 = st.columns(4)

        g1.markdown(f"""
        <div class="metric-card">
          <div class="label">Sustainability Score</div>
          <div class="value" style="color:{'#2ea043' if ss>=70 else '#d29922' if ss>=40 else '#f85149'}">{ss}</div>
          <div class="unit">/ 100</div>
        </div>""", unsafe_allow_html=True)

        g2.markdown(f"""
        <div class="metric-card">
          <div class="label">CO₂ Avoided (est.)</div>
          <div class="value">{co2_avoided:.5f}</div>
          <div class="unit">tCO2 vs all-performance baseline</div>
        </div>""", unsafe_allow_html=True)

        g3.markdown(f"""
        <div class="metric-card">
          <div class="label">Green Job Share</div>
          <div class="value">{len(green_jobs)/len(h)*100:.0f}%</div>
          <div class="unit">{len(green_jobs)} of {len(h)} jobs</div>
        </div>""", unsafe_allow_html=True)

        g4.markdown(f"""
        <div class="metric-card">
          <div class="label">Total Energy Used</div>
          <div class="value">{total_kwh:.3f}</div>
          <div class="unit">kWh</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("---")

        # ── CO₂ cumulative savings over time ─────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Cumulative CO₂ Over Time")
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
            cumco2 = np.cumsum([j["co2"] for j in h])
            ax.plot(cumco2, color="#e74c3c", linewidth=2)
            ax.fill_between(range(len(cumco2)), cumco2, alpha=0.15, color="#e74c3c")
            ax.set_xlabel("Job Index",  color="#8b949e")
            ax.set_ylabel("Cumulative tCO2", color="#8b949e")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values(): spine.set_color("#21262d")
            st.pyplot(fig); plt.close(fig)

        with col_r:
            st.markdown("#### CO₂ by Priority")
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
            co2_by_p = {}
            for j in h:
                co2_by_p[j["priority"]] = co2_by_p.get(j["priority"], 0) + j["co2"]
            prios  = list(co2_by_p.keys())
            values = list(co2_by_p.values())
            colors = [PRIORITY_COLOR.get(p, "#8b949e") for p in prios]
            bars = ax.bar(prios, values, color=colors, width=0.5)
            ax.set_ylabel("Total CO₂ (tCO2)", color="#8b949e")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values(): spine.set_color("#21262d")
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
                        f"{v:.5f}", ha="center", fontsize=8, color="#e6edf3")
            st.pyplot(fig); plt.close(fig)

        # ── Energy efficiency by server ───────────────────────────
        st.markdown("#### Energy Efficiency by Server (kWh per job)")
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
        df_h = pd.DataFrame(h)
        eff   = df_h.groupby("chosen_dc")["power_kwh"].mean().sort_values()
        colors_eff = ["#2ea043" if v == eff.min() else "#58a6ff" for v in eff.values]
        ax.bar(eff.index, eff.values, color=colors_eff, width=0.6)
        ax.set_ylabel("Avg kWh/job", color="#8b949e")
        ax.tick_params(colors="#8b949e", axis="x", rotation=30)
        for spine in ax.spines.values(): spine.set_color("#21262d")
        ax.set_title("Most efficient server highlighted in green", color="#8b949e", fontsize=9)
        st.pyplot(fig); plt.close(fig)

        # ── Carbon intensity heatmap (load type × day) ───────────
        st.markdown("#### Carbon Intensity Heatmap (Load Type × Day)")
        try:
            states_df = pd.DataFrame([j["state"] for j in h])
            states_df["carbon_factor"] = states_df["carbon_factor"].astype(float)
            pivot = states_df.pivot_table(
                values="carbon_factor",
                index="load_type",
                columns="day_of_week",
                aggfunc="mean",
            )
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
                im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                               vmin=0, vmax=1)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, color="#8b949e", rotation=30, fontsize=8)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, color="#8b949e", fontsize=8)
                plt.colorbar(im, ax=ax, label="Avg Carbon Factor")
                st.pyplot(fig); plt.close(fig)
        except Exception:
            st.caption("Not enough variety in states to show heatmap yet.")

        # ── Sustainability score band ─────────────────────────────
        st.markdown("---")
        band_color = "#2ea043" if ss >= 70 else "#d29922" if ss >= 40 else "#f85149"
        band_label = "Excellent 🌟" if ss >= 70 else "Good ✅" if ss >= 40 else "Needs Improvement ⚠️"
        advice = {
            "Excellent 🌟": "Your workloads are well optimised for sustainability. Keep scheduling green-priority jobs during high-carbon windows.",
            "Good ✅": "Consider shifting more jobs to 'green' priority and scheduling during low carbon-intensity periods.",
            "Needs Improvement ⚠️": "A high proportion of performance jobs or high-carbon scheduling periods is raising your footprint. Use the carbon-aware suggestion in the sidebar.",
        }[band_label]
        st.markdown(
            f'<div style="background:#161b22; border:2px solid {band_color}; border-radius:10px; '
            f'padding:16px 20px;">'
            f'<div style="font-size:1.3rem; font-weight:700; color:{band_color}">'
            f'Sustainability Rating: {band_label}</div>'
            f'<div style="color:#8b949e; margin-top:6px; font-size:0.85rem">{advice}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"GreenMind AI Scheduler · Backend: {API_BASE} · "
    f"{n_servers} servers · Tier-pool routing · "
    f"Steel Industry + Google Cluster datasets"
)
