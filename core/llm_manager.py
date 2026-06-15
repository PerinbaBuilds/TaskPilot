def explain(dc: str, green: float, load: float, cost: float, priority: str) -> str:
    reasons = []

    if priority == "green":
        reasons.append("renewable energy optimization")
    elif priority == "performance":
        reasons.append("low-latency execution")
    else:
        reasons.append("balanced energy-performance tradeoff")

    reasons.append("high green energy" if green >= 0.6 else "low green energy")
    reasons.append("low load"          if load  <  0.5 else "high load")
    reasons.append("low cost"          if cost  < 10   else "high cost")

    return (
        f"Job assigned to {dc} due to {', '.join(reasons)}. "
        f"State: green_ratio={green}, load={load}, cost={cost}, priority={priority}."
    )


def compare_and_explain(scores: dict, state, priority: str) -> dict:
    green, load, _job_pressure, cost = state
    explanations = {}

    for dc, score in scores.items():
        reasons = [
            "good green energy" if green >= 0.6 else "low green energy",
            "low load"          if load  <  0.5 else "high load",
            "low cost"          if cost  < 10   else "high cost",
            f"priority={priority}",
        ]
        explanations[dc] = f"{dc} score={round(score, 2)} | " + ", ".join(reasons)

    return explanations
