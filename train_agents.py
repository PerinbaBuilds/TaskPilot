"""
Multi-agent RL training for GreenMind cloud scheduler.

Each agent represents a data centre and learns to score incoming jobs
using linear function approximation (numpy) and TD-error weight updates.
"""
import os
from agents import Agent
from rl_env import CloudEnv
from config import get_weights

STEEL = os.path.join(os.path.dirname(__file__), "steel_industry_data.csv")
BORG  = None  # set to borg_traces_data.csv path if available

EPISODES = 500
MODE     = "balanced"  # "green" | "balanced" | "performance"


def train():
    agents = {
        "DC_A": Agent("DC_A"),
        "DC_B": Agent("DC_B"),
        "DC_C": Agent("DC_C"),
    }

    env     = CloudEnv(STEEL, BORG)
    weights = get_weights(MODE)

    print(f"Training {len(agents)} agents for {EPISODES} episodes (mode={MODE})...\n")

    for episode in range(EPISODES):
        state = env.reset()

        scores = {name: agent.act(state) for name, agent in agents.items()}
        chosen = max(scores, key=scores.get)

        _, reward = env.step(scores[chosen], weights)

        for agent in agents.values():
            agent.learn(state, reward)

        if (episode + 1) % 100 == 0:
            score_str = "  ".join(f"{n}={s:.3f}" for n, s in scores.items())
            print(f"Episode {episode + 1:4d} | chosen={chosen} | reward={reward:.4f} | {score_str}")

    print("\nTraining complete.")
    return agents


if __name__ == "__main__":
    train()
