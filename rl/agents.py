import numpy as np


class TierAgent:
    """
    RL agent that selects a server within a priority tier pool.

    Uses linear function approximation with one weight vector per server
    (multi-action Q-learning). The agent learns to balance load distribution
    while optimising for tier-specific objectives (CO2, cost, performance).

    State:  [server0_load_norm, server1_load_norm, server2_load_norm,
             carbon_factor, energy_price, job_cpu, job_mem]
    Action: index of server to assign (0, 1, or 2 within tier pool)
    Reward: quality score − overload penalty
    """

    N_FEATURES = 7

    def __init__(self, priority: str, n_servers: int = 3,
                 lr: float = 0.05, gamma: float = 0.9, epsilon: float = 0.15):
        self.priority  = priority
        self.n_servers = n_servers
        self.lr        = lr
        self.gamma     = gamma
        self.epsilon   = epsilon        # exploration rate (decays over time)
        self.steps     = 0
        # One weight row per server
        self.W = np.zeros((n_servers, self.N_FEATURES), dtype=np.float32)

    def build_features(self, server_loads: list[float], carbon: float,
                       energy_price: float, job_cpu: float, job_mem: float) -> np.ndarray:
        """Normalise inputs into a fixed-length feature vector."""
        max_load = max(server_loads) + 1e-9
        norm = [l / max_load for l in server_loads[:3]]
        while len(norm) < 3:
            norm.append(0.0)
        return np.array([
            *norm,
            1.0 - carbon,           # high = clean energy
            energy_price,
            job_cpu,
            job_mem,
        ], dtype=np.float32)

    def q_values(self, feat: np.ndarray) -> np.ndarray:
        return self.W @ feat

    def act(self, feat: np.ndarray, explore: bool = False) -> int:
        """ε-greedy action. Returns server index within pool."""
        eps = self.epsilon * max(0.1, 1.0 - self.steps / 2000)
        if explore and np.random.rand() < eps:
            return int(np.random.randint(self.n_servers))
        return int(np.argmax(self.q_values(feat)))

    def learn(self, feat: np.ndarray, action: int, reward: float,
              next_feat: np.ndarray | None = None) -> float:
        """TD(0) update. Returns td_error for logging."""
        q_now = float(np.dot(self.W[action], feat))
        if next_feat is not None:
            q_next = float(np.max(self.q_values(next_feat)))
            target = reward + self.gamma * q_next
        else:
            target = reward
        td = target - q_now
        self.W[action] += self.lr * td * feat
        self.steps += 1
        return td
