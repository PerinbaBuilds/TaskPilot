import numpy as np
from core.data_loader import DatasetLoader


class CloudEnv:
    """
    RL environment for sustainable cloud job scheduling.

    State vector: [green_score, system_load, job_size, energy_cost]
    Reward:       weighted combination of greenness, low load, and low cost
    """

    def __init__(self, steel_path, borg_path=None):
        self.data  = DatasetLoader(steel_path, borg_path)
        self.state = self._sample_state()

    def _sample_state(self) -> np.ndarray:
        green, cost = self.data.sample_energy()
        load, job   = self.data.sample_workload()
        return np.array([green, load, job, cost], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.state = self._sample_state()
        return self.state

    def step(self, score: float, weights: list) -> tuple[np.ndarray, float]:
        """
        Args:
            score:   agent's predicted utility for the current state
            weights: [w_green, w_perf, w_cost] from config.get_weights()

        Returns:
            (next_state, reward)
        """
        green, load, job, cost = self.state
        w_green, w_perf, w_cost = weights

        # Reward: higher green + lower cost + moderate load is better
        reward = float(
            w_green * green
            - w_cost  * cost
            - w_perf  * load
        )

        self.state = self._sample_state()
        return self.state, reward
