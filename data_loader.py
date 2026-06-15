import pandas as pd
import numpy as np


class DatasetLoader:
    """Loads Steel Industry energy data and Borg workload traces for RL training."""

    def __init__(self, steel_path, borg_path=None):
        self.steel = pd.read_csv(steel_path)
        self._normalise_steel()

        if borg_path:
            self.borg = pd.read_csv(borg_path)
        else:
            self.borg = None

    def _normalise_steel(self):
        ukwh_min = self.steel["Usage_kWh"].min()
        ukwh_rng = self.steel["Usage_kWh"].max() - ukwh_min + 1e-9
        co2_min  = self.steel["CO2(tCO2)"].min()
        co2_rng  = self.steel["CO2(tCO2)"].max() - co2_min + 1e-9
        self.steel["norm_energy"] = (self.steel["Usage_kWh"] - ukwh_min) / ukwh_rng
        self.steel["norm_co2"]   = (self.steel["CO2(tCO2)"] - co2_min)  / co2_rng

    def sample_energy(self):
        row  = self.steel.sample(1).iloc[0]
        green = 1.0 - float(row["norm_co2"])      # higher = greener
        cost  = float(row["norm_energy"])
        return green, cost

    def sample_workload(self):
        if self.borg is not None:
            row  = self.borg.sample(1).iloc[0]
            load = float(row.get("mean_cpu_usage_rate", 0.5))
            job  = float(row.get("requested_ram", 0.5))
        else:
            load = np.random.uniform(0.2, 0.9)
            job  = np.random.uniform(0.1, 1.0)
        return load, job


class DatasetJobLoader:
    """Streams jobs from a CSV task file (used by the FastAPI scheduler)."""

    def __init__(self, task_file):
        self.df  = pd.read_csv(task_file)
        self.ptr = 0

    def has_next(self):
        return self.ptr < len(self.df)

    def next_job(self):
        row      = self.df.iloc[self.ptr]
        self.ptr += 1
        return {
            "cpu":      min(1.0, float(row.get("cpu",    0.5))),
            "memory":   min(1.0, float(row.get("memory", 0.5))),
            "latency":  self._latency_from_cpu(row),
            "priority": self._priority_from_size(row),
        }

    def _latency_from_cpu(self, row):
        cpu = float(row.get("cpu", 0))
        if cpu > 0.7:
            return "high"
        elif cpu > 0.4:
            return "medium"
        return "low"

    def _priority_from_size(self, row):
        if float(row.get("memory", 0)) > 0.7:
            return "performance"
        elif float(row.get("cpu", 0)) > 0.6:
            return "balanced"
        return "green"
