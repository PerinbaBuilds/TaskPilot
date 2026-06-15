from data_loader import DatasetLoader


class EnergyModel:
    """Thin wrapper around DatasetLoader for energy-only sampling."""

    def __init__(self, steel_path, borg_path=None):
        self.data = DatasetLoader(steel_path, borg_path)

    def sample(self) -> dict:
        green, cost = self.data.sample_energy()
        return {
            "green_ratio": green,
            "energy_cost": cost,
        }
