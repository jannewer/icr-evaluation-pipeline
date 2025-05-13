from dagster import Config


class KFoldConfig(Config):
    seed: int = 1
    n_splits: int = 5
    stratify: bool = True
    n_bins: int = 10
