from dagster import Config


class KFoldConfig(Config):
    n_splits: int = 5  # TODO: Think about the value of n_splits
    shuffle: bool = True
    stratify: bool = True
    n_bins: int = 10
