from dagster import Config


class KFoldConfig(Config):
    n_splits: int = 5  # TODO: Think about the value of n_splits
    shuffle: bool = True
    stratify: bool = True
    n_bins: int = 10


class PreprocessingConfig(Config):
    col_missing_value_threshold: float = (
        0.5  # Drop columns with more than this fraction of missing values
    )
    row_missing_value_threshold: float = (
        0.05  # Drop rows with more than this fraction of missing values
    )
