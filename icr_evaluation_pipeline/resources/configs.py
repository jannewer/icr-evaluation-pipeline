from dagster import Config


class KFoldConfig(Config):
    n_splits: int = 5  # TODO: Think about the value of n_splits
    shuffle: bool = True
    stratify: bool = True
    n_bins: int = 10


class PreprocessingConfig(Config):
    # Drop columns with more than this fraction of missing values
    col_missing_value_threshold: float = 0.5
    # Drop rows with missing values if max. this fraction of rows has missing values
    row_missing_value_threshold: float = 0.05
    # At least this fraction of samples should be covered by the top $num_categories_threshold categories
    coverage_threshold: float = 0.9
    # Max. this number of categories should be needed to cover $coverage_threshold percent of the samples
    num_categories_threshold: int = 20


class RarityScoreConfig(Config):
    rarity_measure: str = "cb_loop"
    n_neighbors: int = (
        None  # If None, the default value of the rarity measure will be used
    )
    min_rarity_score: float = 0.5
    cb_loop_extent: int = 2
    l2min_psi: int = 1


class ModelConfig(RarityScoreConfig):
    rarity_adjustment_method: str = "bootstrap_sampling"
