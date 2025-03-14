from dagster import Config


class TrainAndTestDatasetConfig(Config):
    seed: int = 1
    train_size: float = 0.8
