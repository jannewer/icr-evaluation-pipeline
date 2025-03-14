import pandas as pd
from dagster import DagsterType


def is_dataframe_tuple(_, value: object) -> bool:  # noqa: ANN001
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], pd.DataFrame)
        and isinstance(value[1], pd.DataFrame)
    )


DataFrameTuple = DagsterType(
    type_check_fn=is_dataframe_tuple,
    name="DataFrameTuple",
    description="Tuple containing exactly two pandas DataFrames.",
)
