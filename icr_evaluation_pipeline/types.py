from typing import Type, Any

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


def Triple(type1: Type, type2: Type, type3: Type) -> DagsterType:
    def type_check(_, value: Any) -> bool:  # noqa: ANN001, ANN401
        return (
            isinstance(value, tuple)
            and len(value) == 3
            and isinstance(value[0], type1)
            and isinstance(value[1], type2)
            and isinstance(value[2], type3)
        )

    name = f"Triple[{type1.__name__}, {type2.__name__}, {type3.__name__}]"
    description = (
        f"Tuple containing ({type1.__name__}, {type2.__name__}, {type3.__name__})."
    )

    return DagsterType(type_check_fn=type_check, name=name, description=description)
