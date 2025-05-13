from typing import Type, Any, Dict, Iterator

import pandas as pd
from dagster import DagsterType

_type_cache: Dict[str, DagsterType] = {}


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
    type_name = f"Triple[{type1.__name__}, {type2.__name__}, {type3.__name__}]"

    # Check cache first to avoid creating duplicate types
    if type_name in _type_cache:
        return _type_cache[type_name]

    def type_check(_, value: Any) -> bool:  # noqa: ANN001, ANN401
        return (
            isinstance(value, tuple)
            and len(value) == 3
            and isinstance(value[0], type1)
            and isinstance(value[1], type2)
            and isinstance(value[2], type3)
        )

    dagster_type = DagsterType(
        type_check_fn=type_check,
        name=type_name,
        description=f"Tuple containing ({type1.__name__}, {type2.__name__}, {type3.__name__}).",
    )

    _type_cache[type_name] = dagster_type
    return dagster_type


def is_iterator(_, value: object) -> bool:  # noqa: ANN001
    return isinstance(value, Iterator)


IteratorType = DagsterType(
    type_check_fn=is_iterator,
    name="IteratorType",
    description="An iterator.",
)
