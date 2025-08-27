from uuid import uuid4
from typing import Any, Type, TypeVar
from collections import defaultdict

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def generate_short_name() -> str:
    return str(uuid4())


def sum_or_none(elements):
    if all(el is None for el in elements):
        return None
    return sum(el for el in elements if el is not None)


def _get_tuple_of_values_from_object(obj: BaseModel, params: set[str]) -> tuple[Any, ...]:
    return tuple(
        tuple(value) if isinstance(value, list) else value
        for key in params
        for value in [getattr(obj, key)]
    )


def group_objects_excluding_fields(objects: list[T], fields: set) -> dict[str, list[T]]:
    """Group objects by the specified fields."""

    value_mapper = defaultdict(list)
    obj_type = type(objects[0])
    for obj in objects:
        key = _get_tuple_of_values_from_object(obj, obj_type.model_fields.keys() - fields)
        value_mapper[repr(key)].append(obj)
    return value_mapper


def get_extra_param_values(
    class_type: Type[BaseModel], objects: list[BaseModel], params_to_aggregate: set[str]
) -> dict[str, Any]:
    other_params = class_type.model_fields.keys() - params_to_aggregate
    other_params_val_mapper = {}
    for key in other_params:
        first_val = getattr(objects[0], key)
        values = set(
            [
                tuple(getattr(obj, key)) if isinstance(first_val, list) else getattr(obj, key)
                for obj in objects
            ]
        )
        if len(values) > 1:
            raise NotImplementedError(
                f"Aggregating {class_type=} with different {values=} for {key=} is not supported yet."
            )
        other_params_val_mapper[key] = first_val
    return other_params_val_mapper


def weighted_average_or_none(values, weights):
    # Filter out pairs where either value or weight is None
    filtered = [
        (v, w) for v, w in zip(values, weights, strict=False) if v is not None and w is not None
    ]

    if not filtered:
        return None

    total_weight = sum(w for _, w in filtered)
    if total_weight == 0:
        return None

    weighted_sum = sum(v * w for v, w in filtered)
    return weighted_sum / total_weight
