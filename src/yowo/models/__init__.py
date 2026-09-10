"""Public surface for the models package.

Exports:
    ModelMeta      — metadata dataclass for a registered model variant.
    get            — look up ModelMeta by (ModelFamily, ModelSize).
    get_cls        — look up classification ModelMeta by (ModelFamily, ModelSize).
    get_obb        — look up OBB ModelMeta by (ModelFamily, ModelSize).
    get_for_task   — look up ModelMeta by task, family and size (one dispatch).
    list_available — return all registered DETECTION ModelMeta entries.
    list_all_registered — return every entry in EVERY registry as (task, meta).
    registered_tasks — the task names that select a registry.
    register       — add a new variant to the registry at runtime.
    resolve_weights — resolve a ModelSpec to a local .pt file path.
"""

from yowo.models._registry import (
    ModelMeta,
    get,
    get_cls,
    get_for_task,
    get_obb,
    list_all_registered,
    list_available,
    register,
    registered_tasks,
)
from yowo.models._weights import resolve_weights

__all__ = [
    "ModelMeta",
    "get",
    "get_cls",
    "get_for_task",
    "get_obb",
    "list_all_registered",
    "list_available",
    "register",
    "registered_tasks",
    "resolve_weights",
]
