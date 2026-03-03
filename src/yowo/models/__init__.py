"""Public surface for the models package.

Exports:
    ModelMeta      — metadata dataclass for a registered model variant.
    get            — look up ModelMeta by (ModelFamily, ModelSize).
    get_cls        — look up classification ModelMeta by (ModelFamily, ModelSize).
    list_available — return all registered ModelMeta entries.
    register       — add a new variant to the registry at runtime.
    resolve_weights — resolve a ModelSpec to a local .pt file path.
"""

from yowo.models._registry import ModelMeta, get, get_cls, list_available, register
from yowo.models._weights import resolve_weights

__all__ = [
    "ModelMeta",
    "get",
    "get_cls",
    "list_available",
    "register",
    "resolve_weights",
]
