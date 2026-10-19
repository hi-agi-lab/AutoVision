# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Generic name -> object registry with conflict detection.

Used for methods, optimizers, dataset formats, and task models. Third-party
plugins can merge into a registry via entry points later; the in-process
registry is the single mechanism.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps names to objects with conflict-checked registration."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}
        self._hidden: set[str] = set()

    def register(
        self, *names: str, include_in_list: bool = True
    ) -> Callable[[T], T]:
        """Decorator registering the decorated object under one or more names."""

        def decorator(item: T) -> T:
            for name in names:
                if name in self._items:
                    raise ValueError(
                        f"Duplicate registration in {self.name} registry: "
                        f"'{name}' is already registered."
                    )
                self._items[name] = item
                if not include_in_list:
                    self._hidden.add(name)
            return item

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.name}: '{name}'. Available: {self.list_names(include_hidden=True)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def list_names(self, include_hidden: bool = False) -> list[str]:
        names = set(self._items)
        if not include_hidden:
            names -= self._hidden
        return sorted(names)

    def items(self, include_hidden: bool = False) -> Iterator[tuple[str, T]]:
        for name in self.list_names(include_hidden=include_hidden):
            yield name, self._items[name]

    def __len__(self) -> int:
        return len(self._items)
