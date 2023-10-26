from __future__ import annotations

import inspect
import warnings
from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

from polars.exceptions import PolarsExperimentalWarning
from polars.utils.various import find_stacklevel

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    P = ParamSpec("P")
    T = TypeVar("T")


def issue_experimental_warning(message: str, *, version: str) -> None:
    """
    Issue a deprecation warning.

    Parameters
    ----------
    message
        The message associated with the warning.
    version
        The Polars version number in which the warning is first issued.
        This argument is used to help developers determine when to remove the
        deprecated functionality.

    """
    warnings.warn(message, PolarsExperimentalWarning, stacklevel=find_stacklevel())


def experimental(version: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to mark a function as experimental."""

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:

            issue_experimental_warning(
                f"`{function.__name__}` is experimental functionality.",
                version=version,
            )
            return function(*args, **kwargs)

        wrapper.__doc__ = _insert_docstring_experimental_warning(function.__doc__)
        wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]
        return wrapper

    return decorate


def _insert_docstring_experimental_warning(doc: str) -> str:
    split = doc.find("\n\n")
    print(split)
    warning = "\n\n        .. warning::\n            This functionality is experimental."
    new = doc[:split] + warning + doc[split:]
    return new
