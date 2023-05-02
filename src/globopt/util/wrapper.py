"""Implementation of OpenAI's gym-like wrappers for, e.g., problems."""

from typing import Any, Generic, TypeVar

WrappedInstance = TypeVar("WrappedInstance")


class Wrapper(Generic[WrappedInstance]):
    """Generic class for wrapping."""

    def __init__(self, to_wrap: WrappedInstance) -> None:
        """Instantiate the wrapper around an instance to be wrapped.

        Parameters
        ----------
        to_wrap : WrappedInstance
            Class instance to be wrapped.
        """
        self.wrapped = to_wrap

    @property
    def unwrapped(self) -> WrappedInstance:
        """'Returns the original wrapped instance."""
        return getattr(self.wrapped, "unwrapped", self.wrapped)

    def __getattr__(self, name: str) -> Any:
        """Reroutes attributes to the wrapped instance."""
        if name.startswith("_"):
            raise AttributeError(f"Accessing private attribute '{name}' is prohibited.")
        return getattr(self.wrapped, name)

    def __str__(self) -> str:
        """Returns the wrapped instance string."""
        return f"<{self.__class__.__name__}{self.wrapped.__str__()}>"

    def __repr__(self) -> str:
        """Returns the wrapped instance representation."""
        return f"<{self.__class__.__name__}{self.wrapped.__repr__()}>"
