__version__ = "0.0.0"

__all__ = ["go", "nmgo"]

from .myopic.algorithm import go
from .nonmyopic.algorithm import nmgo
