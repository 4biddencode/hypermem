"""HyperMEM - AI memory system that never forgets what matters.

Usage:
    from hypermem import HyperMEM

    hm = HyperMEM()
    result = await hm.add_message("user", "My name is Emanuel")
    print(result.tagged)  # Newly tagged memory
    print(result.recalled)  # Relevant existing memories
"""

from .engine import HyperMEM
from .types import HyperMemConfig, HyperMem, RecallResult, AddMessageResult

__all__ = [
    "HyperMEM", "HyperMemConfig", "HyperMem", "RecallResult", "AddMessageResult",
]
__version__ = "1.0.0"
